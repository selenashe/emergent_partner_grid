"""Audit matched training, sampling exposure, rollouts, and maintained comparisons."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
from safetensors.numpy import load_file
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_record_paths


def load(path):
    return resolve_record_paths(json.loads(Path(path).read_text()))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_sampling(prefix, manifest, cache):
    record = load(str(prefix) + '_sampling_audit.json')
    path = Path(record['schedule_path'])
    assert digest(path) == record['schedule_sha256']
    if str(path) not in cache:
        with np.load(path) as data:
            cache[str(path)] = {k: data[k] for k in data.files}
    plan = cache[str(path)]
    with np.load(str(prefix) + '_sampling_audit.npz') as data:
        actual = {k: data[k] for k in data.files}
    pairs, n = plan['capability_pairs'], int(plan['n_layouts'])
    c, r = len(pairs), plan['layouts'].shape[1]
    assert n == 1096 and r == 20
    np.testing.assert_array_equal(actual['capability_pairs'], pairs)
    full, tail = divmod(record['allocated_episode_prefix'], c)
    common = np.bincount(plan['layouts'][:full].ravel(), minlength=n)
    planned = np.tile(common, (c, 1))
    allocated = np.full(c, full, dtype=np.int64)
    for profile in plan['profile_order'][full, :tail]:
        np.add.at(planned[profile], plan['layouts'][full], 1)
        allocated[profile] += 1
    np.testing.assert_array_equal(allocated, actual['episodes_allocated'])
    assert np.ptp(allocated) <= 1
    started, completed = planned.copy(), planned.copy()
    active, not_started = [], []
    ids = actual['active_episode_ids']
    assert len(ids) == len(np.unique(ids)) == 256
    assert ids.min() >= 0 and ids.max() < record['allocated_episode_prefix']
    for slot, episode in enumerate(ids):
        packet, offset = divmod(int(episode), c)
        profile = int(plan['profile_order'][packet, offset])
        np.testing.assert_array_equal(actual['active_capability'][slot], pairs[profile])
        active.append(profile)
        index, time = int(actual['active_round_idx'][slot]), int(actual['active_round_time'][slot])
        assert 0 <= index < r and 0 <= time < 100
        if index == 0 and time == 0:
            not_started.append(profile)
        np.add.at(started[profile], plan['layouts'][packet, index + int(time > 0):], -1)
        np.add.at(completed[profile], plan['layouts'][packet, index:], -1)
    np.testing.assert_array_equal(started, actual['rounds_started'])
    np.testing.assert_array_equal(completed, actual['rounds_completed'])
    np.testing.assert_array_equal(allocated - np.bincount(active, minlength=c), actual['episodes_completed'])
    np.testing.assert_array_equal(allocated - np.bincount(not_started, minlength=c), actual['episodes_started'])
    assert int(actual['environment_steps'].sum()) == record['collected_environment_steps'] == manifest['effective_timesteps_per_policy']
    assert int(started.sum()) == record['rounds_started']
    assert int(completed.sum()) == record['rounds_completed']
    assert np.all(np.count_nonzero(started, axis=1) == n)
    return dict(environment_steps=record['collected_environment_steps'], profiles=c,
                layouts_per_profile=n, allocated_profile_count_difference=int(np.ptp(allocated)),
                schedule_prefix_exact=True)


def audit_rollout(prefix, population, condition, summary):
    count = 480 if population == 'train' else 440
    profiles = 24 if population == 'train' else 22
    assert summary['n_episodes'] == count and summary['n_rounds'] == 20 * count
    with h5py.File(str(prefix) + '_' + population + '.h5') as f:
        assert f['dones'].shape == (count, 2000)
        done = f['dones'][:].astype(bool)
        assert done.any(axis=1).all()
        steps = done.argmax(axis=1) + 1
        alive = np.arange(2000)[None, :] < steps[:, None]
        rounds, success = f['round_done'][:].astype(bool), f['success'][:].astype(bool)
        assert np.all((rounds & alive).sum(axis=1) == 20)
        assert not np.any(success & alive & ~rounds)
        rewards = f['rewards'][:]
        assert np.isfinite(rewards[alive]).all()
        returns = (rewards * alive).sum(axis=1, dtype=np.float64)
        np.testing.assert_allclose(returns.mean(), summary['mean_ep_return'], atol=2e-4, rtol=0)
        rate = (success & alive).sum() / (20 * count)
        np.testing.assert_allclose(rate, summary['round_success_rate'], atol=1e-8, rtol=0)
        inferred = (1.01 * 20 * rate - summary['mean_ep_return']) / .01
        np.testing.assert_allclose(steps.mean(), inferred, atol=.02, rtol=0)
        initial = f['is_t0'][:].astype(bool) & alive
        assert np.all(initial.sum(axis=1) == 20)
        allocations, assignment, index = f['ego_alloc_action'][:], f['partner_assignment'][:], f['round_idx'][:]
        assert np.isin(allocations[initial], [1, 2]).all()
        assert np.all(allocations[alive & ~initial] == 0)
        assert np.isin(assignment[alive], [1, 2]).all()
        same_round = alive[:, 1:] & alive[:, :-1] & (index[:, 1:] == index[:, :-1])
        assert not np.any((assignment[:, 1:] != assignment[:, :-1]) & same_round)
        assert f['capability_pool'].shape == (profiles, 2)
        _, per_profile = np.unique(f['capability_index_per_ep'][:], return_counts=True)
        np.testing.assert_array_equal(per_profile, np.full(profiles, 20))
        if condition.startswith('rnn'):
            assert f['hidden_state'].shape == (count, 2000, 128)
        for round_index in range(20):
            rate_by_round = ((success & alive) & (index == round_index)).sum() / count
            np.testing.assert_allclose(rate_by_round, summary['per_round_success_rate'][round_index], atol=1e-7, rtol=0)
    return dict(episodes=count, rounds=20 * count, episode_return=float(returns.mean()),
                success=float(rate), episode_steps=float(steps.mean()), allocation_locked=True)


def audit(manifest_path):
    from bash.submit_v1_matched_training import verify
    manifest = verify(manifest_path)
    assert manifest['status'] == 'completed'
    checkpoints = Path(manifest['versions']['v1_matched']['checkpoint_root'])
    cache, training, rollouts = {}, [], []
    for job in manifest['training_jobs']:
        prefix = checkpoints / job['tag']
        config = load(str(prefix) + '_config.json')
        assert config == resolve_record_paths(yaml.safe_load(Path(job['config_path']).read_text()))
        weights = load_file(str(prefix) + '.safetensors')
        assert all(np.isfinite(tensor).all() for tensor in weights.values())
        exposure = audit_sampling(prefix, manifest, cache)
        training.append(dict(condition=job['condition'], seed=job['seed'], finite_parameters=True, **exposure))
        directory = Path(manifest['evaluation_inputs']['v1_matched'][str(job['seed'])])
        eval_prefix = directory / job['tag']
        summary = load(str(eval_prefix) + '_summary.json')
        for population in ('train', 'test'):
            measurements = audit_rollout(eval_prefix, population, job['condition'], summary[population])
            rollouts.append(dict(version='v1_matched', condition=job['condition'], seed=job['seed'],
                                 population=population, **measurements))
    for population, folder in (('train', 'familiar'), ('test', 'novel')):
        report = load(Path(manifest['aggregate_root']) / folder / 'comparison.json')
        assert set(report['trajectories']) == {'v1_matched', 'v2', 'v3'}
        assert report['seeds'] == manifest['evaluation_seeds']
        for condition in dict.fromkeys(j['condition'] for j in manifest['training_jobs']):
            series = [r for r in rollouts if r['condition'] == condition and r['population'] == population]
            entry = next(a for a in report['aggregates'] if a['version'] == 'v1_matched' and a['condition'] == condition)
            assert entry['n_seeds'] == 10
            for metric in ('episode_return', 'success', 'episode_steps'):
                values = np.array([r[metric] for r in series])
                tolerance = 2e-4 if metric == 'episode_return' else 1e-7
                np.testing.assert_allclose(values.mean(), entry[metric + '_mean'], atol=tolerance, rtol=0)
                np.testing.assert_allclose(values.std(ddof=1), entry[metric + '_sd'], atol=tolerance, rtol=0)
    output = Path(manifest['aggregate_root'])
    result = dict(manifest=str(manifest_path), verified_at_utc=datetime.now(timezone.utc).isoformat(),
                  passed=True, frozen_sources_and_configs_verified=True, trained_policies_verified=len(training),
                  full_budget_per_policy=manifest['effective_timesteps_per_policy'],
                  total_environment_steps=sum(r['environment_steps'] for r in training),
                  evaluation_rollouts_verified=len(rollouts), training_checks=training, rollout_checks=rollouts,
                  baseline_status='active', comparison_versions=['v1_matched', 'v2', 'v3'])
    (output / 'v1_matched_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Verified 40 matched policies, sampling exposure, and 80 evaluation rollouts.')
    print(output / 'v1_matched_verification.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    audit(args.manifest.resolve())
