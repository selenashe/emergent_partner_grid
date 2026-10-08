"""Profile grouping must preserve terminal rewards and independent seed weights."""
from pathlib import Path
import sys
import json

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval import plot_profile_performance as plots


@pytest.fixture
def rollout(tmp_path):
    path = tmp_path / 'rollout.h5'
    pool = np.array([[1, 4], [4, 1]], dtype=np.int32)
    # Deliberately interleave profiles instead of assuming contiguous episodes.
    indices = np.array([1, 0, 0, 1], dtype=np.int32)
    lengths = np.array([3, 4, 3, 4])
    alive = np.arange(6)[None, :] < lengths[:, None]
    dones = np.arange(6)[None, :] >= lengths[:, None] - 1
    round_done = np.zeros((4, 6), bool)
    round_done[:, 0] = True
    round_done[np.arange(4), lengths - 1] = True
    round_done[~alive] = True
    success = np.zeros((4, 6), bool)
    success[1:3, 0] = True
    success[np.arange(3), lengths[:3] - 1] = True
    success[~alive] = True
    rewards = np.full((4, 6), -.01, np.float32)
    rewards[success] = 1
    rewards[~alive] = 999  # padding cannot contribute to profile returns
    capability = np.repeat(pool[indices, None, :], 6, axis=1)
    capability[~alive] = -99
    round_idx = np.ones((4, 6), np.int32)
    round_idx[:, 0] = 0
    with h5py.File(path, 'w') as f:
        for key, value in dict(capability_pool=pool, capability_index_per_ep=indices,
                               dones=dones, round_done=round_done, success=success,
                               rewards=rewards, capability=capability, round_idx=round_idx).items():
            f[key] = value
    return path


def test_profile_labels_terminal_transition_and_padding(rollout):
    rows = plots.profile_metrics(rollout, [(1, 4), (4, 1)], episodes_per_profile=2, rounds_per_episode=2)
    assert [(r['delay_red'], r['delay_blue']) for r in rows] == [(1, 4), (4, 1)]
    assert [r['success'] for r in rows] == [1, .25]
    assert [r['episode_return'] for r in rows] == pytest.approx([1.985, .47])
    assert [r['episode_steps'] for r in rows] == [3.5, 3.5]
    assert [r['per_round_success'] for r in rows] == [[1, 1], [0, .5]]
    assert all(r['n_episodes'] == 2 and r['n_rounds'] == 4 for r in rows)


@pytest.mark.parametrize('problem', ['no_done', 'wrong_capability', 'duplicate_round', 'duplicate_profile', 'unequal_episodes'])
def test_malformed_profile_records_fail(rollout, problem):
    with h5py.File(rollout, 'r+') as f:
        if problem == 'no_done':
            f['dones'][0] = False
        elif problem == 'wrong_capability':
            f['capability'][0, 1] = [1, 4]
        elif problem == 'duplicate_round':
            f['round_idx'][0, 2] = 0
        elif problem == 'duplicate_profile':
            f['capability_pool'][1] = [1, 4]
        elif problem == 'unequal_episodes':
            f['capability_index_per_ep'][0] = 0
            f['capability'][0, :3] = np.array([[1, 4]] * 3)
    with pytest.raises(ValueError):
        plots.profile_metrics(rollout, [(1, 4), (4, 1)], episodes_per_profile=2, rounds_per_episode=2)


def policy_rows():
    rows = []
    for version in plots.VERSIONS:
        for seed in range(1, 11):
            value = float(seed > 5)
            for red, blue in [(1, 4), (4, 1)]:
                rows.append(dict(version=version, condition='condition', seed=seed,
                    delay_red=red, delay_blue=blue, success=value, episode_return=value,
                    episode_steps=10 + value, per_round_success=[value, value]))
    return rows


def test_all_ten_seed_profile_means_and_sample_sd():
    result = plots.aggregate_profiles(policy_rows(), list(range(1, 11)), [(1, 4), (4, 1)], ('condition',))
    assert len(result) == 4
    for row in result:
        assert row['n_seeds'] == 10
        assert row['success_mean'] == .5
        assert row['success_sd'] == pytest.approx(np.std([0] * 5 + [1] * 5, ddof=1))
        assert row['episode_steps_mean'] == 10.5
        assert row['per_round_success_mean'] == [.5, .5]


@pytest.mark.parametrize('problem', ['missing', 'duplicate'])
def test_incomplete_seed_profile_grid_cannot_be_averaged(problem):
    rows = policy_rows()
    if problem == 'missing':
        rows.pop()
    else:
        rows.append(rows[-1])
    with pytest.raises(ValueError, match='Missing, duplicate'):
        plots.aggregate_profiles(rows, list(range(1, 11)), [(1, 4), (4, 1)], ('condition',))


def test_split_profiles_reconstruct_pooled_summary_and_reject_mismatch(rollout):
    rows = plots.profile_metrics(rollout, [(1, 4), (4, 1)], 2, 2)
    summary = dict(round_success_rate=.625, mean_ep_return=1.2275,
                   n_episodes=4, n_rounds=8, per_round_success_rate=[.5, .75])
    plots.check_pooled_summary(rows, summary, rollout)
    summary['mean_ep_return'] = 999
    with pytest.raises(ValueError, match='does not reproduce pooled summary'):
        plots.check_pooled_summary(rows, summary, rollout)


def test_third_protocol_is_averaged_and_missing_v3_seed_is_rejected():
    rows = policy_rows()
    rows += [dict(row, version='v3', success=.75) for row in rows if row['version'] == 'v2']
    result = plots.aggregate_profiles(rows, list(range(1, 11)), [(1, 4), (4, 1)],
                                     ('condition',), versions=('v1', 'v2', 'v3'))
    assert len(result) == 6
    assert all(a['success_mean'] == .75 and a['n_seeds'] == 10 for a in result if a['version'] == 'v3')
    rows.pop()
    with pytest.raises(ValueError, match='Missing, duplicate'):
        plots.aggregate_profiles(rows, list(range(1, 11)), [(1, 4), (4, 1)],
                                 ('condition',), versions=('v1', 'v2', 'v3'))


def test_v3_manifest_with_only_v3_source_renders_all_protocol_profile_plots(tmp_path):
    source = tmp_path / 'frozen_v3'
    pool_file = source / 'jaxmarl/environments/coordination_grid/capability_populations.py'
    pool_file.parent.mkdir(parents=True)
    pool_file.write_text('TRAIN_CAPABILITY_PAIRS = TEST_CAPABILITY_PAIRS = [(1, 4), (4, 1)]\n')
    pool = np.array([[1, 4], [4, 1]], np.int32)
    indices = np.repeat(np.arange(2), 20)
    inputs = {}
    for version in ('v1', 'v2', 'v3'):
        inputs[version] = {}
        for seed in (1, 2):
            directory = tmp_path / 'inputs' / version / str(seed)
            directory.mkdir(parents=True)
            inputs[version][str(seed)] = str(directory)
            success = np.broadcast_to(np.arange(20) < seed * 5, (40, 20))
            rewards = np.where(success, 1., -.01)
            for condition in plots.CONDITIONS:
                prefix = directory / f'{condition}_seed{seed}'
                with h5py.File(str(prefix) + '_test.h5', 'w') as f:
                    for key, value in dict(capability_pool=pool, capability_index_per_ep=indices,
                            capability=np.repeat(pool[indices, None, :], 20, axis=1),
                            dones=np.broadcast_to(np.arange(20) == 19, (40, 20)),
                            round_done=np.ones((40, 20), bool), success=success, rewards=rewards,
                            round_idx=np.broadcast_to(np.arange(20), (40, 20))).items():
                        f[key] = value
                summary = dict(n_episodes=40, n_rounds=800, round_success_rate=float(success.mean()),
                               mean_ep_return=float(rewards.sum(axis=1).mean()),
                               per_round_success_rate=success.mean(axis=0).tolist())
                Path(str(prefix) + '_summary.json').write_text(json.dumps({'test': summary}))
    output = tmp_path / 'plots'
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(action_selection='categorical', evaluation_seeds=[1, 2],
        versions={'v3': {'frozen_source_root': str(source)}}, evaluation_inputs=inputs,
        aggregate_root=str(output))))
    plots.run(manifest, populations=('test',))
    directory = output / 'novel/per_profile'
    report = json.loads((directory / 'profile_comparison.json').read_text())
    assert report['versions'] == ['v1', 'v2', 'v3']
    assert len(report['aggregates']) == 24
    assert len(report['sources']) == 48
    assert all(a['n_seeds'] == 2 for a in report['aggregates'])
    for name in ('profile_success', 'profile_episode_return', 'profile_episode_steps',
                 'profile_round_success_v1', 'profile_round_success_v2', 'profile_round_success_v3'):
        assert (directory / f'{name}.png').is_file()
        assert (directory / f'{name}.pdf').is_file()
