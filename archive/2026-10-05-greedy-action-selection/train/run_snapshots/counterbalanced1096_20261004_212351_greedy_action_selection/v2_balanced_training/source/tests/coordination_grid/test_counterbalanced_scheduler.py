"""Exposure and queue invariants, including asynchronous workers and cutoff tails."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'train'))
from counterbalanced_scheduler import (build_schedule, save_schedule, load_schedule,
    initial_queue, dispatch, lookup, accumulate_audit, write_audit)


def test_complete_cycles_have_exact_pair_counts_and_full_without_replacement_passes():
    pairs = [(1, 4), (4, 1), (3, 9)]
    schedule = build_schedule(pairs, 1096, 20, 2000)
    assert schedule.passes_per_cycle == 5
    assert schedule.n_episodes % (274 * 3) == 0
    for bag in schedule.layouts.reshape(-1, 1096):
        np.testing.assert_array_equal(np.sort(bag), np.arange(1096))
    np.testing.assert_array_equal(np.sort(schedule.profile_order, axis=1),
                                  np.broadcast_to(np.arange(3), schedule.profile_order.shape))
    ids = np.arange(schedule.n_episodes)
    caps, layouts = lookup(ids, schedule.layouts, schedule.profile_order, schedule.capability_pairs)
    counts = []
    for pair in pairs:
        visits = np.bincount(layouts[np.all(caps == pair, axis=1)].ravel(), minlength=1096)
        assert visits.min() == visits.max()
        counts.append(visits)
    assert np.all(np.asarray(counts) == counts[0])


def test_every_episode_queue_prefix_is_profile_balanced_and_packets_are_paired():
    s = build_schedule([(1, 4), (4, 1), (3, 9)], 17, 20, 100)
    counts = np.zeros(3, int)
    for cap in s.profile_order.ravel():
        counts[cap] += 1
        assert counts.max() - counts.min() <= 1
    caps, layouts = lookup(np.arange(s.n_episodes), s.layouts, s.profile_order, s.capability_pairs)
    for i in range(0, s.n_episodes, 3):
        assert len(np.unique(caps[i:i + 3], axis=0)) == 3
        assert np.all(layouts[i:i + 3] == layouts[i])


def test_seed_reproducibility_and_layout_prefix_shared_with_single_partner_control():
    a = build_schedule([(1, 4), (4, 1)], 1096, 20, 100)
    b = build_schedule([(1, 4)], 1096, 20, 1000)
    c = build_schedule([(1, 4), (4, 1)], 1096, 20, 100)
    np.testing.assert_array_equal(a.layouts, b.layouts[:len(a.layouts)])
    np.testing.assert_array_equal(a.layouts, c.layouts)
    np.testing.assert_array_equal(a.profile_order, c.profile_order)
    other = build_schedule([(1, 4), (4, 1)], 1096, 20, 100, seed=2027)
    assert not np.array_equal(a.layouts, other.layouts)


def test_jitted_asynchronous_dispatch_never_reuses_or_skips_an_episode():
    q = initial_queue(17, 3, 1096)
    allocated = list(range(17))
    rng = np.random.default_rng(3)
    step = jax.jit(dispatch)
    for _ in range(100):
        done = rng.random(17) < 0.25
        before = np.asarray(q.slot_episode)
        base = int(q.next_episode)
        q = step(q, jnp.asarray(done))
        after = np.asarray(q.slot_episode)
        np.testing.assert_array_equal(after[~done], before[~done])
        np.testing.assert_array_equal(after[done], np.arange(base, base + done.sum()))
        allocated.extend(after[done].tolist())
        assert len(np.unique(after)) == 17
    assert allocated == list(range(int(q.next_episode)))


def test_audit_conserves_steps_and_distinguishes_started_from_completed(tmp_path):
    pairs = np.array([(1, 4), (4, 1)])
    info = {'capability': jnp.asarray(np.broadcast_to(pairs, (3, 2, 2))),
            'layout_idx': jnp.array([[0, 1], [0, 1], [2, 3]]),
            'round_done': jnp.array([[0, 0], [1, 1], [0, 0]]),
            'round_idx': jnp.array([[0, 0], [0, 0], [1, 1]])}
    q = initial_queue(2, 2, 4)
    q = jax.jit(accumulate_audit)(q, info, jnp.array([[0, 0], [1, 1], [0, 0]]),
                                 jnp.zeros((3, 2), bool), jnp.asarray(pairs))
    assert np.asarray(q.layout_counts).sum(axis=(1, 2)).tolist() == [4, 2, 6]
    assert np.asarray(q.episode_counts).sum(axis=1).tolist() == [2, 0]
    schedule = build_schedule(pairs, 4, 20, 10)
    path = tmp_path / 'plan.npz'
    save_schedule(schedule, path)
    for i in range(4):
        (tmp_path / f'layout_{i}.json').write_text('{}')
    vmapped = jax.tree_util.tree_map(lambda x: x[None], q)
    env = SimpleNamespace(round_idx=jnp.array([[1, 1]]), time=jnp.array([[1, 1]]),
                          capability=jnp.array([pairs]))
    config = {'NUM_SEEDS': 1, 'NUM_UPDATES': 1, 'NUM_ENVS': 2, 'NUM_STEPS': 3,
              'EXPERIMENT_VERSION': 'test', 'ENV_KWARGS': {'layouts_dir': str(tmp_path)}}
    write_audit(vmapped, env, path, config, tmp_path / 'policy.safetensors')
    report = json.loads((tmp_path / 'policy_sampling_audit.json').read_text())
    assert report['collected_environment_steps'] == 6
    assert report['rounds_started'] == 4 and report['rounds_completed'] == 2
    assert report['allocated_episode_count_difference'] == 0


def test_serialization_and_capacity(tmp_path):
    s = build_schedule([(1, 4), (4, 1)], 1096, 20, 3000257)
    assert s.n_episodes >= 3000257
    p = tmp_path / 'schedule.npz'
    summary = save_schedule(s, p)
    restored = load_schedule(p)
    np.testing.assert_array_equal(restored.layouts, s.layouts)
    assert restored.summary() == s.summary()
    assert len(summary['sha256']) == 64


@pytest.mark.parametrize('pairs,n,r,eps', [([], 4, 20, 4), ([(1, 4), (1, 4)], 4, 20, 4),
                                         ([(1, -1)], 4, 20, 4), ([(1, 4)], 0, 20, 4)])
def test_invalid_plans_fail_before_training(pairs, n, r, eps):
    with pytest.raises(ValueError):
        build_schedule(pairs, n, r, eps)
