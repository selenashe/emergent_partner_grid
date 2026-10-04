"""Preallocated, paired layout packets and a global asynchronous episode queue.

All profiles get the same R-layout packet once before the next packet.
Every full layout pass is without replacement. R/gcd(N,R) passes close a
cycle without padding, shortening episodes, or changing the step budget.
JAX is imported only by the queue/audit helpers, not by plan generation.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Any
import hashlib
import json
import math

import numpy as np

SAMPLING_PROTOCOL = "paired_counterbalanced_v1"


@dataclass
class BalancedSchedule:
    layouts: np.ndarray             # (packets, R), shared across profiles
    profile_order: np.ndarray       # (packets, C), permutation of 0..C-1
    capability_pairs: np.ndarray    # (C, 2)
    seed: int
    n_layouts: int
    passes_per_cycle: int

    @property
    def n_episodes(self):
        return int(self.layouts.shape[0] * len(self.capability_pairs))

    @property
    def rounds(self):
        return int(self.layouts.shape[1])

    def summary(self):
        c = len(self.capability_pairs)
        return {"sampling_protocol": SAMPLING_PROTOCOL, "seed": self.seed,
                "n_layouts": self.n_layouts, "n_profiles": c, "rounds_per_episode": self.rounds,
                "passes_per_cycle": self.passes_per_cycle,
                "episodes_per_profile_per_cycle": self.n_layouts * self.passes_per_cycle // self.rounds,
                "rounds_per_cycle": self.n_layouts * self.passes_per_cycle * c,
                "visits_per_profile_layout_per_cycle": self.passes_per_cycle,
                "packets": len(self.layouts), "preallocated_episodes": self.n_episodes,
                "preallocated_rounds": self.n_episodes * self.rounds,
                "maximum_allocated_episode_imbalance_at_any_prefix": 1}


def build_schedule(capability_pairs, n_layouts, rounds, minimum_episodes, seed=2026):
    pairs = np.asarray(capability_pairs, dtype=np.int32)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
        raise ValueError("Expected nonempty (C,2) capability pairs")
    if len(np.unique(pairs, axis=0)) != len(pairs) or np.any(pairs < 0):
        raise ValueError("Profiles must be unique, nonnegative pairs")
    if min(n_layouts, rounds, minimum_episodes) < 1:
        raise ValueError("Layout/round/episode counts must be positive")
    passes = rounds // math.gcd(n_layouts, rounds)
    packets_per_cycle = n_layouts * passes // rounds
    cycles = math.ceil(minimum_episodes / (packets_per_cycle * len(pairs)))
    # Independent streams preserve identical layout prefixes for diverse/single controls.
    layout_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    order_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    layouts = np.empty((cycles * packets_per_cycle, rounds), dtype=np.int32)
    for cycle in range(cycles):
        bag = np.concatenate([layout_rng.permutation(n_layouts) for _ in range(passes)])
        layouts[cycle * packets_per_cycle:(cycle + 1) * packets_per_cycle] = bag.reshape(-1, rounds)
    order = np.empty((len(layouts), len(pairs)), dtype=np.int32)
    for packet in range(len(layouts)):
        order[packet] = order_rng.permutation(len(pairs))
    return BalancedSchedule(layouts, order, pairs, int(seed), int(n_layouts), passes)


def save_schedule(schedule, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, layouts=schedule.layouts, profile_order=schedule.profile_order,
             capability_pairs=schedule.capability_pairs, seed=schedule.seed,
             n_layouts=schedule.n_layouts, passes_per_cycle=schedule.passes_per_cycle)
    summary = schedule.summary()
    summary["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def load_schedule(path):
    with np.load(path, allow_pickle=False) as data:
        return BalancedSchedule(data['layouts'], data['profile_order'], data['capability_pairs'],
                                int(data['seed']), int(data['n_layouts']), int(data['passes_per_cycle']))


class QueueState(NamedTuple):
    next_episode: Any
    slot_episode: Any
    layout_counts: Any              # (3,C,N): rounds started/completed, steps
    episode_counts: Any             # (2,C): episodes started/completed


def initial_queue(num_envs, n_profiles, n_layouts):
    import jax.numpy as jnp
    return QueueState(jnp.int32(num_envs), jnp.arange(num_envs, dtype=jnp.int32),
                      jnp.zeros((3, n_profiles, n_layouts), dtype=jnp.int32),
                      jnp.zeros((2, n_profiles), dtype=jnp.int32))


def dispatch(queue, done):
    """One fresh, unique episode per terminal worker, in stable slot order."""
    import jax.numpy as jnp
    ranks = jnp.cumsum(done.astype(jnp.int32)) - 1
    slots = jnp.where(done, queue.next_episode + ranks, queue.slot_episode)
    return queue._replace(next_episode=queue.next_episode + done.sum(dtype=jnp.int32),
                          slot_episode=slots)


def lookup(episode_ids, shared_layouts, profile_order, capability_pairs):
    c = capability_pairs.shape[0]
    packets = episode_ids // c
    profiles = profile_order[packets, episode_ids % c]
    return capability_pairs[profiles], shared_layouts[packets]


def accumulate_audit(queue, info, times, episode_done, capability_pairs):
    import jax.numpy as jnp
    caps = info['capability'].reshape(-1, 2)
    profile = jnp.argmax(jnp.all(caps[:, None, :] == capability_pairs[None, :, :], axis=-1), axis=1)
    layout = info['layout_idx'].reshape(-1).astype(jnp.int32)
    starts = (times.reshape(-1) == 0).astype(jnp.int32)
    finishes = info['round_done'].reshape(-1).astype(jnp.int32)
    counts = queue.layout_counts
    counts = counts.at[0, profile, layout].add(starts)
    counts = counts.at[1, profile, layout].add(finishes)
    counts = counts.at[2, profile, layout].add(jnp.ones_like(starts))
    episode_starts = starts * (info['round_idx'].reshape(-1) == 0).astype(jnp.int32)
    eps = queue.episode_counts.at[0, profile].add(episode_starts)
    eps = eps.at[1, profile].add(episode_done.reshape(-1).astype(jnp.int32))
    return queue._replace(layout_counts=counts, episode_counts=eps)


def write_audit(queue, env_state, schedule_path, config, checkpoint_path):
    """Persist empirical exposure, including the unfinished tail at 60M steps."""
    schedule = load_schedule(schedule_path)
    base = Path(checkpoint_path).with_suffix('')
    for seed_index in range(int(config['NUM_SEEDS'])):
        counts = np.asarray(queue.layout_counts)[seed_index]
        eps = np.asarray(queue.episode_counts)[seed_index]
        allocated = int(np.asarray(queue.next_episode)[seed_index])
        c = len(schedule.capability_pairs)
        full, tail = divmod(allocated, c)
        allocation_counts = np.full(c, full, dtype=np.int64)
        if tail:
            allocation_counts[schedule.profile_order[full, :tail]] += 1
        assert int(counts[2].sum()) == config['NUM_UPDATES'] * config['NUM_ENVS'] * config['NUM_STEPS']
        assert int(eps[1].sum()) * schedule.rounds <= int(counts[1].sum())
        assert allocation_counts.max() - allocation_counts.min() <= 1
        ids = np.asarray(queue.slot_episode)[seed_index]
        assert len(np.unique(ids)) == config['NUM_ENVS'] and ids.max() < allocated
        suffix = '' if int(config['NUM_SEEDS']) == 1 else f'_seed{seed_index}'
        prefix = str(base) + suffix + '_sampling_audit'
        layout_ids = [p.stem for p in sorted(Path(config['ENV_KWARGS']['layouts_dir']).glob('*.json'))]
        np.savez(prefix + '.npz', capability_pairs=schedule.capability_pairs,
                 layout_ids=np.asarray(layout_ids), rounds_started=counts[0],
                 rounds_completed=counts[1], environment_steps=counts[2],
                 episodes_started=eps[0], episodes_completed=eps[1],
                 episodes_allocated=allocation_counts, active_episode_ids=ids,
                 active_round_idx=np.asarray(env_state.round_idx)[seed_index],
                 active_round_time=np.asarray(env_state.time)[seed_index],
                 active_capability=np.asarray(env_state.capability)[seed_index])
        rows = [{"capability": pair.tolist(), "episodes_allocated": int(allocation_counts[i]),
                 "episodes_started": int(eps[0, i]), "episodes_completed": int(eps[1, i]),
                 "rounds_started": int(counts[0, i].sum()), "rounds_completed": int(counts[1, i].sum()),
                 "environment_steps": int(counts[2, i].sum()),
                 "unique_layouts_started": int(np.count_nonzero(counts[0, i]))}
                for i, pair in enumerate(schedule.capability_pairs)]
        report = {"sampling_protocol": SAMPLING_PROTOCOL, "experiment_version": config['EXPERIMENT_VERSION'],
                  "allocation_protocol": config.get('ALLOCATION_PROTOCOL', 'fixed_v1'),
                  "schedule_path": str(schedule_path),
                  "schedule_sha256": hashlib.sha256(Path(schedule_path).read_bytes()).hexdigest(),
                  "allocated_episode_prefix": allocated, "unused_preallocated_episodes": schedule.n_episodes - allocated,
                  "allocated_episode_count_difference": int(allocation_counts.max() - allocation_counts.min()),
                  "collected_environment_steps": int(counts[2].sum()),
                  "rounds_started": int(counts[0].sum()), "rounds_completed": int(counts[1].sum()),
                  "profiles": rows, "arrays": prefix + '.npz',
                  "note": "Allocation balance is exact up to one episode. Started/completed exposure can differ at the fixed-step cutoff; full matrices and active episodes are retained."}
        Path(prefix + '.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'[sampling audit] {prefix}.json', flush=True)
