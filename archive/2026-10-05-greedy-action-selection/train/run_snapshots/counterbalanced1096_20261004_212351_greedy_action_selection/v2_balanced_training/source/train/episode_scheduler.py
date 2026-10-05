"""Original v1/v2 random capability/layout sampler for CoordinationGrid.

This is not a generator or hyperparameter sweep. Counterbalanced training
uses counterbalanced_scheduler.py instead.

The scientific unit that must remain stable is:

    one capability profile per 20-round partner episode

Layouts within a partner episode are sampled uniformly with replacement
from the supplied training corpus. Capability pairs are sampled
uniformly with replacement from the training pool (one per partner
episode). Basic balancing across the schedule falls out of large-N
uniform sampling; we do NOT enforce an exhaustive capability × layout
cross-product any more.

This module still returns a materialized ``(capability, layout_seq)``
schedule so the trainer can advance a per-slot cursor inside
``jax.lax.scan`` (deterministic + JIT-friendly). The schedule is
long enough that a training run never wraps in practice; it wraps
harmlessly if it does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class EpisodeSchedule:
    """A flat schedule of scheduled partner episodes.

    ``schedule_capability[i]``  = the (d_R, d_B) pair for the i-th ep.
    ``schedule_layouts[i]``     = the (rounds_per_episode,) layout ids.
    """
    schedule_capability: np.ndarray   # (n_eps_total, 2) int32
    schedule_layouts: np.ndarray      # (n_eps_total, R) int32
    n_cap: int
    n_layouts: int
    rounds_per_episode: int
    n_eps_total: int
    partner_capability_pairs: np.ndarray  # (n_cap, 2) int32
    seed: int


def build_schedule(
    partner_capability_pairs: Sequence[Sequence[int]],
    n_layouts_train: int,
    rounds_per_episode: int,
    n_eps_total: int,
    seed: int = 0,
) -> EpisodeSchedule:
    """Materialize a schedule of ``n_eps_total`` partner episodes.

    Each episode gets one capability pair sampled uniformly from the
    training pool and ``rounds_per_episode`` layouts sampled uniformly
    with replacement from ``[0, n_layouts_train)``.
    """
    cap_pairs = np.asarray(
        [[int(a), int(b)] for a, b in partner_capability_pairs],
        dtype=np.int32,
    )
    if cap_pairs.ndim != 2 or cap_pairs.shape[1] != 2:
        raise ValueError(
            f"partner_capability_pairs must be shape (K, 2); got {cap_pairs.shape}"
        )
    if n_layouts_train < 1:
        raise ValueError("n_layouts_train must be >= 1")
    if rounds_per_episode < 1:
        raise ValueError("rounds_per_episode must be >= 1")
    if n_eps_total < 1:
        raise ValueError("n_eps_total must be >= 1")

    rng = np.random.default_rng(seed)
    cap_idx = rng.integers(0, cap_pairs.shape[0], size=n_eps_total)
    schedule_capability = cap_pairs[cap_idx]                              # (E, 2)
    schedule_layouts = rng.integers(
        0, n_layouts_train, size=(n_eps_total, rounds_per_episode), dtype=np.int32,
    )
    return EpisodeSchedule(
        schedule_capability=schedule_capability.astype(np.int32),
        schedule_layouts=schedule_layouts,
        n_cap=int(cap_pairs.shape[0]),
        n_layouts=int(n_layouts_train),
        rounds_per_episode=int(rounds_per_episode),
        n_eps_total=int(n_eps_total),
        partner_capability_pairs=cap_pairs,
        seed=int(seed),
    )


def initial_episode_cursor(num_envs: int, schedule: EpisodeSchedule) -> np.ndarray:
    """Give each vmap slot its own starting cursor, evenly spaced along
    the schedule so slots don't co-visit the same (capability, layout).
    """
    starts = (np.arange(num_envs) * (schedule.n_eps_total // max(num_envs, 1)))
    return (starts % schedule.n_eps_total).astype(np.int32)


def summarize_schedule(schedule: EpisodeSchedule, *, verbose: bool = True) -> dict:
    """Report coverage of the original random sampler, without balance quotas."""
    cap_pool = schedule.partner_capability_pairs
    counts = np.zeros(cap_pool.shape[0], dtype=np.int64)
    for ci, cap in enumerate(cap_pool):
        counts[ci] = int(np.all(schedule.schedule_capability == cap[None, :], axis=1).sum())
    summary = {
        "n_cap":              int(cap_pool.shape[0]),
        "n_layouts":          int(schedule.n_layouts),
        "rounds_per_episode": int(schedule.rounds_per_episode),
        "n_eps_total":        int(schedule.n_eps_total),
        "eps_per_cap_min":    int(counts.min()),
        "eps_per_cap_mean":   float(counts.mean()),
        "eps_per_cap_max":    int(counts.max()),
        "seed":               int(schedule.seed),
    }
    if verbose:
        print("[capability/layout schedule summary]")
        for k, v in summary.items():
            print(f"    {k:>22s}: {v}")
    return summary


if __name__ == "__main__":
    import argparse
    from jaxmarl.environments.coordination_grid import TRAIN_CAPABILITY_PAIRS

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_layouts_train", type=int, default=1000)
    p.add_argument("--rounds_per_episode", type=int, default=20)
    p.add_argument("--n_eps_total", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sched = build_schedule(
        partner_capability_pairs=TRAIN_CAPABILITY_PAIRS,
        n_layouts_train=args.n_layouts_train,
        rounds_per_episode=args.rounds_per_episode,
        n_eps_total=args.n_eps_total,
        seed=args.seed,
    )
    summarize_schedule(sched)
