"""Balanced (capability_pair, layout) sweep schedule for the coordination-
grid final experiment.

One *sweep* contains:
    * ``n_cap * n_layouts_train = pairings_per_sweep`` (capability, layout)
      pairs.
    * grouped into partner episodes of ``rounds_per_episode`` rounds each,
      where every episode carries a fixed capability profile (c_R, c_B)
      and 20 distinct layouts.
    * each capability profile sees every training layout exactly once.

For the default final experiment:
    n_cap=30 (training capability pairs), n_layouts_train=1600, R=20
        pairings_per_sweep         = 30 * 1600 = 48000
        episodes_per_cap_per_sweep = 1600 / 20 = 80
        episodes_per_sweep         = 80 * 30 = 2400

The trainer stitches ``n_sweeps`` such sweeps back-to-back to build the
full training schedule. Each vmap slot starts on its own sweep boundary
(via ``initial_episode_cursor``) and advances one episode at a time.

This module is standalone — it is imported by both the trainer and the
sanity-check script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass
class SweepSchedule:
    """A flat schedule of scheduled partner episodes.

    ``schedule_capability[i]`` = the (c_R, c_B) pair for the i-th episode.
    ``schedule_layouts[i]``    = the (rounds_per_episode,) layout indices
                                 that episode will visit, in order.
    """
    schedule_capability: np.ndarray  # (n_eps_total, 2) int32
    schedule_layouts: np.ndarray     # (n_eps_total, R) int32
    n_sweeps: int
    n_cap: int
    n_layouts: int
    rounds_per_episode: int
    pairings_per_sweep: int
    episodes_per_sweep: int
    episodes_per_cap_per_sweep: int
    partner_capability_pairs: np.ndarray  # (n_cap, 2) int32
    seed: int

    @property
    def n_eps_total(self) -> int:
        return int(self.schedule_capability.shape[0])


def _one_sweep(
    capability_pairs: np.ndarray,     # (n_cap, 2) int32
    n_layouts: int,
    rounds_per_episode: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build one balanced sweep. Returns
        cap_arr:     (episodes_per_sweep, 2) int32
        layouts_arr: (episodes_per_sweep, R) int32
    Structure: for each capability pair, shuffle the ``n_layouts`` layouts,
    chunk into ``n_layouts / R`` episodes of length R. Aggregate across
    capabilities, then shuffle the *episode order*.
    """
    n_cap = capability_pairs.shape[0]
    if n_layouts % rounds_per_episode != 0:
        raise ValueError(
            f"n_layouts ({n_layouts}) must be divisible by rounds_per_episode "
            f"({rounds_per_episode}) so every layout fits exactly once."
        )
    eps_per_cap = n_layouts // rounds_per_episode
    eps_per_sweep = n_cap * eps_per_cap

    cap_list = []
    layout_list = []
    for ci in range(n_cap):
        layout_perm = rng.permutation(n_layouts).astype(np.int32)
        chunks = layout_perm.reshape(eps_per_cap, rounds_per_episode)
        cap_list.append(
            np.broadcast_to(capability_pairs[ci], (eps_per_cap, 2)).astype(np.int32).copy()
        )
        layout_list.append(chunks)

    cap_arr = np.concatenate(cap_list, axis=0)          # (eps_per_sweep, 2)
    layouts_arr = np.concatenate(layout_list, axis=0)   # (eps_per_sweep, R)
    ep_perm = rng.permutation(eps_per_sweep)
    return cap_arr[ep_perm], layouts_arr[ep_perm]


def build_schedule(
    partner_capability_pairs: Sequence[Sequence[int]],
    n_layouts_train: int,
    rounds_per_episode: int,
    n_sweeps: int,
    seed: int = 0,
) -> SweepSchedule:
    """Concatenate ``n_sweeps`` independent balanced sweeps into one flat
    schedule.
    """
    cap_pairs = np.asarray(
        [[int(a), int(b)] for a, b in partner_capability_pairs],
        dtype=np.int32,
    )
    if cap_pairs.ndim != 2 or cap_pairs.shape[1] != 2:
        raise ValueError(
            f"partner_capability_pairs must be shape (K, 2); got {cap_pairs.shape}"
        )
    rng = np.random.default_rng(seed)
    cap_chunks = []
    layout_chunks = []
    for s in range(n_sweeps):
        cap_arr, layouts_arr = _one_sweep(
            cap_pairs, n_layouts_train, rounds_per_episode, rng,
        )
        cap_chunks.append(cap_arr)
        layout_chunks.append(layouts_arr)
    schedule_capability = np.concatenate(cap_chunks, axis=0)      # (S*eps, 2)
    schedule_layouts = np.concatenate(layout_chunks, axis=0)      # (S*eps, R)
    eps_per_cap = n_layouts_train // rounds_per_episode
    return SweepSchedule(
        schedule_capability=schedule_capability,
        schedule_layouts=schedule_layouts,
        n_sweeps=n_sweeps,
        n_cap=int(cap_pairs.shape[0]),
        n_layouts=int(n_layouts_train),
        rounds_per_episode=int(rounds_per_episode),
        pairings_per_sweep=int(cap_pairs.shape[0]) * int(n_layouts_train),
        episodes_per_sweep=int(cap_pairs.shape[0]) * eps_per_cap,
        episodes_per_cap_per_sweep=int(eps_per_cap),
        partner_capability_pairs=cap_pairs,
        seed=int(seed),
    )


def initial_episode_cursor(
    num_envs: int, schedule: SweepSchedule,
) -> np.ndarray:
    """Sweep-aligned starting cursor for each vmap slot. Worker w starts
    at sweep ``w % n_sweeps``, within-sweep episode index 0. This preserves
    the per-worker sweep-boundary alignment (so a worker never crosses a
    sweep boundary mid-episode by construction).
    """
    starts = (np.arange(num_envs) % schedule.n_sweeps) * schedule.episodes_per_sweep
    return starts.astype(np.int32)


# --------------------------------------------------------------------------- #
# Sanity check                                                                #
# --------------------------------------------------------------------------- #

def sanity_check_schedule(schedule: SweepSchedule, *, verbose: bool = True) -> dict:
    """Verify each sweep's structural properties. Returns a summary dict.
    Raises AssertionError on violations.
    """
    R = schedule.rounds_per_episode
    C = schedule.n_cap
    K = schedule.n_layouts
    E = schedule.episodes_per_sweep
    P = schedule.pairings_per_sweep
    assert schedule.schedule_capability.shape == (E * schedule.n_sweeps, 2)
    assert schedule.schedule_layouts.shape == (E * schedule.n_sweeps, R)

    cap_pool = schedule.partner_capability_pairs  # (n_cap, 2)

    for s in range(schedule.n_sweeps):
        cap_slice = schedule.schedule_capability[s * E:(s + 1) * E]
        L_slice = schedule.schedule_layouts[s * E:(s + 1) * E]

        # (a) total pairings in this sweep.
        n_pairings = int(L_slice.size)
        assert n_pairings == P, (
            f"sweep {s}: got {n_pairings} pairings, expected {P}"
        )
        # (b) each capability pair appears exactly episodes_per_cap_per_sweep times.
        for ci, cap in enumerate(cap_pool):
            row_mask = np.all(cap_slice == cap[None, :], axis=1)
            n_eps_this_cap = int(row_mask.sum())
            assert n_eps_this_cap == schedule.episodes_per_cap_per_sweep, (
                f"sweep {s} cap={tuple(cap.tolist())}: got {n_eps_this_cap} "
                f"episodes, expected {schedule.episodes_per_cap_per_sweep}"
            )
            # (c) within this cap, each layout appears exactly once.
            layouts_seen = L_slice[row_mask].reshape(-1)
            unique, counts = np.unique(layouts_seen, return_counts=True)
            assert unique.shape[0] == K, (
                f"sweep {s} cap={tuple(cap.tolist())}: saw {unique.shape[0]} "
                f"unique layouts, expected {K}"
            )
            assert int(counts.min()) == 1 and int(counts.max()) == 1, (
                f"sweep {s} cap={tuple(cap.tolist())}: layout counts "
                f"min={counts.min()} max={counts.max()}, expected all 1"
            )

    summary = {
        "n_sweeps":                       schedule.n_sweeps,
        "n_cap":                          C,
        "n_layouts":                      K,
        "rounds_per_episode":             R,
        "pairings_per_sweep":             P,
        "episodes_per_sweep":             E,
        "episodes_per_cap_per_sweep":     schedule.episodes_per_cap_per_sweep,
        "n_eps_total":                    schedule.n_eps_total,
        "seed":                           schedule.seed,
    }
    if verbose:
        print("[sweep schedule sanity check]")
        for k, v in summary.items():
            print(f"    {k:>28s}: {v}")
        print(f"    ✓ per sweep: {P} total pairings ({C} cap × {K} layouts), "
              "each cap×layout exactly once")
        print(f"    ✓ per sweep: {schedule.episodes_per_cap_per_sweep} partner "
              f"episodes per cap, {schedule.episodes_per_sweep} total")
    return summary


if __name__ == "__main__":
    import argparse
    from jaxmarl.environments.coordination_grid import TRAINING_CAPABILITY_PAIRS

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_layouts_train", type=int, default=1600)
    p.add_argument("--rounds_per_episode", type=int, default=20)
    p.add_argument("--n_sweeps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sched = build_schedule(
        partner_capability_pairs=TRAINING_CAPABILITY_PAIRS,
        n_layouts_train=args.n_layouts_train,
        rounds_per_episode=args.rounds_per_episode,
        n_sweeps=args.n_sweeps,
        seed=args.seed,
    )
    sanity_check_schedule(sched)
