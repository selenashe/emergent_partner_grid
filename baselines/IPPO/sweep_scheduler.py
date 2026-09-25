"""Balanced (z, layout) sweep schedule for the final experiment.

One *sweep* contains:
    * ``n_z`` * ``n_layouts_train`` = ``pairings_per_sweep`` (z, layout) pairs.
    * grouped into partner episodes of ``rounds_per_episode`` rounds each,
      where every episode carries a fixed z and 20 distinct layouts.
    * each z sees every training layout exactly once.

For the default final experiment:
    n_z=5, n_layouts_train=1600, rounds_per_episode=20
        pairings_per_sweep         = 5 * 1600 = 8000
        episodes_per_z_per_sweep   = 1600 / 20 = 80
        episodes_per_sweep         = 80 * 5   = 400

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

    ``schedule_z[i]``       = the z value for the i-th scheduled episode.
    ``schedule_layouts[i]`` = the (rounds_per_episode,) layout indices
                              that episode will visit, in order.
    """
    schedule_z: np.ndarray          # (n_eps_total,) float32
    schedule_layouts: np.ndarray    # (n_eps_total, R) int32
    n_sweeps: int
    n_z: int
    n_layouts: int
    rounds_per_episode: int
    pairings_per_sweep: int
    episodes_per_sweep: int
    episodes_per_z_per_sweep: int
    partner_z_values: np.ndarray     # (n_z,) float32
    seed: int

    @property
    def n_eps_total(self) -> int:
        return int(self.schedule_z.shape[0])


def _one_sweep(
    z_values: np.ndarray,           # (n_z,) float32
    n_layouts: int,
    rounds_per_episode: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build one balanced sweep. Returns
        z_arr:       (episodes_per_sweep,) float32
        layouts_arr: (episodes_per_sweep, R) int32
    Structure: for each z, shuffle the ``n_layouts`` layouts, chunk into
    ``n_layouts / R`` episodes of length R. Aggregate across z, then shuffle
    the *episode order*.
    """
    n_z = z_values.shape[0]
    if n_layouts % rounds_per_episode != 0:
        raise ValueError(
            f"n_layouts ({n_layouts}) must be divisible by rounds_per_episode "
            f"({rounds_per_episode}) so every layout fits exactly once."
        )
    eps_per_z = n_layouts // rounds_per_episode
    eps_per_sweep = n_z * eps_per_z

    z_list = []
    layout_list = []
    for zi, zv in enumerate(z_values):
        layout_perm = rng.permutation(n_layouts).astype(np.int32)
        # Slice into eps_per_z groups of rounds_per_episode layouts each.
        chunks = layout_perm.reshape(eps_per_z, rounds_per_episode)
        z_list.append(np.full(eps_per_z, zv, dtype=np.float32))
        layout_list.append(chunks)

    z_arr = np.concatenate(z_list, axis=0)                    # (eps_per_sweep,)
    layouts_arr = np.concatenate(layout_list, axis=0)          # (eps_per_sweep, R)
    # Shuffle episode order so consecutive episodes don't cluster by z.
    ep_perm = rng.permutation(eps_per_sweep)
    return z_arr[ep_perm], layouts_arr[ep_perm]


def build_schedule(
    partner_z_values: Sequence[float],
    n_layouts_train: int,
    rounds_per_episode: int,
    n_sweeps: int,
    seed: int = 0,
) -> SweepSchedule:
    """Concatenate ``n_sweeps`` independent balanced sweeps into one flat
    schedule.
    """
    z_values = np.asarray(partner_z_values, dtype=np.float32)
    rng = np.random.default_rng(seed)
    z_chunks = []
    layout_chunks = []
    for s in range(n_sweeps):
        z_arr, layouts_arr = _one_sweep(
            z_values, n_layouts_train, rounds_per_episode, rng,
        )
        z_chunks.append(z_arr)
        layout_chunks.append(layouts_arr)
    schedule_z = np.concatenate(z_chunks, axis=0)                 # (S*eps,) float32
    schedule_layouts = np.concatenate(layout_chunks, axis=0)      # (S*eps, R) int32
    eps_per_z = n_layouts_train // rounds_per_episode
    return SweepSchedule(
        schedule_z=schedule_z,
        schedule_layouts=schedule_layouts,
        n_sweeps=n_sweeps,
        n_z=int(z_values.shape[0]),
        n_layouts=int(n_layouts_train),
        rounds_per_episode=int(rounds_per_episode),
        pairings_per_sweep=int(z_values.shape[0]) * int(n_layouts_train),
        episodes_per_sweep=int(z_values.shape[0]) * eps_per_z,
        episodes_per_z_per_sweep=int(eps_per_z),
        partner_z_values=z_values,
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
    Z = schedule.n_z
    K = schedule.n_layouts
    E = schedule.episodes_per_sweep
    P = schedule.pairings_per_sweep
    assert schedule.schedule_z.shape[0] == E * schedule.n_sweeps
    assert schedule.schedule_layouts.shape == (E * schedule.n_sweeps, R)

    z_pool = schedule.partner_z_values

    for s in range(schedule.n_sweeps):
        z_slice = schedule.schedule_z[s * E:(s + 1) * E]
        L_slice = schedule.schedule_layouts[s * E:(s + 1) * E]

        # (a) total pairings in this sweep.
        n_pairings = int(L_slice.size)
        assert n_pairings == P, (
            f"sweep {s}: got {n_pairings} pairings, expected {P}"
        )
        # (b) each z appears exactly episodes_per_z_per_sweep times.
        for zi, zv in enumerate(z_pool):
            n_eps_this_z = int(np.isclose(z_slice, zv, atol=1e-6).sum())
            assert n_eps_this_z == schedule.episodes_per_z_per_sweep, (
                f"sweep {s} z={zv}: got {n_eps_this_z} episodes, "
                f"expected {schedule.episodes_per_z_per_sweep}"
            )
            # (c) within this z, each layout appears exactly once.
            row_mask = np.isclose(z_slice, zv, atol=1e-6)
            layouts_seen = L_slice[row_mask].reshape(-1)
            unique, counts = np.unique(layouts_seen, return_counts=True)
            assert unique.shape[0] == K, (
                f"sweep {s} z={zv}: saw {unique.shape[0]} unique layouts, "
                f"expected {K}"
            )
            assert int(counts.min()) == 1 and int(counts.max()) == 1, (
                f"sweep {s} z={zv}: layout counts min={counts.min()} "
                f"max={counts.max()}, expected all 1"
            )

    summary = {
        "n_sweeps":                  schedule.n_sweeps,
        "n_z":                       Z,
        "n_layouts":                 K,
        "rounds_per_episode":        R,
        "pairings_per_sweep":        P,
        "episodes_per_sweep":        E,
        "episodes_per_z_per_sweep":  schedule.episodes_per_z_per_sweep,
        "n_eps_total":               schedule.n_eps_total,
        "seed":                      schedule.seed,
    }
    if verbose:
        print("[sweep schedule sanity check]")
        for k, v in summary.items():
            print(f"    {k:>26s}: {v}")
        print("    ✓ per sweep: 8000 total pairings, "
              "1600 pairings per z, each layout×z exactly once")
        print(f"    ✓ per sweep: {schedule.episodes_per_z_per_sweep} "
              f"partner episodes per z, {schedule.episodes_per_sweep} "
              f"total partner episodes across all z's")
    return summary


if __name__ == "__main__":
    # Quick CLI: build one sweep with the default final-experiment settings
    # and run the sanity check.
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partner_z_values", type=float, nargs="+",
                   default=[0.1, 0.3, 0.5, 0.7, 0.9])
    p.add_argument("--n_layouts_train", type=int, default=1600)
    p.add_argument("--rounds_per_episode", type=int, default=20)
    p.add_argument("--n_sweeps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sched = build_schedule(
        partner_z_values=args.partner_z_values,
        n_layouts_train=args.n_layouts_train,
        rounds_per_episode=args.rounds_per_episode,
        n_sweeps=args.n_sweeps,
        seed=args.seed,
    )
    sanity_check_schedule(sched)
