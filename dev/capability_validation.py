"""Pre-PPO validation for the capability-vector CoordinationGrid design.

For each (layout, capability_pair) we analytically compute how many env
steps it would take for the ego + partner to complete each of the two
possible role allocations:

    * Allocation A: ego → RED,  partner → BLUE  (cooldown = c_B).
    * Allocation B: ego → BLUE, partner → RED   (cooldown = c_R).

Assuming both agents follow shortest-path navigation and there are no
collisions (a reasonable approximation on 7×7 with well-separated goals),
completion time is:

    call_success(alloc) = max(
        1 + BFS(ego_start,     ego_goal),
        2 + (BFS(partner_start, partner_goal) - 1) * partner_cooldown
    )

    success(alloc) = (call_success <= max_steps)

We then report, over the layout corpus and the training / held-out
capability pools separately:

    * fraction of layouts where the optimal allocation flips across
      capability profiles;
    * mean / median regret in "steps to complete" for the wrong allocation
      (only over layouts where at least one allocation is feasible);
    * ORACLE success rate: always pick the allocation that is optimal for
      *this* (layout, cap);
    * PARTNER-BLIND success rate: pick a fixed allocation per layout
      (the one that maximises mean success across the capability pool),
      regardless of cap.

We only proceed to PPO if oracle ≫ partner-blind on the held-out
capability pool: that is, if the partner's capability is actually a
useful thing to infer.

Usage:
    python dev/capability_validation.py \
        --layouts_dir dev/grids_final/layouts/val \
        --max_steps 15 \
        --out dev/train_logs/capability_validation_val.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capability_selection import (  # noqa: E402
    INFEASIBLE,
    completion_time,
)

from jaxmarl.environments.coordination_grid import (  # noqa: E402
    CAPABILITY_VALUES,
    TRAINING_CAPABILITY_PAIRS,
    HELDOUT_CAPABILITY_PAIRS,
    bfs_distance_map,
)


def load_layouts(layouts_dir: str) -> List[dict]:
    """Load a directory of layout JSONs and precompute BFS distances from
    the ego_start / partner_start to each of the two goals.
    """
    paths = sorted(glob.glob(os.path.join(layouts_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"No *.json under {layouts_dir}")
    out: List[dict] = []
    for p in paths:
        with open(p, "r") as f:
            raw = json.load(f)
        grid = np.array(raw["grid"], dtype=np.int32)
        h, w = grid.shape
        wall = (grid == 1)
        # JSON uses [row, col]; env converts to (x, y).
        def rc2xy(rc):
            return np.array([rc[1], rc[0]], dtype=np.int32)
        ego = rc2xy(raw["ego_start"])
        partner = rc2xy(raw["partner_start"])
        red = rc2xy(raw["red_goal"])
        blue = rc2xy(raw["blue_goal"])
        # BFS distance from each goal, then index by start cell.
        d_red = bfs_distance_map(wall, red)
        d_blue = bfs_distance_map(wall, blue)
        ex, ey = int(ego[0]), int(ego[1])
        px, py = int(partner[0]), int(partner[1])
        out.append({
            "path": p,
            "ego_to_red":     int(d_red[ey, ex]),
            "ego_to_blue":    int(d_blue[ey, ex]),
            "partner_to_red": int(d_red[py, px]),
            "partner_to_blue": int(d_blue[py, px]),
        })
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _cap_pool_analysis(layouts: List[dict],
                       cap_pairs: Sequence[Tuple[int, int]],
                       max_steps: int,
                       step_penalty: float = 0.01,
                       success_reward: float = 1.0) -> Dict:
    """For a fixed pool of capability pairs, compute completion times, the
    optimal-allocation flip fraction across the pool, oracle success, and
    partner-blind success.

    Returns:
        {
          "n_layouts": int,
          "n_cap_pairs": int,
          "cap_pairs": [(cR, cB), ...],
          "flip_fraction": float,           # frac of layouts where optimal
                                            # allocation is not constant across
                                            # the cap pool.
          "regret_when_wrong_mean": float,  # mean regret (steps) when the
                                            # non-optimal allocation is chosen,
                                            # counted only over
                                            # (layout, cap) where at least one
                                            # allocation succeeds.
          "regret_when_wrong_median": float,
          "regret_fail_frac": float,        # frac of (layout, cap) where BOTH
                                            # allocations infeasible (regret undefined).
          "oracle_success_rate": float,
          "partner_blind_success_rate": float,
          "oracle_minus_blind": float,
          "oracle_avg_time": float,         # avg completion time (successful only)
          "blind_avg_time": float,
        }

    Two allocations per (layout, cap):
        A (ego→RED, partner→BLUE, cool=c_B)
        B (ego→BLUE, partner→RED, cool=c_R)
    """
    n_L = len(layouts)
    n_C = len(cap_pairs)
    # Precompute completion times for all (layout, cap, alloc).
    # times[i, j, 0] = alloc A completion for layout i, cap j
    # times[i, j, 1] = alloc B completion for layout i, cap j
    times = np.full((n_L, n_C, 2), INFEASIBLE, dtype=np.int64)
    for i, ld in enumerate(layouts):
        for j, (cr, cb) in enumerate(cap_pairs):
            ta = completion_time(
                ego_path=ld["ego_to_red"],
                partner_path=ld["partner_to_blue"],
                partner_cool=cb,
                max_steps=max_steps,
            )
            tb = completion_time(
                ego_path=ld["ego_to_blue"],
                partner_path=ld["partner_to_red"],
                partner_cool=cr,
                max_steps=max_steps,
            )
            times[i, j, 0] = ta
            times[i, j, 1] = tb

    # Success: any (layout, cap, alloc) with time < INFEASIBLE.
    success = times < INFEASIBLE

    # Optimal allocation per (layout, cap):
    #   0 = A, 1 = B, -1 = both infeasible.
    best_time = np.min(times, axis=2)                  # (L, C)
    optimal = np.argmin(times, axis=2)                 # (L, C)
    both_bad = ~success.any(axis=2)                    # (L, C)
    optimal_masked = np.where(both_bad, -1, optimal)   # (L, C)

    # ---- flip fraction: layouts where NOT all cap profiles share the same
    # optimal allocation. Layouts where at least one cap has both infeasible
    # are counted as ambiguous and excluded from the denominator only if EVERY
    # cap has both infeasible (which shouldn't happen); we count them as
    # non-flipping conservatively.
    flip_flags = np.zeros(n_L, dtype=bool)
    for i in range(n_L):
        row = optimal_masked[i]
        # Only consider caps where an optimum is defined.
        row_defined = row[row >= 0]
        if len(row_defined) < 2:
            flip_flags[i] = False
        else:
            flip_flags[i] = (len(set(row_defined.tolist())) > 1)
    flip_fraction = float(flip_flags.mean())

    # ---- regret when wrong ----
    # For each (layout, cap) where an optimum exists (at least one alloc
    # succeeds), regret := time(wrong_alloc) - time(best_alloc). If the
    # wrong alloc is infeasible we assign INFEASIBLE - best_time (large).
    regrets = []
    fail_both = 0
    for i in range(n_L):
        for j in range(n_C):
            ta, tb = int(times[i, j, 0]), int(times[i, j, 1])
            if ta >= INFEASIBLE and tb >= INFEASIBLE:
                fail_both += 1
                continue
            best = min(ta, tb)
            worst = max(ta, tb)
            if worst >= INFEASIBLE:
                # Wrong alloc infeasible. Use `max_steps + 1 - best` as a
                # bounded regret proxy so the median doesn't blow up.
                regrets.append(max_steps + 1 - best)
            else:
                regrets.append(worst - best)
    regrets_arr = np.asarray(regrets, dtype=np.int64) if regrets else np.array([0])
    regret_mean = float(regrets_arr.mean())
    regret_median = float(np.median(regrets_arr))
    regret_fail_frac = fail_both / (n_L * n_C)

    # ---- oracle success rate ----
    oracle_success = np.zeros((n_L, n_C), dtype=bool)
    oracle_time = np.full((n_L, n_C), np.nan, dtype=np.float64)
    for i in range(n_L):
        for j in range(n_C):
            if optimal_masked[i, j] >= 0:
                oracle_success[i, j] = True
                oracle_time[i, j] = float(times[i, j, optimal_masked[i, j]])
    oracle_rate = float(oracle_success.mean())
    oracle_avg_time = (
        float(oracle_time[np.isfinite(oracle_time)].mean())
        if oracle_success.any() else float("nan")
    )

    # ---- partner-blind success rate ----
    # Blind policy: pick a fixed allocation per LAYOUT (does not read cap).
    # We take the max over the two possible fixed choices of the mean success
    # over the capability pool (i.e. the BEST partner-blind policy).
    blind_success_A = success[:, :, 0].mean(axis=1)   # (L,) mean over cap
    blind_success_B = success[:, :, 1].mean(axis=1)
    blind_choice = np.where(blind_success_A >= blind_success_B, 0, 1)  # (L,)
    # Success under blind: pick blind_choice[i] alloc for every cap.
    blind_success = np.zeros((n_L, n_C), dtype=bool)
    blind_time = np.full((n_L, n_C), np.nan, dtype=np.float64)
    for i in range(n_L):
        a = int(blind_choice[i])
        for j in range(n_C):
            if success[i, j, a]:
                blind_success[i, j] = True
                blind_time[i, j] = float(times[i, j, a])
    blind_rate = float(blind_success.mean())
    blind_avg_time = (
        float(blind_time[np.isfinite(blind_time)].mean())
        if blind_success.any() else float("nan")
    )

    # ---- Expected per-round reward under the training reward model ----
    # reward = success_reward on success minus step_penalty per step; on
    # a failure (both allocs infeasible) reward = -step_penalty * max_steps.
    def _reward_from_time_and_success(succ_mask, time_arr):
        # succ_mask: (L, C) bool. time_arr: (L, C) float or NaN.
        succ_reward = np.where(succ_mask, success_reward, 0.0)
        step_cost = np.where(
            succ_mask,
            step_penalty * np.nan_to_num(time_arr, nan=0.0),
            step_penalty * max_steps,   # failure eats the whole horizon
        )
        return float((succ_reward - step_cost).mean())

    oracle_reward = _reward_from_time_and_success(oracle_success, oracle_time)
    blind_reward  = _reward_from_time_and_success(blind_success, blind_time)

    return {
        "n_layouts": int(n_L),
        "n_cap_pairs": int(n_C),
        "cap_pairs": [list(map(int, c)) for c in cap_pairs],
        "flip_fraction": flip_fraction,
        "regret_when_wrong_mean": regret_mean,
        "regret_when_wrong_median": regret_median,
        "regret_fail_frac": regret_fail_frac,
        "oracle_success_rate": oracle_rate,
        "partner_blind_success_rate": blind_rate,
        "oracle_minus_blind": oracle_rate - blind_rate,
        "oracle_avg_time": oracle_avg_time,
        "blind_avg_time": blind_avg_time,
        "oracle_expected_reward_per_round": oracle_reward,
        "blind_expected_reward_per_round":  blind_reward,
        "oracle_minus_blind_reward":        oracle_reward - blind_reward,
        "step_penalty":                     step_penalty,
        "success_reward":                   success_reward,
    }


def _per_cap_success_breakdown(layouts, cap_pairs, max_steps):
    """Per-capability oracle/blind success (useful sanity)."""
    rows = []
    for cap in cap_pairs:
        # single-cap analysis is degenerate but still tells us the
        # per-cap oracle rate.
        analysis = _cap_pool_analysis(layouts, [cap], max_steps)
        rows.append({
            "cap": list(map(int, cap)),
            "oracle_success_rate": analysis["oracle_success_rate"],
            "oracle_avg_time":     analysis["oracle_avg_time"],
        })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layouts_dir",
                   default="dev/grids_final/layouts/val",
                   help="Directory of layout JSONs to analyse.")
    p.add_argument("--max_steps", type=int, default=15,
                   help="Per-round step horizon (matches training env).")
    p.add_argument("--step_penalty", type=float, default=0.01,
                   help="Per-step penalty (matches env reward).")
    p.add_argument("--success_reward", type=float, default=1.0,
                   help="Per-round success reward (matches env reward).")
    p.add_argument("--out", default="",
                   help="Optional path to dump the full JSON report.")
    args = p.parse_args()

    print(f"[cap-validation] loading layouts from {args.layouts_dir} ...")
    layouts = load_layouts(args.layouts_dir)
    print(f"[cap-validation] {len(layouts)} layouts loaded.")

    print("\n===== TRAINING capability pool =====")
    train_res = _cap_pool_analysis(
        layouts, TRAINING_CAPABILITY_PAIRS, args.max_steps,
        step_penalty=args.step_penalty, success_reward=args.success_reward,
    )
    for k, v in train_res.items():
        if k not in ("cap_pairs",):
            print(f"    {k:>28s}: {v}")

    print("\n===== HELD-OUT capability pool =====")
    heldout_res = _cap_pool_analysis(
        layouts, HELDOUT_CAPABILITY_PAIRS, args.max_steps,
        step_penalty=args.step_penalty, success_reward=args.success_reward,
    )
    for k, v in heldout_res.items():
        if k not in ("cap_pairs",):
            print(f"    {k:>28s}: {v}")

    # Per-capability oracle sanity for the held-out set (quick to run).
    print("\n===== Per-cap oracle rate (HELD-OUT) =====")
    for row in _per_cap_success_breakdown(
        layouts, HELDOUT_CAPABILITY_PAIRS, args.max_steps,
    ):
        print(f"    cap=({row['cap'][0]},{row['cap'][1]})  "
              f"oracle_rate={row['oracle_success_rate']:.3f}  "
              f"avg_time={row['oracle_avg_time']:.2f}")

    print("\nHeadline gap (oracle - blind):")
    print(f"    training : {train_res['oracle_minus_blind']:+.3f}  "
          f"(oracle={train_res['oracle_success_rate']:.3f}, "
          f"blind={train_res['partner_blind_success_rate']:.3f})")
    print(f"    held-out : {heldout_res['oracle_minus_blind']:+.3f}  "
          f"(oracle={heldout_res['oracle_success_rate']:.3f}, "
          f"blind={heldout_res['partner_blind_success_rate']:.3f})")

    report = {
        "layouts_dir": os.path.abspath(args.layouts_dir),
        "n_layouts":   len(layouts),
        "max_steps":   int(args.max_steps),
        "capability_values": list(CAPABILITY_VALUES),
        "training_capability_pairs":
            [list(map(int, p)) for p in TRAINING_CAPABILITY_PAIRS],
        "heldout_capability_pairs":
            [list(map(int, p)) for p in HELDOUT_CAPABILITY_PAIRS],
        "training_pool_analysis": train_res,
        "heldout_pool_analysis":  heldout_res,
        "per_cap_heldout":
            _per_cap_success_breakdown(
                layouts, HELDOUT_CAPABILITY_PAIRS, args.max_steps,
            ),
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[cap-validation] wrote {args.out}")


if __name__ == "__main__":
    main()
