"""Pre-PPO analytical sanity check for the capability-vector design.

For each (layout, capability_pair) we analytically compute how many env
steps it would take for the ego + partner to complete each of the two
possible role allocations:

    * Allocation A: ego → RED,  partner → BLUE  (partner uses d_B).
    * Allocation B: ego → BLUE, partner → RED   (partner uses d_R).

Under the reference-style delay semantics (matches CoordinationGrid v2):

    call_success(alloc) = max(
        1 + BFS(ego_start,     ego_goal),
        2 + (BFS(partner_start, partner_goal) - 1) * (partner_delay + 1)
    )

    success(alloc) = (call_success <= max_steps)

We then report, over the layout corpus and the training / test capability
pools separately:

    * fraction of layouts where the optimal allocation flips across
      capability profiles;
    * mean / median regret in "steps to complete" for the wrong allocation
      (only over layouts where at least one allocation is feasible);
    * ORACLE success rate: always pick the allocation that is optimal for
      *this* (layout, cap);
    * PARTNER-BLIND success rate: pick a fixed allocation per layout
      (the one that maximises mean success across the capability pool),
      regardless of cap.

We only proceed to PPO if oracle > blind by a meaningful reward margin
on the TRAINING pool (test-pool numbers are reported for context only).

Usage:
    python dev/capability_validation.py \\
        --layouts_dir dev/grids_capability_selected/layouts/train \\
        --max_steps 100 \\
        --step_penalty 0.01 \\
        --out dev/train_logs/capability_validation.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capability_selection import (  # noqa: E402
    INFEASIBLE,
    completion_time,
    _reward_from_time,
)

from jaxmarl.environments.coordination_grid import (  # noqa: E402
    TRAIN_CAPABILITY_PAIRS,
    TEST_CAPABILITY_PAIRS,
    bfs_distance_map,
)


def load_layouts(layouts_dir: str) -> List[dict]:
    """Load layout JSONs and precompute BFS distances from ego_start /
    partner_start to each of the two goals.
    """
    paths = sorted(glob.glob(os.path.join(layouts_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"No *.json under {layouts_dir}")
    out: List[dict] = []
    for p in paths:
        with open(p, "r") as f:
            raw = json.load(f)
        grid = np.array(raw["grid"], dtype=np.int32)
        wall = (grid == 1)

        def rc2xy(rc):
            return np.array([rc[1], rc[0]], dtype=np.int32)

        ego = rc2xy(raw["ego_start"])
        partner = rc2xy(raw["partner_start"])
        red = rc2xy(raw["red_goal"])
        blue = rc2xy(raw["blue_goal"])
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


def _cap_pool_analysis(layouts: List[dict],
                       cap_pairs: Sequence[Tuple[int, int]],
                       max_steps: int,
                       step_penalty: float = 0.01,
                       success_reward: float = 1.0) -> Dict:
    """Analytical per-(layout, capability, allocation) evaluation."""
    n_L = len(layouts)
    n_C = len(cap_pairs)
    # times[i, j, 0] = allocation A completion; times[i, j, 1] = allocation B.
    times = np.full((n_L, n_C, 2), INFEASIBLE, dtype=np.int64)
    for i, ld in enumerate(layouts):
        for j, (d_r, d_b) in enumerate(cap_pairs):
            ta = completion_time(
                ego_path=ld["ego_to_red"],
                partner_path=ld["partner_to_blue"],
                partner_delay=int(d_b),
                max_steps=max_steps,
            )
            tb = completion_time(
                ego_path=ld["ego_to_blue"],
                partner_path=ld["partner_to_red"],
                partner_delay=int(d_r),
                max_steps=max_steps,
            )
            times[i, j, 0] = ta
            times[i, j, 1] = tb

    success = times < INFEASIBLE
    optimal = np.argmin(times, axis=2)
    both_bad = ~success.any(axis=2)
    optimal_masked = np.where(both_bad, -1, optimal)

    flip_flags = np.zeros(n_L, dtype=bool)
    for i in range(n_L):
        row_defined = optimal_masked[i][optimal_masked[i] >= 0]
        flip_flags[i] = (len(set(row_defined.tolist())) > 1) if len(row_defined) >= 2 else False
    flip_fraction = float(flip_flags.mean())

    # Regret when wrong.
    regrets: List[int] = []
    fail_both = 0
    for i in range(n_L):
        for j in range(n_C):
            ta, tb = int(times[i, j, 0]), int(times[i, j, 1])
            if ta >= INFEASIBLE and tb >= INFEASIBLE:
                fail_both += 1
                continue
            best = min(ta, tb)
            worst = max(ta, tb)
            regrets.append(
                (max_steps + 1 - best) if worst >= INFEASIBLE else (worst - best)
            )
    regrets_arr = np.asarray(regrets, dtype=np.int64) if regrets else np.array([0])
    regret_mean = float(regrets_arr.mean())
    regret_median = float(np.median(regrets_arr))
    regret_fail_frac = fail_both / max(n_L * n_C, 1)

    # Oracle policy.
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

    # Partner-blind policy (best fixed alloc per layout, no cap info).
    # Choose the fixed allocation per layout by EXPECTED REWARD across the
    # capability pool (matching capability_selection.evaluate_layout). Using
    # success rate is wrong at max_steps=100 where nearly every (layout,cap)
    # succeeds under either allocation — success-rate ties trivially, and
    # the reported "blind reward" ends up equal to fixed-A instead of
    # max(fixed-A, fixed-B).
    rewards = np.full((n_L, n_C, 2), 0.0, dtype=np.float64)
    for i in range(n_L):
        for j in range(n_C):
            rewards[i, j, 0] = _reward_from_time(
                int(times[i, j, 0]), max_steps, step_penalty, success_reward,
            )
            rewards[i, j, 1] = _reward_from_time(
                int(times[i, j, 1]), max_steps, step_penalty, success_reward,
            )
    fixed_A_reward_per_layout = rewards[:, :, 0].mean(axis=1)   # (L,)
    fixed_B_reward_per_layout = rewards[:, :, 1].mean(axis=1)
    blind_choice = np.where(
        fixed_A_reward_per_layout >= fixed_B_reward_per_layout, 0, 1,
    )                                                            # (L,)
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

    # Oracle reward from per-(layout,cap) reward-max allocation, matching
    # what capability_selection.evaluate_layout returns.
    oracle_reward_per_cell = np.maximum(rewards[:, :, 0], rewards[:, :, 1])
    oracle_reward = float(oracle_reward_per_cell.mean())
    # Blind reward using the per-layout best fixed allocation.
    blind_reward_per_cell = np.take_along_axis(
        rewards, blind_choice[:, None, None].repeat(n_C, axis=1), axis=2,
    ).squeeze(-1)
    blind_reward = float(blind_reward_per_cell.mean())

    # Worst-case successful completion time (for max_steps guidance).
    max_completed = int(
        oracle_time[np.isfinite(oracle_time)].max()
        if oracle_success.any() else 0
    )

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
        "blind_expected_reward_per_round": blind_reward,
        "oracle_minus_blind_reward": oracle_reward - blind_reward,
        "max_observed_oracle_completion": max_completed,
        "step_penalty": step_penalty,
        "success_reward": success_reward,
    }


def _per_cap_success_breakdown(layouts, cap_pairs, max_steps):
    rows = []
    for cap in cap_pairs:
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
                   default="dev/grids_capability_selected/layouts/train")
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--step_penalty", type=float, default=0.01)
    p.add_argument("--success_reward", type=float, default=1.0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    print(f"[cap-validation] loading layouts from {args.layouts_dir} ...")
    layouts = load_layouts(args.layouts_dir)
    print(f"[cap-validation] {len(layouts)} layouts loaded.")

    print("\n===== TRAIN capability pool =====")
    train_res = _cap_pool_analysis(
        layouts, TRAIN_CAPABILITY_PAIRS, args.max_steps,
        step_penalty=args.step_penalty, success_reward=args.success_reward,
    )
    for k, v in train_res.items():
        if k != "cap_pairs":
            print(f"    {k:>32s}: {v}")

    print("\n===== TEST (novel) capability pool =====")
    test_res = _cap_pool_analysis(
        layouts, TEST_CAPABILITY_PAIRS, args.max_steps,
        step_penalty=args.step_penalty, success_reward=args.success_reward,
    )
    for k, v in test_res.items():
        if k != "cap_pairs":
            print(f"    {k:>32s}: {v}")

    print("\n===== Per-cap oracle rate (TEST novel) =====")
    for row in _per_cap_success_breakdown(
        layouts, TEST_CAPABILITY_PAIRS, args.max_steps,
    ):
        print(f"    cap=({row['cap'][0]},{row['cap'][1]})  "
              f"oracle_rate={row['oracle_success_rate']:.3f}  "
              f"avg_time={row['oracle_avg_time']:.2f}")

    print("\nHeadline reward gap (oracle - blind):")
    print(f"    train : {train_res['oracle_minus_blind_reward']:+.4f}")
    print(f"    test  : {test_res['oracle_minus_blind_reward']:+.4f}")

    report = {
        "layouts_dir": os.path.abspath(args.layouts_dir),
        "n_layouts":   len(layouts),
        "max_steps":   int(args.max_steps),
        "train_capability_pairs":
            [list(map(int, p)) for p in TRAIN_CAPABILITY_PAIRS],
        "test_capability_pairs":
            [list(map(int, p)) for p in TEST_CAPABILITY_PAIRS],
        "train_pool_analysis": train_res,
        "test_pool_analysis":  test_res,
        "per_cap_test":
            _per_cap_success_breakdown(
                layouts, TEST_CAPABILITY_PAIRS, args.max_steps,
            ),
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[cap-validation] wrote {args.out}")


if __name__ == "__main__":
    main()
