"""Message-conditional held-out analysis.

For every rollout we already record:
  * msg_t0           — message ego sent on the free-comm step
  * partner_goal     — the goal the partner committed to at t=1
  * correct_ego_goal — the complementary goal
  * ego trajectory   — full sequence of ego positions

This script groups episodes by ``msg_t0 ∈ {NONE, M0, M1}`` and computes:
    success rate, mean length, nav_fail, assignment_error
for each group.

For the NONE subgroup we additionally ask: *when does the ego's trajectory
first become consistent with the correct (complementary) goal?*
Two operational definitions:

  a) ``t_first_reduce`` — first t ≥ 1 at which the ego's BFS distance to
     the correct goal is strictly less than at ego_start. This is the
     earliest signed evidence that the ego is moving toward the right target.
     If ego never gets closer, we record ``max_steps``.

  b) ``t_first_reach`` — first t at which ego actually occupies the correct
     goal cell. Only defined for successful (or partially successful)
     episodes.

Under z=0.5 (NONE regime), the partner reveals its committed goal through
its motion. The partner's first move (t=1) already determines which goal
it's heading toward under the greedy BFS policy. So the earliest ego could
'know' the correct goal is at step t=2 (after observing partner's t=1
position). Under an idealized ego, ``t_first_reduce`` should therefore
cluster around 2. A larger value indicates ego either (i) doesn't infer
from partner motion, or (ii) infers but doesn't commit to the correct goal.

Usage:
    python dev/diagnose_msg_conditional.py \\
        --params dev/train_logs/stage_b_z1.0_seed1_<TS>.safetensors \\
        --trials 64 \\
        --out dev/train_logs/stage_b_msg_conditional.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "baselines" / "IPPO"))
sys.path.insert(0, str(_REPO / "dev"))
from ippo_rnn_coordination_grid import ActorCriticCommRNN, ScannedRNN
from diagnose_stage_b import _run_rollouts

from jaxmarl.wrappers.baselines import load_params
from jaxmarl.environments.coordination_grid import (
    CoordinationGrid, GOAL_UNSET, GOAL_RED, GOAL_BLUE,
)
from coordination_grid_baselines import _bfs_distances


MSG_NAME = {0: "NONE", 1: "M0", 2: "M1"}


def _msg_conditional(r):
    """r is the dict returned by diagnose_stage_b._run_rollouts."""
    env: CoordinationGrid = r["env"]
    dones = r["dones"]                             # (T, B)
    successes = r["successes"]                     # (T, B) float
    agent_pos = r["agent_pos"]                     # (T, B, 2, 2)
    partner_goal = r["partner_goal"]               # (T, B)
    pending_msg = r["pending_msg"]                 # (T, B)  msg AT that step
    layout_idxs = r["layout_idxs"]                 # (B,)
    T, B = dones.shape

    first_done = np.argmax(dones.astype(np.int32), axis=0)         # (B,)
    ep_length = first_done + 1                                      # (B,)
    msg_t0 = pending_msg[0]                                         # (B,)
    idx_for_pg = np.maximum(first_done, 1)
    partner_goal_final = partner_goal[idx_for_pg, np.arange(B)]     # (B,)

    red_xy = np.asarray(env.red_goals)[layout_idxs]                 # (B, 2)
    blue_xy = np.asarray(env.blue_goals)[layout_idxs]               # (B, 2)
    is_red_committed = (partner_goal_final == GOAL_RED)             # (B,)
    correct_xy = np.where(is_red_committed[:, None], blue_xy, red_xy)
    wrong_xy = np.where(is_red_committed[:, None], red_xy, blue_xy)

    # Precompute per-LAYOUT BFS distance-to-red / distance-to-blue tables so
    # we don't rerun BFS per episode. We only need distances relative to the
    # correct/wrong goals though, so store per-layout distance-to-RED and
    # distance-to-BLUE.
    wall_maps_np = np.asarray(env.wall_maps)                        # (K, H, W)
    red_goals_np = np.asarray(env.red_goals)                        # (K, 2)
    blue_goals_np = np.asarray(env.blue_goals)                      # (K, 2)
    K = env.n_layouts
    dist_red = np.stack([
        _bfs_distances(wall_maps_np[k], int(red_goals_np[k, 0]),
                       int(red_goals_np[k, 1]))
        for k in range(K)
    ], axis=0)                                                       # (K, H, W)
    dist_blue = np.stack([
        _bfs_distances(wall_maps_np[k], int(blue_goals_np[k, 0]),
                       int(blue_goals_np[k, 1]))
        for k in range(K)
    ], axis=0)

    # For each column b: dist(ego_pos[t], correct_goal) time-series.
    ego_pos_seq = agent_pos[:, :, 0, :]                              # (T, B, 2)

    ep_records = []
    ego_reached_correct = np.zeros(B, dtype=bool)
    ego_reached_wrong = np.zeros(B, dtype=bool)
    partner_reached_committed = np.zeros(B, dtype=bool)
    succ_terminal = successes[first_done, np.arange(B)].astype(bool)
    d_correct_traj: list[np.ndarray] = []                            # per-ep, len = ep_length
    t_first_reduce = np.full(B, -1, dtype=np.int32)                  # -1 if never
    t_first_reach_correct = np.full(B, -1, dtype=np.int32)           # -1 if never

    for b in range(B):
        end = int(first_done[b]) + 1
        li = int(layout_idxs[b])
        c = correct_xy[b]
        w = wrong_xy[b]
        d_red_layout = dist_red[li]
        d_blue_layout = dist_blue[li]
        d_correct_layout = d_blue_layout if is_red_committed[b] else d_red_layout
        # distance along ego trajectory to CORRECT goal (or -1 if unreachable)
        ep_ego = ego_pos_seq[:end, b]                                # (end, 2)
        pp = agent_pos[:end, b, 1, :]                                # partner
        d_seq = np.array([
            int(d_correct_layout[int(y), int(x)]) for (x, y) in ep_ego
        ])
        d_correct_traj.append(d_seq)
        ego_reached_correct[b] = bool(np.any(np.all(ep_ego == c, axis=-1)))
        ego_reached_wrong[b] = bool(np.any(np.all(ep_ego == w, axis=-1)))
        partner_reached_committed[b] = bool(
            np.any(np.all(pp == w, axis=-1))
        )
        # first t >= 1 with d_seq[t] < d_seq[0]
        d0 = d_seq[0]
        reduce_idxs = np.where((np.arange(len(d_seq)) >= 1) & (d_seq < d0))[0]
        if len(reduce_idxs) > 0:
            t_first_reduce[b] = int(reduce_idxs[0])
        # first t with ego on correct goal (i.e., d_seq[t] == 0, if reachable)
        if (d_seq == 0).any():
            t_first_reach_correct[b] = int(np.argmax(d_seq == 0))

    # Bucket per episode.
    outcomes = np.empty(B, dtype=object)
    for b in range(B):
        if partner_goal_final[b] == GOAL_UNSET:
            outcomes[b] = "partner_unset"
        elif succ_terminal[b]:
            outcomes[b] = "success"
        elif ego_reached_wrong[b] and not ego_reached_correct[b]:
            outcomes[b] = "assignment_error"
        elif not ego_reached_correct[b] and not ego_reached_wrong[b]:
            outcomes[b] = "nav_fail"
        elif ego_reached_correct[b] and not partner_reached_committed[b]:
            outcomes[b] = "partner_blocked"
        else:
            outcomes[b] = "timing_mismatch"

    return {
        "n_layouts": K,
        "n_episodes": B,
        "msg_t0": msg_t0,
        "partner_goal_final": partner_goal_final,
        "correct_xy": correct_xy,
        "ep_length": ep_length,
        "succ_terminal": succ_terminal,
        "outcomes": outcomes,
        "ego_reached_correct": ego_reached_correct,
        "ego_reached_wrong": ego_reached_wrong,
        "partner_reached_committed": partner_reached_committed,
        "t_first_reduce": t_first_reduce,
        "t_first_reach_correct": t_first_reach_correct,
        "d_correct_traj": d_correct_traj,     # list of variable-length arrays
        "d0_ego_to_correct": np.array([tr[0] for tr in d_correct_traj]),
        "max_steps": int(env.max_steps),
    }


def _fmt_row(label, n, succ, length, nav, assign):
    return (
        f"    {label:>6s}   n={n:>5d}   "
        f"success={succ:.3f}   mean_steps={length:5.2f}   "
        f"nav_fail={nav:.3f}   assignment_error={assign:.3f}"
    )


def _summarize(split_name, c):
    B = c["n_episodes"]
    outcomes = c["outcomes"]
    msg_t0 = c["msg_t0"]
    ep_length = c["ep_length"]
    print(f"\n=== {split_name} ===")
    print(f"  {B} episodes across {c['n_layouts']} layouts")

    overall_succ = float((outcomes == "success").mean())
    overall_nav = float((outcomes == "nav_fail").mean())
    overall_ae = float((outcomes == "assignment_error").mean())
    print(_fmt_row("ALL", B, overall_succ, float(ep_length.mean()),
                   overall_nav, overall_ae))
    for m in (0, 1, 2):
        mask = (msg_t0 == m)
        n = int(mask.sum())
        if n == 0:
            continue
        succ = float((outcomes[mask] == "success").mean())
        nav = float((outcomes[mask] == "nav_fail").mean())
        ae = float((outcomes[mask] == "assignment_error").mean())
        length = float(ep_length[mask].mean())
        print(_fmt_row(MSG_NAME[m], n, succ, length, nav, ae))

    # NONE-specific trajectory analysis
    none_mask = (msg_t0 == 0)
    if none_mask.any():
        d0 = c["d0_ego_to_correct"][none_mask]
        tfr = c["t_first_reduce"][none_mask]
        tfrc = c["t_first_reach_correct"][none_mask]
        # 'never' encoded as -1 -> convert to max_steps for reporting purposes
        tfr_report = np.where(tfr < 0, c["max_steps"], tfr).astype(np.float32)
        tfrc_report = np.where(tfrc < 0, c["max_steps"], tfrc).astype(np.float32)
        # Cases where ego makes NO progress at all (never reduced dist)
        frac_no_progress = float((tfr < 0).mean())
        frac_ever_reached_correct = float((tfrc >= 0).mean())

        print(f"\n  [NONE-only trajectory analysis]  (n={int(none_mask.sum())})")
        print(f"    ego BFS distance to CORRECT goal at ego_start:  "
              f"mean={float(d0.mean()):.2f}  median={float(np.median(d0)):.1f}")
        print(f"    fraction NONE-episodes with NO forward progress: "
              f"{frac_no_progress:.3f}")
        print(f"    fraction NONE-episodes where ego EVER reached correct: "
              f"{frac_ever_reached_correct:.3f}")
        print(f"    t_first_reduce  (first step ego is CLOSER to correct "
              f"than at ego_start):")
        print(f"      mean={float(tfr_report.mean()):5.2f}  "
              f"median={float(np.median(tfr_report)):.1f}  "
              f"histogram over [1..{c['max_steps']}]:")
        bins = np.arange(1, c["max_steps"] + 2)
        hist, _ = np.histogram(tfr_report, bins=bins)
        for b_lo, cnt in zip(bins[:-1], hist):
            if cnt == 0:
                continue
            print(f"        t={int(b_lo):>2d}: {int(cnt):>5d}   "
                  f"({cnt / len(tfr_report):.3f})")
        # Split t_first_reduce distribution by ultimate outcome
        for label in ("success", "assignment_error", "nav_fail",
                      "partner_blocked", "timing_mismatch"):
            sub = (outcomes[none_mask] == label)
            n_sub = int(sub.sum())
            if n_sub == 0:
                continue
            sub_tfr = tfr_report[sub]
            print(f"      | outcome={label:<18s} n={n_sub:>4d}  "
                  f"t_first_reduce mean={float(sub_tfr.mean()):.2f}  "
                  f"median={float(np.median(sub_tfr)):.1f}")


def _to_json(c):
    return {
        "n_episodes": int(c["n_episodes"]),
        "n_layouts": int(c["n_layouts"]),
        "max_steps": int(c["max_steps"]),
        "by_msg_t0": {
            MSG_NAME[m]: {
                "n": int((c["msg_t0"] == m).sum()),
                "success": float(
                    (c["outcomes"][c["msg_t0"] == m] == "success").mean()
                ) if int((c["msg_t0"] == m).sum()) > 0 else 0.0,
                "nav_fail": float(
                    (c["outcomes"][c["msg_t0"] == m] == "nav_fail").mean()
                ) if int((c["msg_t0"] == m).sum()) > 0 else 0.0,
                "assignment_error": float(
                    (c["outcomes"][c["msg_t0"] == m] == "assignment_error").mean()
                ) if int((c["msg_t0"] == m).sum()) > 0 else 0.0,
                "mean_length": float(
                    c["ep_length"][c["msg_t0"] == m].mean()
                ) if int((c["msg_t0"] == m).sum()) > 0 else 0.0,
            } for m in (0, 1, 2)
        },
        "none_trajectory": {
            "n_episodes": int((c["msg_t0"] == 0).sum()),
            "t_first_reduce": {
                "raw": c["t_first_reduce"][c["msg_t0"] == 0].tolist(),
            },
            "t_first_reach_correct": {
                "raw": c["t_first_reach_correct"][c["msg_t0"] == 0].tolist(),
            },
            "d0_ego_to_correct": {
                "raw": c["d0_ego_to_correct"][c["msg_t0"] == 0].tolist(),
            },
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", required=True)
    p.add_argument("--config",
                   default=str(_REPO / "baselines/IPPO/config"
                                        / "ippo_rnn_coordination_grid.yaml"))
    p.add_argument("--val-dir",
                   default=str(_REPO / "dev/grids/layouts/val"))
    p.add_argument("--test-dir",
                   default=str(_REPO / "dev/grids/layouts/test"))
    p.add_argument("--trials", type=int, default=64)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config))
    params = load_params(args.params)
    print(f"loaded params from {args.params}")

    key = jax.random.PRNGKey(args.seed)
    all_json = {}
    for split, ldir in (("val", args.val_dir), ("test", args.test_dir)):
        key, sub = jax.random.split(key)
        print(f"\nrolling out {split} × {args.trials} trials/layout...")
        r = _run_rollouts(params, cfg, ldir, sub, args.trials)
        c = _msg_conditional(r)
        _summarize(split, c)
        all_json[split] = _to_json(c)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_json, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
