"""Per-rollout diagnostic for held-out CoordinationGrid evaluation.

Loads a trained policy, runs it over the val + test layout pools, and
records for every episode:

  * ``msg_t0``               — the message the ego sent on the free-comm step
  * ``partner_goal``         — the goal the partner committed to at t=1
  * ``correct_ego_goal``     — the complementary goal the ego should target
  * ``ego_reached_correct``  — did the ego's position ever equal that goal?
  * ``ego_reached_wrong``    — did the ego ever occupy the partner's goal
                                (== "targeted same goal" failure)?
  * ``partner_reached_goal`` — did the partner ever reach its committed goal?
  * ``success``, ``length``

From these each episode is classified into exactly one bucket:

  success                 — coordinated on complementary goals
  assignment_error        — ego walked to the same goal the partner is heading
                            to, without ever visiting the correct one
  nav_fail                — ego never reached either goal
  partner_blocked         — ego reached correct goal but partner never arrived
                            at its committed goal (likely ego blocking path)
  timing_mismatch         — both reached their goals but not on the same step
                            (rare because partner STAYs once on goal)
  partner_unset           — episode ended with ``partner_goal == UNSET`` (only
                            possible if the episode ended before t=1 — shouldn't
                            happen for the current env)

Usage:
    python dev/diagnose_stage_b.py \\
        --params dev/train_logs/stage_b_z1.0_seed1_<TS>.safetensors \\
        --val-dir dev/grids/layouts/val \\
        --test-dir dev/grids/layouts/test \\
        --trials 64 \\
        --out dev/train_logs/stage_b_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

# Make the trainer module importable (for the network class).
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "baselines" / "IPPO"))
from ippo_rnn_coordination_grid import (
    ActorCriticCommRNN,
    ScannedRNN,
)

from jaxmarl.wrappers.baselines import load_params
from jaxmarl.environments.coordination_grid import (
    CoordinationGrid,
    GOAL_UNSET, GOAL_RED, GOAL_BLUE,
)


MSG_NAME = {0: "NONE", 1: "M0", 2: "M1"}


def _run_rollouts(params, config, layouts_dir, key, n_trials):
    """Rollout ``n_trials`` per layout under jitted vmap. Returns numpy arrays
    of shape (T, B) or (T, B, ...) plus per-column layout metadata."""
    ekw = config["ENV_KWARGS"]
    # Config schema BC: prefer scalar partner_z if present; else pick the
    # first entry of partner_z_values (Stage C+D configs); else default 0.5.
    if "partner_z" in ekw:
        pz = float(ekw["partner_z"])
    elif ekw.get("partner_z_values"):
        pz = float(list(ekw["partner_z_values"])[0])
    else:
        pz = 0.5
    env = CoordinationGrid(
        layouts_dir=layouts_dir,
        partner_z=pz,
        max_steps=ekw["max_steps"],
        step_penalty=ekw.get("step_penalty", 0.01),
        success_reward=ekw.get("success_reward", 1.0),
        # Stage-B-style single-round diagnostic — force single-round even if
        # config was set up for multi-round training.
        rounds_per_episode=1,
    )
    network = ActorCriticCommRNN(action_dim=env.n_ego_actions, config=config)
    K = env.n_layouts
    N = int(n_trials)
    B = K * N
    layout_idxs = jnp.repeat(jnp.arange(K, dtype=jnp.int32), N)  # (B,)

    key_reset, key_step = jax.random.split(key)
    reset_keys = jax.random.split(key_reset, B)
    obs, states = jax.vmap(env.reset_to_layout)(reset_keys, layout_idxs)
    hstate0 = ScannedRNN.initialize_carry(B, config["GRU_HIDDEN_DIM"])
    done_prev0 = jnp.zeros((B,), dtype=bool)

    @jax.jit
    def rollout(params, obs, states, hstate, done_prev, key):
        def body(carry, _):
            obs, states, hstate, done_prev, key = carry
            obs_a0 = obs["agent_0"]
            obs_in = jax.tree_util.tree_map(lambda x: x[None, :], obs_a0)
            done_in = done_prev[None, :]
            hstate, pi, _ = network.apply(params, hstate, (obs_in, done_in))
            key, ka, ks = jax.random.split(key, 3)
            action = pi.sample(seed=ka).squeeze(0)                 # (B,)
            step_keys = jax.random.split(ks, B)
            obs, states, reward, done, info = jax.vmap(
                env.step_env, in_axes=(0, 0, {"agent_0": 0})
            )(step_keys, states, {"agent_0": action})
            return (obs, states, hstate, done["__all__"], key), (
                action,
                info["success"].astype(jnp.float32),
                done["__all__"],
                states.agent_pos,          # (B, 2, 2)
                states.partner_goal,       # (B,)
                states.pending_message,    # (B,) — ego's msg from THIS step
            )
        init = (obs, states, hstate, done_prev, key)
        _, out = jax.lax.scan(body, init, None, length=env.max_steps)
        return out

    (actions, successes, dones,
     agent_pos, partner_goal, pending_msg) = rollout(
        params, obs, states, hstate0, done_prev0, key_step
    )

    return {
        "env": env,
        "layout_idxs": np.asarray(layout_idxs),
        "actions": np.asarray(actions),                # (T, B)
        "successes": np.asarray(successes),            # (T, B) float
        "dones": np.asarray(dones),                    # (T, B) bool
        "agent_pos": np.asarray(agent_pos),            # (T, B, 2, 2) int
        "partner_goal": np.asarray(partner_goal),      # (T, B) int
        "pending_msg": np.asarray(pending_msg),        # (T, B) int
    }


def _classify(r):
    """Post-process the arrays into per-episode outcomes."""
    env: CoordinationGrid = r["env"]
    dones = r["dones"]                               # (T, B)
    successes = r["successes"]
    agent_pos = r["agent_pos"]
    partner_goal = r["partner_goal"]
    pending_msg = r["pending_msg"]
    layout_idxs = r["layout_idxs"]
    T, B = dones.shape

    # First done step per column (env horizon = T so guaranteed to fire).
    first_done = np.argmax(dones.astype(np.int32), axis=0)          # (B,)
    ep_length = first_done + 1                                       # (B,)

    # t=0 message: state.pending_message after the first step_env call.
    msg_t0 = pending_msg[0]                                          # (B,)

    # Partner goal at terminal (committed at t=1; unchanged thereafter).
    # Take max(first_done, 1) defensively — if first_done==0, no commit yet.
    idx_for_pg = np.maximum(first_done, 1)
    partner_goal_final = partner_goal[idx_for_pg, np.arange(B)]      # (B,)

    # Per-column goal geometry.
    red_xy = np.asarray(env.red_goals)[layout_idxs]                  # (B, 2)
    blue_xy = np.asarray(env.blue_goals)[layout_idxs]                # (B, 2)
    is_red_committed = (partner_goal_final == GOAL_RED)              # (B,)
    partner_goal_xy = np.where(is_red_committed[:, None], red_xy, blue_xy)
    correct_ego_goal_xy = np.where(is_red_committed[:, None], blue_xy, red_xy)
    correct_ego_goal_id = np.where(is_red_committed, GOAL_BLUE, GOAL_RED)

    # Did ego / partner ever visit each goal? Iterate up to (and including)
    # the terminal step; post-terminal frames are from auto-reset and don't
    # belong to this episode.
    ego_pos_seq = agent_pos[:, :, 0, :]                              # (T, B, 2)
    partner_pos_seq = agent_pos[:, :, 1, :]                          # (T, B, 2)

    ego_reached_correct = np.zeros(B, dtype=bool)
    ego_reached_wrong = np.zeros(B, dtype=bool)
    partner_reached_committed = np.zeros(B, dtype=bool)
    for b in range(B):
        end = int(first_done[b]) + 1  # inclusive of terminal
        ep = ego_pos_seq[:end, b]
        pp = partner_pos_seq[:end, b]
        c = correct_ego_goal_xy[b]
        w = partner_goal_xy[b]
        ego_reached_correct[b] = bool(np.any(np.all(ep == c, axis=-1)))
        ego_reached_wrong[b] = bool(np.any(np.all(ep == w, axis=-1)))
        partner_reached_committed[b] = bool(np.any(np.all(pp == w, axis=-1)))

    succ_terminal = successes[first_done, np.arange(B)].astype(bool)

    outcomes = np.empty(B, dtype=object)
    for b in range(B):
        if partner_goal_final[b] == GOAL_UNSET:
            outcomes[b] = "partner_unset"
            continue
        if succ_terminal[b]:
            outcomes[b] = "success"
        elif ego_reached_wrong[b] and not ego_reached_correct[b]:
            outcomes[b] = "assignment_error"
        elif not ego_reached_correct[b] and not ego_reached_wrong[b]:
            outcomes[b] = "nav_fail"
        elif ego_reached_correct[b] and not partner_reached_committed[b]:
            outcomes[b] = "partner_blocked"
        else:
            # Both reached their goals but not simultaneously — rare because
            # BFS-greedy partner STAYs at goal. Group under timing_mismatch.
            outcomes[b] = "timing_mismatch"

    return {
        "T": T, "B": B, "K": int(env.n_layouts), "N": int(B // env.n_layouts),
        "first_done": first_done,
        "ep_length": ep_length,
        "msg_t0": msg_t0,
        "partner_goal_final": partner_goal_final,
        "correct_ego_goal_id": correct_ego_goal_id,
        "ego_reached_correct": ego_reached_correct,
        "ego_reached_wrong": ego_reached_wrong,
        "partner_reached_committed": partner_reached_committed,
        "succ_terminal": succ_terminal,
        "outcomes": outcomes,
        "layout_paths": list(env.layout_paths),
        "layout_idxs": layout_idxs,
    }


def _summarize(c):
    """Aggregate + pretty-print the per-episode classification."""
    B = c["B"]
    K = c["K"]
    N = c["N"]
    outcomes = c["outcomes"]
    counts = Counter(outcomes.tolist())
    order = ["success", "assignment_error", "partner_blocked",
             "nav_fail", "timing_mismatch", "partner_unset"]

    print(f"  {B} episodes across {K} layouts × {N} trials, "
          f"mean length {c['ep_length'].mean():.1f}")
    for k in order:
        n = counts.get(k, 0)
        if n == 0 and k not in ("success", "assignment_error",
                                "partner_blocked", "nav_fail"):
            continue
        print(f"    {k:>22s}: {n/B:.3f}  ({n}/{B})")

    # t=0 message marginal
    msg = c["msg_t0"]
    print(f"  t0 msg marginal: "
          f"NONE={float((msg == 0).mean()):.3f}  "
          f"M0={float((msg == 1).mean()):.3f}  "
          f"M1={float((msg == 2).mean()):.3f}")

    # Message × partner-goal contingency (only for M0/M1 rows — NONE is 50/50).
    pg = c["partner_goal_final"]
    for m in (0, 1, 2):
        rows = (msg == m)
        n = int(rows.sum())
        if n == 0:
            continue
        pr = float(((pg == GOAL_RED) & rows).sum()) / max(n, 1)
        pb = float(((pg == GOAL_BLUE) & rows).sum()) / max(n, 1)
        print(f"    msg={MSG_NAME[m]:>4s}  n={n:>4d}   "
              f"P(partner=RED)={pr:.3f}  P(partner=BLUE)={pb:.3f}")

    # Conditional: on failures, what did ego do?
    fail_mask = (outcomes != "success")
    if fail_mask.any():
        rc = c["ego_reached_correct"][fail_mask].mean()
        rw = c["ego_reached_wrong"][fail_mask].mean()
        pb2 = c["partner_reached_committed"][fail_mask].mean()
        print(f"  on failures (n={int(fail_mask.sum())}):")
        print(f"    ego ever at CORRECT goal:  {rc:.3f}")
        print(f"    ego ever at WRONG goal:    {rw:.3f}")
        print(f"    partner reached its goal:  {pb2:.3f}")


def _to_json(c):
    """Cheap-to-serialize summary payload."""
    outcomes = c["outcomes"]
    counts = Counter(outcomes.tolist())
    payload = {
        "n_episodes": int(c["B"]),
        "n_layouts": int(c["K"]),
        "n_trials_per_layout": int(c["N"]),
        "counts": dict(counts),
        "fractions": {k: v / c["B"] for k, v in counts.items()},
        "mean_ep_length": float(c["ep_length"].mean()),
        "msg_t0_marginal": {
            "NONE": float((c["msg_t0"] == 0).mean()),
            "M0":   float((c["msg_t0"] == 1).mean()),
            "M1":   float((c["msg_t0"] == 2).mean()),
        },
        "per_layout": [],
    }
    # Per-layout classification
    for k in range(c["K"]):
        idxs = np.arange(k * c["N"], (k + 1) * c["N"])
        oc = outcomes[idxs]
        payload["per_layout"].append({
            "layout_idx": int(k),
            "layout_path": c["layout_paths"][k],
            "success_rate": float(c["succ_terminal"][idxs].mean()),
            "counts": dict(Counter(oc.tolist())),
            "n_ego_reached_correct": int(c["ego_reached_correct"][idxs].sum()),
            "n_ego_reached_wrong": int(c["ego_reached_wrong"][idxs].sum()),
            "n_partner_reached_committed": int(
                c["partner_reached_committed"][idxs].sum()
            ),
        })
    return payload


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--params", required=True,
                   help="Path to .safetensors from ippo_rnn_coordination_grid.py")
    p.add_argument("--config",
                   default=str(_REPO_ROOT
                                / "baselines/IPPO/config"
                                / "ippo_rnn_coordination_grid.yaml"),
                   help="Training yaml (used for network + env kwargs).")
    p.add_argument("--val-dir",
                   default=str(_REPO_ROOT / "dev/grids/layouts/val"))
    p.add_argument("--test-dir",
                   default=str(_REPO_ROOT / "dev/grids/layouts/test"))
    p.add_argument("--trials", type=int, default=64,
                   help="Rollouts per layout.")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default=None,
                   help="Optional JSON output path.")
    args = p.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config))
    params = load_params(args.params)
    print(f"loaded params from {args.params}")

    key = jax.random.PRNGKey(args.seed)
    all_json = {}
    for split, ldir in (("val", args.val_dir), ("test", args.test_dir)):
        key, sub = jax.random.split(key)
        print(f"\n=== {split}: {ldir} × {args.trials} trials/layout ===")
        r = _run_rollouts(params, cfg, ldir, sub, args.trials)
        c = _classify(r)
        _summarize(c)
        all_json[split] = _to_json(c)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_json, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
