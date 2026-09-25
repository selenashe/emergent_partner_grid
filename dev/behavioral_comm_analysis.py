"""Behavioral-communication analysis for CoordinationGrid rollouts.

For any subset of episodes with a well-defined ``correct_goal`` (the goal
partner did NOT commit to — so the ego should target it), this module
measures the incremental value of *observing partner motion* against the
random-choice-at-t=0 baseline.

The core observable per step is the ego's *intent* — which goal is the ego's
movement heading toward this step. Intent is derived from the BFS distances
to each goal:

    delta_correct = dist(ego[t-1], correct) - dist(ego[t], correct)   # +1 if closer
    delta_wrong   = dist(ego[t-1], wrong)   - dist(ego[t], wrong)
    intent[t] =
        CORRECT     if delta_correct > delta_wrong
        WRONG       if delta_wrong > delta_correct
        AMBIG       otherwise  (includes STATIC + both goals equally close/far)

Env-timing note (important for the indices below):

    step_env call    input state.time     what happens
    -------------    ----------------     ------------
      #1                    0             free-comm step; both agents FORCED STAY.
      #2                    1             ego's first real move; partner also
                                          moves for the first time (BFS-greedy).
                                          Ego has NO partner-motion evidence yet.
      #3                    2             ego's first move where the OBSERVED
                                          partner position has changed from
                                          partner_start. First step at which
                                          behavioral inference is possible.

So the intents array is indexed by (step_env call number - 1):
    intent[0]  = step #1 → always AMBIG (forced stay). Uninformative; skipped.
    intent[1]  = step #2, ego's first real move — the pre-observation baseline.
    intent[2]  = step #3, first post-observation move — the "did ego update on
                 partner motion?" step.

The interesting metrics:

  * ``alignment_curve[t]`` = P(intent[t] == CORRECT) over episodes.
    Under a NONE-msg ego with no info about partner: alignment[1] ~ 0.5.
    An ego that infers from partner motion: alignment[t] rises for t ≥ 2.

  * ``switch_matrix``: 3x3 confusion of intent[t=1] × intent[t=2]. The
    diagonal is "ego didn't change its intended target"; the off-diagonal
    entry ``(WRONG, CORRECT)`` counts episodes where ego SWITCHED toward the
    correct goal after observing partner's first move.

  * ``P(switch_to_correct_given_wrong_at_t1)`` — the headline number for
    behavioral-comm value: given the ego started wrong, how often does it
    fix itself after seeing partner move once?

Usage (CLI):
    python dev/behavioral_comm_analysis.py \\
        --params dev/train_logs/stage_b_z1.0_seed1_<TS>.safetensors \\
        --trials 64 \\
        --out   dev/train_logs/stage_b_behavioral_comm.json

Usage (as a library — from any future Stage-C/D/E analysis):
    from behavioral_comm_analysis import (
        run_rollouts, compute_intents_for_rollouts,
        alignment_curve, switch_summary, msg_conditional_alignment,
    )
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "baselines" / "IPPO"))
sys.path.insert(0, str(_REPO / "dev"))
from ippo_rnn_coordination_grid import ActorCriticCommRNN, ScannedRNN

from jaxmarl.wrappers.baselines import load_params
from jaxmarl.environments.coordination_grid import (
    CoordinationGrid,
    GOAL_UNSET, GOAL_RED, GOAL_BLUE,
)
from coordination_grid_baselines import _bfs_distances


MSG_NAME = {0: "NONE", 1: "M0", 2: "M1"}
INTENT_LABELS = ("CORRECT", "WRONG", "AMBIG")


# --------------------------------------------------------------------------- #
# Rollout — same shape as diagnose_stage_b._run_rollouts, kept local so this  #
# module has no dependency on that file's internals.                          #
# --------------------------------------------------------------------------- #

def run_rollouts(
    params, config, layouts_dir: str, key, n_trials_per_layout: int
) -> dict:
    """Jitted vmap rollout over K × N (layout, trial) pairs.

    Returns numpy arrays keyed for downstream analysis:
      - layout_idxs (B,)
      - actions (T, B), pending_msg (T, B), partner_goal (T, B)
      - dones (T, B), successes (T, B)
      - agent_pos (T, B, 2, 2)   [(x, y) for [ego, partner]]
      - env: the CoordinationGrid used (for post-hoc geometry lookups)
    """
    ekw = config["ENV_KWARGS"]
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
        # Single-round rollouts for the behavioral-comm analysis (this module
        # measures per-round intent trajectories; multi-round aggregation is
        # up to the caller).
        rounds_per_episode=1,
        augment_symmetries=False,
    )
    network = ActorCriticCommRNN(action_dim=env.n_ego_actions, config=config)
    K = env.n_layouts
    N = int(n_trials_per_layout)
    B = K * N
    layout_idxs = jnp.repeat(jnp.arange(K, dtype=jnp.int32), N)

    key_reset, key_step = jax.random.split(key)
    reset_keys = jax.random.split(key_reset, B)
    obs, states = jax.vmap(env.reset_to_layout)(reset_keys, layout_idxs)
    hstate = ScannedRNN.initialize_carry(B, config["GRU_HIDDEN_DIM"])
    done_prev = jnp.zeros((B,), dtype=bool)

    @jax.jit
    def rollout(params, obs, states, hstate, done_prev, key):
        def body(carry, _):
            obs, states, hstate, done_prev, key = carry
            oa = obs["agent_0"]
            obs_in = jax.tree_util.tree_map(lambda x: x[None, :], oa)
            done_in = done_prev[None, :]
            hstate, pi, _ = network.apply(params, hstate, (obs_in, done_in))
            key, ka, ks = jax.random.split(key, 3)
            action = pi.sample(seed=ka).squeeze(0)
            step_keys = jax.random.split(ks, B)
            obs, states, reward, done, info = jax.vmap(
                env.step_env, in_axes=(0, 0, {"agent_0": 0})
            )(step_keys, states, {"agent_0": action})
            return (obs, states, hstate, done["__all__"], key), (
                action,
                info["success"].astype(jnp.float32),
                done["__all__"],
                states.agent_pos,
                states.partner_goal,
                states.pending_message,
            )
        init = (obs, states, hstate, done_prev, key)
        _, out = jax.lax.scan(body, init, None, length=env.max_steps)
        return out

    (actions, successes, dones,
     agent_pos, partner_goal, pending_msg) = rollout(
        params, obs, states, hstate, done_prev, key_step
    )
    return {
        "env": env,
        "layout_idxs": np.asarray(layout_idxs),
        "actions": np.asarray(actions),
        "successes": np.asarray(successes),
        "dones": np.asarray(dones),
        "agent_pos": np.asarray(agent_pos),
        "partner_goal": np.asarray(partner_goal),
        "pending_msg": np.asarray(pending_msg),
    }


# --------------------------------------------------------------------------- #
# Intent computation                                                          #
# --------------------------------------------------------------------------- #

def _bfs_dist_tables(env: CoordinationGrid
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-layout BFS distance-to-RED and distance-to-BLUE. Shape (K, H, W)."""
    wall_maps_np = np.asarray(env.wall_maps)
    red_goals_np = np.asarray(env.red_goals)
    blue_goals_np = np.asarray(env.blue_goals)
    K = env.n_layouts
    dist_red = np.stack([
        _bfs_distances(wall_maps_np[k], int(red_goals_np[k, 0]),
                       int(red_goals_np[k, 1]))
        for k in range(K)
    ], axis=0)
    dist_blue = np.stack([
        _bfs_distances(wall_maps_np[k], int(blue_goals_np[k, 0]),
                       int(blue_goals_np[k, 1]))
        for k in range(K)
    ], axis=0)
    return dist_red, dist_blue


def compute_intents_for_rollouts(r: dict) -> dict:
    """From the rollouts dict returned by ``run_rollouts``, compute:

      - correct_goal_id (B,)                — GOAL_RED or GOAL_BLUE for each ep
      - dist_to_correct (T+1, B)            — ego BFS dist to correct goal per step
      - dist_to_wrong   (T+1, B)
      - intent          (T, B)              — one of {0:CORRECT, 1:WRONG, 2:AMBIG}
      - alive_step      (T, B) bool         — per-episode valid steps mask
                                              (True for steps s.t. s <= first_done)
      - msg_t0          (B,)
      - success_terminal (B,) bool
      - first_done      (B,)
    """
    env: CoordinationGrid = r["env"]
    dones = r["dones"]                     # (T, B)
    successes = r["successes"]
    agent_pos = r["agent_pos"]
    partner_goal = r["partner_goal"]
    pending_msg = r["pending_msg"]
    layout_idxs = r["layout_idxs"]
    T, B = dones.shape

    first_done = np.argmax(dones.astype(np.int32), axis=0)         # (B,)
    msg_t0 = pending_msg[0]                                         # (B,)

    # Partner's committed goal is stable from t=1 onward; take at first_done
    # (or max(first_done, 1) to be safe if an episode ended at index 0).
    idx_for_pg = np.maximum(first_done, 1)
    partner_goal_final = partner_goal[idx_for_pg, np.arange(B)]     # (B,)

    is_red_committed = (partner_goal_final == GOAL_RED)             # (B,)
    correct_goal_id = np.where(
        is_red_committed, GOAL_BLUE, GOAL_RED
    ).astype(np.int32)                                              # (B,)

    dist_red, dist_blue = _bfs_dist_tables(env)                     # each (K, H, W)

    # Per-step distances along ego trajectory. Includes step 0 (ego_start).
    # We stitch together: distance BEFORE step 1 = at ego_start = agent_pos_pre.
    # But run_rollouts recorded state.agent_pos AFTER each step_env; there is
    # no explicit "before step 1" record. Reconstruct step-0 (ego_start) from
    # env.ego_starts[layout_idxs] (ego doesn't move on the t=0 forced-STAY step).
    ego_start_xy = np.asarray(env.ego_starts)[layout_idxs]          # (B, 2)
    ego_pos_full = np.zeros((T + 1, B, 2), dtype=np.int32)          # (T+1, B, 2)
    ego_pos_full[0] = ego_start_xy
    ego_pos_full[1:] = agent_pos[:, :, 0, :]

    dist_to_correct = np.zeros((T + 1, B), dtype=np.int32)
    dist_to_wrong   = np.zeros((T + 1, B), dtype=np.int32)
    for b in range(B):
        li = int(layout_idxs[b])
        d_correct_layout = dist_blue[li] if is_red_committed[b] else dist_red[li]
        d_wrong_layout   = dist_red[li]  if is_red_committed[b] else dist_blue[li]
        for t in range(T + 1):
            x, y = int(ego_pos_full[t, b, 0]), int(ego_pos_full[t, b, 1])
            dist_to_correct[t, b] = int(d_correct_layout[y, x])
            dist_to_wrong[t, b]   = int(d_wrong_layout[y, x])

    # Intent per step. intent[t] uses (dist@t-1, dist@t) → labels for steps 1..T
    # We store intents indexed as intent[t=1..T-1] in a length-T array where
    # intent[i] is the intent AT movement step (i+1). But it's clearer to
    # store intent_at_step of shape (T, B) where intent_at_step[i] corresponds
    # to the i-th step_env call (state.time going 0->1, 1->2, ..., T-1->T).
    intent = np.full((T, B), -1, dtype=np.int8)  # -1 = uninitialized
    for i in range(T):
        # step i moves ego from ego_pos_full[i] to ego_pos_full[i+1].
        delta_c = dist_to_correct[i] - dist_to_correct[i + 1]
        delta_w = dist_to_wrong[i]   - dist_to_wrong[i + 1]
        intent[i][delta_c > delta_w] = 0  # CORRECT
        intent[i][delta_w > delta_c] = 1  # WRONG
        # AMBIG when equal (includes both zero i.e. STATIC, and both nonzero equal)
        intent[i][delta_c == delta_w] = 2

    alive_step = (np.arange(T)[:, None] <= first_done[None, :])
    succ_terminal = successes[first_done, np.arange(B)].astype(bool)

    return {
        "T": T, "B": B, "n_layouts": int(env.n_layouts),
        "msg_t0": msg_t0,
        "partner_goal_final": partner_goal_final,
        "correct_goal_id": correct_goal_id,
        "dist_to_correct": dist_to_correct,
        "dist_to_wrong": dist_to_wrong,
        "intent": intent,
        "alive_step": alive_step,
        "first_done": first_done,
        "success_terminal": succ_terminal,
        "ego_start_dist_to_correct": dist_to_correct[0],
        "ego_start_dist_to_wrong":   dist_to_wrong[0],
    }


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #

def alignment_curve(intents: dict, mask: Optional[np.ndarray] = None) -> dict:
    """P(intent[t] == CORRECT) vs t, averaged over the selected episode subset.

    ``mask`` is a (B,) bool array selecting episodes to include (default: all).
    Also returns the alignment against WRONG and the AMBIG rate.
    Only steps where the episode was still alive are counted.
    """
    intent = intents["intent"]                 # (T, B)
    alive = intents["alive_step"]              # (T, B)
    T, B = intent.shape
    if mask is None:
        mask = np.ones(B, dtype=bool)
    cols = mask
    p_correct = np.zeros(T, dtype=np.float32)
    p_wrong = np.zeros(T, dtype=np.float32)
    p_ambig = np.zeros(T, dtype=np.float32)
    n = np.zeros(T, dtype=np.int32)
    for t in range(T):
        alive_t = alive[t] & cols
        n_t = int(alive_t.sum())
        n[t] = n_t
        if n_t == 0:
            continue
        it = intent[t, alive_t]
        p_correct[t] = float((it == 0).mean())
        p_wrong[t]   = float((it == 1).mean())
        p_ambig[t]   = float((it == 2).mean())
    return {
        "p_correct": p_correct.tolist(),
        "p_wrong":   p_wrong.tolist(),
        "p_ambig":   p_ambig.tolist(),
        "n": n.tolist(),
    }


def switch_matrix(
    intents: dict, mask: Optional[np.ndarray] = None,
    t_from: int = 1, t_to: int = 2,
) -> dict:
    """3x3 confusion of intent[t_from] × intent[t_to] on episodes in ``mask``.

    Indices are 0-based into ``intents['intent']`` (which itself is one entry
    per step_env call). Defaults compare intent[1] (ego's first *real* move,
    with no partner-motion evidence) against intent[2] (first move after
    partner has moved once). intent[0] is the forced-STAY free-comm step,
    always AMBIG, so it is not a useful reference point.
    """
    intent = intents["intent"]
    alive = intents["alive_step"]
    B = intent.shape[1]
    if mask is None:
        mask = np.ones(B, dtype=bool)
    valid = mask & alive[t_from] & alive[t_to]
    a = intent[t_from, valid]
    b = intent[t_to, valid]
    m = np.zeros((3, 3), dtype=np.int64)
    for lbl_from in range(3):
        for lbl_to in range(3):
            m[lbl_from, lbl_to] = int(
                ((a == lbl_from) & (b == lbl_to)).sum()
            )
    total = int(valid.sum())
    # Row-normalized transitions.
    row_norm = m.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_norm = np.where(m.sum(axis=1, keepdims=True) > 0,
                            row_norm / m.sum(axis=1, keepdims=True),
                            0.0)
    return {
        "n": total,
        "counts": m.tolist(),
        "row_normalized": row_norm.tolist(),
        "t_from": t_from, "t_to": t_to,
        "labels": list(INTENT_LABELS),
    }


def summary_headline_stats(
    intents: dict, mask: Optional[np.ndarray] = None,
    t_pre: int = 1, t_post: int = 2,
) -> dict:
    """The one-number headline stats. All indices are 0-based into the
    intent array.

    By default we compare:
      * ``intent[t_pre=1]``  — ego's first REAL move (env step #2). No
        partner-motion evidence at this point.
      * ``intent[t_post=2]`` — ego's first move AFTER partner's first move
        is visible (env step #3).

    Returned:
      * ``p_intent_pre_correct``     — pre-observation alignment (baseline)
      * ``p_intent_post_correct``    — post-observation alignment
      * ``lift_post_over_pre``       — the incremental value of observing
      * ``p_wrong_to_correct``       — recovery rate on 'bad start' episodes
      * ``p_correct_to_wrong``       — abandonment rate on 'good start' eps
    """
    intent = intents["intent"]
    alive = intents["alive_step"]
    B = intent.shape[1]
    if mask is None:
        mask = np.ones(B, dtype=bool)
    v_pre  = mask & alive[t_pre]
    v_post = mask & alive[t_post]
    p_pre  = float((intent[t_pre,  v_pre].astype(int)  == 0).mean()) \
        if int(v_pre.sum())  else float("nan")
    p_post = float((intent[t_post, v_post].astype(int) == 0).mean()) \
        if int(v_post.sum()) else float("nan")
    both = mask & alive[t_pre] & alive[t_post]
    wrong_pre   = both & (intent[t_pre] == 1)
    correct_pre = both & (intent[t_pre] == 0)
    p_recover = float((intent[t_post, wrong_pre]   == 0).mean()) \
        if int(wrong_pre.sum())   else float("nan")
    p_abandon = float((intent[t_post, correct_pre] == 1).mean()) \
        if int(correct_pre.sum()) else float("nan")
    lift = (p_post - p_pre) if not (np.isnan(p_pre) or np.isnan(p_post)) \
        else float("nan")
    return {
        "n_episodes":              int(mask.sum()),
        "t_pre_index":             int(t_pre),
        "t_post_index":            int(t_post),
        "p_intent_pre_correct":    p_pre,
        "p_intent_post_correct":   p_post,
        "lift_post_over_pre":      lift,
        "p_wrong_to_correct":      p_recover,
        "p_correct_to_wrong":      p_abandon,
        "n_wrong_at_pre":          int(wrong_pre.sum()),
        "n_correct_at_pre":        int(correct_pre.sum()),
    }


def msg_conditional_alignment(intents: dict) -> Dict[str, dict]:
    """Alignment curve + headline stats split by the t=0 message."""
    msg_t0 = intents["msg_t0"]
    out: Dict[str, dict] = {}
    for m_id, name in MSG_NAME.items():
        mask = (msg_t0 == m_id)
        if not mask.any():
            continue
        out[name] = {
            "n_episodes":       int(mask.sum()),
            "alignment_curve":  alignment_curve(intents, mask=mask),
            "switch_matrix_pre_post": switch_matrix(intents, mask=mask,
                                                    t_from=1, t_to=2),
            "headline":         summary_headline_stats(intents, mask=mask),
        }
    return out


# --------------------------------------------------------------------------- #
# Pretty-print for CLI                                                        #
# --------------------------------------------------------------------------- #

def _fmt_curve(label, curve, t_range=range(1, 7)):
    """Compact ASCII line: label + p_correct at the given intent-array indices.

    Skips the always-AMBIG t=0 (forced-STAY step) by starting at t=1 by default.
    """
    ps = curve["p_correct"]
    ns = curve["n"]
    parts = [f"i={i}: {ps[i]:.3f} (n={ns[i]})"
             for i in t_range if i < len(ns) and ns[i] > 0]
    return f"    {label:<18s}  " + "   ".join(parts)


def _print_summary(split: str, intents: dict) -> None:
    print(f"\n=== {split} ===")
    print(f"  {intents['B']} episodes, {intents['n_layouts']} layouts")
    all_hd = summary_headline_stats(intents)
    print(f"  overall (all msgs):  P(intent[step2 pre-obs]=CORRECT)="
          f"{all_hd['p_intent_pre_correct']:.3f}"
          f"   P(intent[step3 post-obs]=CORRECT)="
          f"{all_hd['p_intent_post_correct']:.3f}"
          f"   lift={all_hd['lift_post_over_pre']:+.3f}")

    by_msg = msg_conditional_alignment(intents)
    for name in ("NONE", "M0", "M1"):
        if name not in by_msg:
            continue
        d = by_msg[name]
        hd = d["headline"]
        curve = d["alignment_curve"]
        print(f"\n  msg={name}  n={hd['n_episodes']}")
        print(f"    P(intent[step2 PRE-obs]=CORRECT)  = {hd['p_intent_pre_correct']:.3f}"
              f"  (ego's first REAL move; no partner-motion evidence)")
        print(f"    P(intent[step3 POST-obs]=CORRECT) = {hd['p_intent_post_correct']:.3f}"
              f"  (first move after seeing partner's 1st step)")
        print(f"    lift (post−pre)                   = {hd['lift_post_over_pre']:+.3f}")
        print(f"    P(WRONG@pre → CORRECT@post)        = {hd['p_wrong_to_correct']:.3f}"
              f"  (recovery from bad start; n={hd['n_wrong_at_pre']})")
        print(f"    P(CORRECT@pre → WRONG@post)        = {hd['p_correct_to_wrong']:.3f}"
              f"  (abandonment of good start; n={hd['n_correct_at_pre']})")
        print(_fmt_curve("alignment (first 6)", curve, t_range=range(1, 7)))
        print(_fmt_curve("alignment (all)", curve,
                          t_range=range(1, len(curve["p_correct"]))))
        sm = d["switch_matrix_pre_post"]
        print(f"    intent[step 2] → intent[step 3] (row-normalized), n={sm['n']}")
        print(f"      row-lbls={sm['labels']}  col-lbls={sm['labels']}")
        for lbl_i, row in zip(sm["labels"], sm["row_normalized"]):
            row_s = "  ".join(f"{v:.3f}" for v in row)
            print(f"      {lbl_i:>8s}: {row_s}")


def _to_json_payload(split_intents: Dict[str, dict]) -> dict:
    out: Dict[str, dict] = {}
    for split, intents in split_intents.items():
        out[split] = {
            "n_episodes": int(intents["B"]),
            "n_layouts":  int(intents["n_layouts"]),
            "overall_headline": summary_headline_stats(intents),
            "overall_alignment_curve": alignment_curve(intents),
            "by_msg_t0": msg_conditional_alignment(intents),
        }
    return out


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", required=True,
                   help="Path to .safetensors from ippo_rnn_coordination_grid.py")
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
    split_intents: Dict[str, dict] = {}
    for split, ldir in (("val", args.val_dir), ("test", args.test_dir)):
        key, sub = jax.random.split(key)
        print(f"\nrolling out {split} × {args.trials} trials/layout...")
        r = run_rollouts(params, cfg, ldir, sub, args.trials)
        intents = compute_intents_for_rollouts(r)
        split_intents[split] = intents
        _print_summary(split, intents)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(_to_json_payload(split_intents), f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
