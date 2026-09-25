"""Behavior × z × round analysis for the Stage C+D policy.

For every (z, round_idx) pair we measure:
  * ``P(NONE | z, r)``, ``P(M0 | z, r)``, ``P(M1 | z, r)`` — the ego's t=0
    message policy conditional on the true partner-type and the round index
    within the partner episode.
  * ``round_success(z, r)`` — the round-level coordination success rate.
  * ``Δ across z per round`` — max-min gap in ``P(msg | z)`` across z, per
    round. If genuine partner-specific convention learning is happening,
    this gap should grow past round 0 (evidence accumulates → message
    policy separates by z). If not (as suggested by the causal shuffle
    control), the gap should stay flat / near zero.

Reuses the multi-round env directly; no changes to trainer / network.

Usage:
    python dev/behavior_by_z_by_round.py \\
        --params dev/train_logs/stage_cd_R20_seed1_<TS>.safetensors \\
        --n-eps-per-z 64 \\
        --out dev/train_logs/stage_cd_behavior_by_z_r.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "baselines" / "IPPO"))
from ippo_rnn_coordination_grid import ActorCriticCommRNN, ScannedRNN

from jaxmarl.wrappers.baselines import load_params
from jaxmarl.environments.coordination_grid import CoordinationGrid


MSG_NAME = {0: "NONE", 1: "M0", 2: "M1"}


def _build_eval_env(config: dict, layouts_dir: str) -> CoordinationGrid:
    ekw = dict(config["ENV_KWARGS"])
    ekw["layouts_dir"] = layouts_dir
    ekw["augment_symmetries"] = False
    ekw.pop("layout_path", None)
    ekw.pop("layout_paths", None)
    return CoordinationGrid(**ekw)


def _rollout(params, env, config, key, n_eps_per_z):
    Z = int(env.n_partner_z)
    N = int(n_eps_per_z)
    B = Z * N
    T = int(env.rounds_per_episode * env.max_steps)

    z_index_per_ep = jnp.repeat(jnp.arange(Z, dtype=jnp.int32), N)   # (B,)
    z_per_ep = env.partner_z_values[z_index_per_ep]                  # (B,)

    key_reset, key_step = jax.random.split(key)
    reset_keys = jax.random.split(key_reset, B)
    obs, states = jax.vmap(env.reset)(reset_keys)
    states = states.replace(z=z_per_ep)                              # force per-slot z
    obs = jax.vmap(env.get_obs)(states)

    hstate0 = ScannedRNN.initialize_carry(B, config["GRU_HIDDEN_DIM"])
    done_prev0 = jnp.zeros((B,), dtype=bool)
    network = ActorCriticCommRNN(action_dim=env.n_ego_actions, config=config)

    @jax.jit
    def rollout(params, obs, states, hstate, done_prev, key):
        def body(carry, _):
            obs, states, hstate, done_prev, key = carry
            pre_time = states.time                                    # (B,)
            pre_round_idx = states.round_idx                          # (B,)
            obs_in = jax.tree_util.tree_map(lambda x: x[None, :], obs["agent_0"])
            done_in = done_prev[None, :]
            hstate, pi, _ = network.apply(params, hstate, (obs_in, done_in))
            key, ka, ks = jax.random.split(key, 3)
            action = pi.sample(seed=ka).squeeze(0)                    # (B,)
            step_keys = jax.random.split(ks, B)
            obs, states, reward, done, info = jax.vmap(
                env.step_env, in_axes=(0, 0, {"agent_0": 0})
            )(step_keys, states, {"agent_0": action})
            return (obs, states, hstate, done["__all__"], key), (
                action,
                pre_time,
                pre_round_idx,
                info["round_done"].astype(jnp.float32),
                info["success"].astype(jnp.float32),
                done["__all__"],
            )
        init_carry = (obs, states, hstate, done_prev, key)
        _, out = jax.lax.scan(body, init_carry, None, length=T)
        return out

    (actions, pre_times, pre_ris, round_dones, successes, dones) = rollout(
        params, obs, states, hstate0, done_prev0, key_step
    )
    return {
        "env": env,
        "z_index_per_ep":  np.asarray(z_index_per_ep),
        "actions":         np.asarray(actions),
        "pre_times":       np.asarray(pre_times),
        "pre_ris":         np.asarray(pre_ris),
        "round_dones":     np.asarray(round_dones),
        "successes":       np.asarray(successes),
        "dones":           np.asarray(dones),
    }


def _analyze(rollout: dict) -> dict:
    env = rollout["env"]
    R = int(env.rounds_per_episode)
    Z = int(env.n_partner_z)
    z_pool = np.asarray(env.partner_z_values)

    dones = rollout["dones"]
    first_done_idx = np.argmax(dones.astype(np.int32), axis=0)
    never_done = ~dones.any(axis=0)
    T_ax = np.arange(dones.shape[0])[:, None]
    alive = ((T_ax <= first_done_idx[None, :]) | never_done[None, :]
             ).astype(np.float32)

    actions = rollout["actions"]
    msgs = actions % 3                                               # (T, B)
    pre_times = rollout["pre_times"]                                 # (T, B)
    pre_ris = rollout["pre_ris"]                                     # (T, B)
    rd_alive = rollout["round_dones"] * alive
    s_alive = rollout["successes"] * alive
    z_idx = rollout["z_index_per_ep"]                                # (B,)

    # For each (zi, r):
    #   - Collect all t=0 events (msg policy sample).
    #   - Collect all round_done events (success outcome sample).
    msg_counts = np.zeros((Z, R, 3), dtype=np.int64)
    n_msg = np.zeros((Z, R), dtype=np.int64)
    n_succ = np.zeros((Z, R), dtype=np.int64)
    n_rounds = np.zeros((Z, R), dtype=np.int64)

    for zi in range(Z):
        col_mask = (z_idx == zi)                                     # (B,) bool
        for r in range(R):
            # T0 events for round r in this z.
            t0_hits = (
                (pre_times == 0)
                & (pre_ris == r)
                & (alive > 0.5)
                & col_mask[None, :]
            )
            m_hits = msgs[t0_hits]
            for mval in (0, 1, 2):
                msg_counts[zi, r, mval] = int((m_hits == mval).sum())
            n_msg[zi, r] = int(m_hits.size)
            # Round-done events for round r in this z.
            rd_hits = (
                (pre_ris == r)
                & (rd_alive > 0.5)
                & col_mask[None, :]
            )
            n_rounds[zi, r] = int(rd_hits.sum())
            n_succ[zi, r] = int((rollout["successes"] * rd_hits).sum())

    return {
        "z_pool": [float(v) for v in z_pool.tolist()],
        "n_partner_z": Z,
        "rounds_per_episode": R,
        "msg_counts":  msg_counts.tolist(),
        "n_msg":       n_msg.tolist(),
        "n_succ":      n_succ.tolist(),
        "n_rounds":    n_rounds.tolist(),
    }


def _fmt_dist(counts, n):
    if n == 0:
        return "  --      --      --   "
    p = counts / n
    return f"NONE={p[0]:.3f}  M0={p[1]:.3f}  M1={p[2]:.3f}"


def _print(analysis: dict, split_name: str) -> None:
    Z = analysis["n_partner_z"]
    R = analysis["rounds_per_episode"]
    z_pool = analysis["z_pool"]
    msg_counts = np.asarray(analysis["msg_counts"])
    n_msg = np.asarray(analysis["n_msg"])
    n_succ = np.asarray(analysis["n_succ"])
    n_rounds = np.asarray(analysis["n_rounds"])

    print(f"\n=== split={split_name} ===")
    key_rs = sorted(set([0, 1, 2, 5, 10, R - 1]) & set(range(R)))
    print(f"  Per (z, round): message policy + round success rate.")
    for zi, zv in enumerate(z_pool):
        print(f"    z={zv:.2f}")
        for r in key_rs:
            nm = int(n_msg[zi, r])
            nr = int(n_rounds[zi, r])
            succ = (n_succ[zi, r] / nr) if nr > 0 else float("nan")
            dist = _fmt_dist(msg_counts[zi, r], nm)
            print(f"      r={r:>2d}  {dist}   succ={succ:.3f}   "
                  f"(n_msg={nm}, n_rounds={nr})")

    print("\n  Separation across z per round (max−min P over the 4 z values):")
    print(f"    {'r':>3s}   Δ_NONE   Δ_M0    Δ_M1")
    for r in range(R):
        n = np.maximum(n_msg[:, r:r+1], 1)                           # (Z,1)
        p_zr = msg_counts[:, r, :] / n                                # (Z, 3)
        dN = float(p_zr[:, 0].max() - p_zr[:, 0].min())
        d0 = float(p_zr[:, 1].max() - p_zr[:, 1].min())
        d1 = float(p_zr[:, 2].max() - p_zr[:, 2].min())
        marker = ""
        if max(dN, d0, d1) > 0.10:
            marker = "  <-- Δ > 0.10"
        print(f"    r={r:>2d}   {dN:.3f}   {d0:.3f}   {d1:.3f}{marker}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--params", required=True)
    p.add_argument("--config",
                   default=str(_REPO / "baselines/IPPO/config"
                                        / "ippo_rnn_coordination_grid.yaml"))
    p.add_argument("--val-dir",
                   default=str(_REPO / "dev/grids/layouts/val"))
    p.add_argument("--test-dir",
                   default=str(_REPO / "dev/grids/layouts/test"))
    p.add_argument("--n-eps-per-z", type=int, default=64)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config))
    params = load_params(args.params)
    print(f"loaded params from {args.params}")

    key = jax.random.PRNGKey(args.seed)
    all_out: Dict[str, dict] = {}
    for split, ldir in (("val", args.val_dir), ("test", args.test_dir)):
        key, sub = jax.random.split(key)
        env = _build_eval_env(cfg, ldir)
        rollout = _rollout(params, env, cfg, sub, args.n_eps_per_z)
        analysis = _analyze(rollout)
        _print(analysis, split)
        all_out[split] = analysis

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
