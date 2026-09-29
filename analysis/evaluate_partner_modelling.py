"""Unified evaluation for the CoordinationGrid partner-modelling replication.

Loads one trained checkpoint (from any of the four conditions), rolls out
on the same 1000-layout distribution across the training and novel test
capability populations, and dumps a per-rollout record from which every
downstream figure / probe can be reconstructed.

Per-step records saved to a per-checkpoint HDF5:
    capability          (E, 2)     int32
    layout_idx          (E, T)     int32
    round_idx           (E, T)     int32
    round_done          (E, T)     bool
    success             (E, T)     bool
    reward              (E, T)     float32
    ego_alloc_action    (E, T)     int32   -- what the ego chose at t=0
    pending_allocation  (E, T)     int32   -- what actually drove the partner
    partner_goal        (E, T)     int32
    hidden_state        (E, T, H)  float32  (only when MODEL_TYPE=rnn)
    time                (E, T)     int32
    is_t0               (E, T)     bool

Where E = n_episodes and T = rounds_per_episode * max_steps.

Also derives the following headline metrics per capability slice:
    round_success_rate
    mean_reward_per_round
    mean_successful_completion_time
    fraction_optimal_allocation (per round_idx)
    partner_assigned_red_given_relative_cap (per relative_red_advantage bin)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "baselines" / "IPPO"))

import jax
import jax.numpy as jnp

import ippo_rnn_coordination_grid as trainer_mod  # unified trainer
from jaxmarl.environments.coordination_grid import (
    CoordinationGrid,
    TRAIN_CAPABILITY_PAIRS,
    TEST_CAPABILITY_PAIRS,
    Allocations,
    GOAL_RED,
    GOAL_BLUE,
)
from jaxmarl.wrappers.baselines import load_params


def rollout_condition(
    params,
    config: Dict,
    capability_pool: Sequence[Sequence[int]],
    n_episodes_per_capability: int,
    layouts_dir: str,
    seed: int,
    save_hidden: bool,
) -> Dict[str, np.ndarray]:
    env_kwargs = dict(config["ENV_KWARGS"])
    env_kwargs["layouts_dir"] = layouts_dir
    env_kwargs["augment_symmetries"] = False
    env_kwargs.pop("layout_path", None)
    env_kwargs.pop("layout_paths", None)
    env_kwargs["partner_capability_pairs"] = [
        [int(a), int(b)] for a, b in capability_pool
    ]
    env = CoordinationGrid(**env_kwargs)

    network, init_hstate_fn, model_type = trainer_mod.build_network(
        config, env.n_ego_actions
    )
    C = int(env.n_partner_capability)
    N = int(n_episodes_per_capability)
    B = C * N
    R = int(env.rounds_per_episode)
    T = R * env.max_steps

    cap_index_per_ep = jnp.repeat(jnp.arange(C, dtype=jnp.int32), N)
    cap_per_ep = env.partner_capability_pairs[cap_index_per_ep]

    key = jax.random.PRNGKey(seed)
    key_reset, key_step = jax.random.split(key)
    reset_keys = jax.random.split(key_reset, B)
    obs, states = jax.vmap(env.reset)(reset_keys)
    states = states.replace(capability=cap_per_ep)
    obs = jax.vmap(env.get_obs)(states)

    hstate = init_hstate_fn(B)
    done_prev = jnp.zeros((B,), dtype=bool)

    def body(carry, _):
        obs, states, hstate, done_prev, key = carry
        obs_a = obs["agent_0"]
        obs_in = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, :], obs_a)
        done_in = done_prev[jnp.newaxis, :]
        new_hstate, pi, _v = network.apply(params, hstate, (obs_in, done_in))
        key, ka, ks = jax.random.split(key, 3)
        action = pi.sample(seed=ka).squeeze(0)
        step_keys = jax.random.split(ks, B)
        obs2, states2, reward, done, info = jax.vmap(
            env.step_env, in_axes=(0, 0, {"agent_0": 0})
        )(step_keys, states, {"agent_0": action})
        return (obs2, states2, new_hstate, done["__all__"], key), (
            reward["agent_0"],
            done["__all__"],
            info["success"].astype(jnp.bool_),
            info["round_done"].astype(jnp.bool_),
            info["round_idx"].astype(jnp.int32),
            info["capability"].astype(jnp.int32),
            info["pending_allocation"].astype(jnp.int32),
            info["ego_alloc_action"].astype(jnp.int32),
            info["partner_goal"].astype(jnp.int32),
            states.layout_idx.astype(jnp.int32),
            states.time.astype(jnp.int32),
            (obs_a["is_t0"] > 0.5),
            (hstate if save_hidden else jnp.zeros((B, 1), dtype=jnp.float32)),
        )

    (_, _, _, _, _), traj = jax.lax.scan(
        body, (obs, states, hstate, done_prev, key_step), None, length=T,
    )

    (rewards, dones, succs, round_dones, r_idx, caps, pending, ego_alloc,
     partner_goal, layout_idx, times, is_t0, hidden) = [np.asarray(x) for x in traj]

    return {
        "rewards": rewards.T,             # (B, T)
        "dones": dones.T,
        "success": succs.T,
        "round_done": round_dones.T,
        "round_idx": r_idx.T,
        "capability": caps.transpose(1, 0, 2),   # (B, T, 2)
        "pending_allocation": pending.T,
        "ego_alloc_action": ego_alloc.T,
        "partner_goal": partner_goal.T,
        "layout_idx": layout_idx.T,
        "time": times.T,
        "is_t0": is_t0.T,
        "hidden_state": hidden.transpose(1, 0, 2),  # (B, T, H)
        "capability_index_per_ep": np.asarray(cap_index_per_ep),
        "capability_pool": np.asarray(env.partner_capability_pairs),
    }


def _first_done_alive_mask(dones: np.ndarray) -> np.ndarray:
    """(T, B) or (B, T) → (B, T) alive mask up to and including first done."""
    if dones.ndim != 2:
        raise ValueError(f"expected 2D dones, got {dones.shape}")
    B, T = dones.shape
    first = np.argmax(dones.astype(np.int32), axis=1)
    never = ~dones.any(axis=1)
    T_ax = np.arange(T)[None, :]
    return ((T_ax <= first[:, None]) | never[:, None]).astype(np.bool_)


def summarize(record: Dict[str, np.ndarray]) -> Dict:
    """Produce per-slice headline metrics."""
    alive = _first_done_alive_mask(record["dones"])
    rd = record["round_done"] & alive
    succ = record["success"] & rd
    r_idx = record["round_idx"]
    reward = record["rewards"] * alive

    n_rounds = int(rd.sum())
    round_success_rate = float(succ.sum() / max(n_rounds, 1))
    ep_return = reward.sum(axis=1)

    # Completion time per successful round: time at which round_done was set.
    completion_times = record["time"][succ]
    mean_completion = float(completion_times.mean()) if completion_times.size else float("nan")

    # Per round-index success.
    R = int(r_idx.max()) + 1 if r_idx.size else 0
    per_round_success = np.zeros(R, dtype=np.float32)
    per_round_n = np.zeros(R, dtype=np.int64)
    for r in range(R):
        m = (r_idx == r) & rd
        n = int(m.sum())
        per_round_n[r] = n
        if n:
            per_round_success[r] = float(succ[m].sum() / n)

    # Allocation-vs-relative-capability. Sample at t==0 (one allocation
    # per round). relative_red_advantage = d_B - d_R (positive = better at RED).
    is_t0 = record["is_t0"]
    caps = record["capability"]
    d_r = caps[..., 0]
    d_b = caps[..., 1]
    rel_adv = d_b - d_r
    # Was the partner assigned RED? partner_goal is written at t=1; sample
    # the round-first allocation via pending_allocation at t=0.
    pending_at_t0 = record["pending_allocation"][is_t0]  # ego→partner map:
    # pending == RED  ⇒ partner takes BLUE ; pending == BLUE ⇒ partner takes RED.
    partner_assigned_red = (pending_at_t0 == int(Allocations.blue))
    rel_at_t0 = rel_adv[is_t0]
    # Bin by sign.
    p_red_if_red_adv = float(partner_assigned_red[rel_at_t0 > 0].mean()) \
        if (rel_at_t0 > 0).any() else float("nan")
    p_red_if_blue_adv = float(partner_assigned_red[rel_at_t0 < 0].mean()) \
        if (rel_at_t0 < 0).any() else float("nan")
    p_red_if_tie = float(partner_assigned_red[rel_at_t0 == 0].mean()) \
        if (rel_at_t0 == 0).any() else float("nan")

    return {
        "n_episodes": int(record["dones"].shape[0]),
        "n_rounds": n_rounds,
        "round_success_rate": round_success_rate,
        "mean_ep_return": float(ep_return.mean()),
        "mean_reward_per_round": float(reward.sum() / max(n_rounds, 1)),
        "mean_successful_completion_time": mean_completion,
        "per_round_success_rate": per_round_success.tolist(),
        "per_round_n": per_round_n.tolist(),
        "partner_assigned_red_given_red_adv":   p_red_if_red_adv,
        "partner_assigned_red_given_blue_adv":  p_red_if_blue_adv,
        "partner_assigned_red_given_tie":       p_red_if_tie,
    }


def _save_hdf5(path: str, record: Dict[str, np.ndarray]) -> None:
    import h5py
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in record.items():
            f.create_dataset(k, data=np.asarray(v))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True,
                   help="JSON dump of the training config (from training run).")
    p.add_argument("--params", required=True,
                   help="safetensors checkpoint (single-seed).")
    p.add_argument("--layouts_dir",
                   default=str(_REPO_ROOT / "dev" / "grids_capability_selected" / "layouts" / "train"))
    p.add_argument("--n_episodes_per_capability", type=int, default=20)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out_prefix", required=True,
                   help="Output path prefix — writes {prefix}_{slice}.h5 and _summary.json")
    p.add_argument("--save_hidden", action="store_true",
                   help="Save GRU hidden states (only meaningful for MODEL_TYPE=rnn).")
    args = p.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    params = load_params(args.params)

    summaries: Dict[str, Dict] = {}
    for slice_name, pool in (
        ("train", TRAIN_CAPABILITY_PAIRS),
        ("test",  TEST_CAPABILITY_PAIRS),
    ):
        print(f"[eval] slice={slice_name}, |pool|={len(pool)}, "
              f"n_eps/cap={args.n_episodes_per_capability}", flush=True)
        rec = rollout_condition(
            params, config, pool,
            n_episodes_per_capability=args.n_episodes_per_capability,
            layouts_dir=args.layouts_dir,
            seed=args.seed,
            save_hidden=args.save_hidden and config.get("MODEL_TYPE", "rnn") == "rnn",
        )
        _save_hdf5(f"{args.out_prefix}_{slice_name}.h5", rec)
        summaries[slice_name] = summarize(rec)
        s = summaries[slice_name]
        print(f"    round_success={s['round_success_rate']:.3f} "
              f"reward_per_round={s['mean_reward_per_round']:+.4f} "
              f"P(partner=RED | red_adv)={s['partner_assigned_red_given_red_adv']:.3f} "
              f"P(partner=RED | blue_adv)={s['partner_assigned_red_given_blue_adv']:.3f}",
              flush=True)

    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"[eval] wrote {args.out_prefix}_summary.json", flush=True)


if __name__ == "__main__":
    main()
