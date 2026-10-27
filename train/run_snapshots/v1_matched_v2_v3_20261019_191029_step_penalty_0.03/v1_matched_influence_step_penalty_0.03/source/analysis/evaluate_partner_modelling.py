"""Unified evaluation for the CoordinationGrid partner-modelling replication.

Loads one trained checkpoint (from any of the four conditions), rolls out
on the same 1000-layout distribution across the training and novel test
capability populations, and dumps a per-rollout record from which every
downstream figure / probe can be reconstructed.

Per-step records saved to a per-slice HDF5:
    capability          (E, T, 2)  int32
    layout_idx          (E, T)     int32
    round_idx           (E, T)     int32
    round_done          (E, T)     bool
    success             (E, T)     bool
    reward              (E, T)     float32
    ego_alloc_action    (E, T)     int32   -- what the ego chose at t=0
    partner_assignment  (E, T)     int32   -- what actually drove the partner
    partner_goal        (E, T)     int32
    hidden_state        (E, T, H)  float32  (only when MODEL_TYPE=rnn +
                                            --save_hidden). This is the
                                            POST-OBSERVATION hstate — the
                                            state that produced this step's
                                            action, i.e. h_t = RNN(h_{t-1}, o_t).
    round_time          (E, T)     int32   -- POST-transition round-local
                                            time (from info["round_time"]).
                                            Use this for completion-time
                                            metrics; matches the analytical
                                            convention.
    is_t0               (E, T)     bool

Where E = n_capability * n_episodes_per_capability, T = R * max_steps.

Derived headline metrics per slice:
    round_success_rate
    mean_reward_per_round
    mean_successful_completion_time
    fraction_optimal_allocation_overall
    fraction_optimal_allocation_by_round
    allocation_regret_mean
    allocation_regret_by_round
    partner_assigned_red_given_relative_cap  (secondary; qualitative)

Tie convention: rounds whose two analytical rewards are equal (within a
tight tolerance) are EXCLUDED from fraction_optimal_allocation but
counted with regret=0 for the regret metric.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "baselines" / "IPPO"))

import jax
import jax.numpy as jnp

import ippo_rnn_coordination_grid as trainer_mod
from jaxmarl.environments.coordination_grid import (
    CoordinationGrid,
    TRAIN_CAPABILITY_PAIRS,
    TEST_CAPABILITY_PAIRS,
    Allocations,
    bfs_distance_map,
)
from jaxmarl.wrappers.baselines import load_params

# Reuse the exact analytical primitive used by capability_selection /
# capability_validation. One source of truth for reward math.
sys.path.insert(0, str(_REPO_ROOT / "dev"))
from capability_selection import completion_time, _reward_from_time  # noqa: E402


_REWARD_TIE_TOL = 1e-9


# --------------------------------------------------------------------------- #
# Analytical optimal-allocation primitive                                     #
# --------------------------------------------------------------------------- #

def per_layout_bfs(env: CoordinationGrid) -> Dict[str, np.ndarray]:
    """Precompute BFS distances per loaded layout, per goal, from the
    ego and partner starts. Returns four (n_layouts,) int arrays.
    """
    N = env.n_layouts
    e2r = np.zeros(N, dtype=np.int64)
    e2b = np.zeros(N, dtype=np.int64)
    p2r = np.zeros(N, dtype=np.int64)
    p2b = np.zeros(N, dtype=np.int64)
    walls_all = np.asarray(env.wall_maps)
    ego_all = np.asarray(env.ego_starts)
    par_all = np.asarray(env.partner_starts)
    red_all = np.asarray(env.red_goals)
    blue_all = np.asarray(env.blue_goals)
    for i in range(N):
        d_red = bfs_distance_map(walls_all[i], red_all[i])
        d_blue = bfs_distance_map(walls_all[i], blue_all[i])
        ex, ey = int(ego_all[i, 0]), int(ego_all[i, 1])
        px, py = int(par_all[i, 0]), int(par_all[i, 1])
        e2r[i] = int(d_red[ey, ex])
        e2b[i] = int(d_blue[ey, ex])
        p2r[i] = int(d_red[py, px])
        p2b[i] = int(d_blue[py, px])
    return {"ego_to_red": e2r, "ego_to_blue": e2b,
            "partner_to_red": p2r, "partner_to_blue": p2b}


def analytical_alloc_rewards(
    dists: Dict[str, np.ndarray],
    layout_idx: np.ndarray,   # (M,) int32
    capability: np.ndarray,   # (M, 2) int32   (d_R, d_B)
    max_steps: int,
    step_penalty: float,
    success_reward: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (reward_A, reward_B) each (M,) float — the analytical reward
    of allocation A (ego RED, partner BLUE) and B (ego BLUE, partner RED)
    for every (layout, capability) row.
    """
    M = layout_idx.shape[0]
    reward_A = np.empty(M, dtype=np.float64)
    reward_B = np.empty(M, dtype=np.float64)
    e2r = dists["ego_to_red"]
    e2b = dists["ego_to_blue"]
    p2r = dists["partner_to_red"]
    p2b = dists["partner_to_blue"]
    for i in range(M):
        li = int(layout_idx[i])
        d_R, d_B = int(capability[i, 0]), int(capability[i, 1])
        tA = completion_time(int(e2r[li]), int(p2b[li]), d_B, max_steps)
        tB = completion_time(int(e2b[li]), int(p2r[li]), d_R, max_steps)
        reward_A[i] = _reward_from_time(int(tA), max_steps, step_penalty, success_reward)
        reward_B[i] = _reward_from_time(int(tB), max_steps, step_penalty, success_reward)
    return reward_A, reward_B


def optimal_allocation_and_regret(
    reward_A: np.ndarray,
    reward_B: np.ndarray,
    chosen_alloc: np.ndarray,   # (M,) int32 — Allocations.red or Allocations.blue
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (is_optimal, regret, is_tie).

    - is_optimal[i] : True iff chosen_alloc[i] is a reward-maximiser.
    - regret[i]     : max(reward_A, reward_B) - reward_chosen[i]. Always >= 0.
    - is_tie[i]     : |reward_A - reward_B| <= _REWARD_TIE_TOL.

    Tie convention: is_optimal is True on ties (both allocs are optimal),
    regret is 0. Downstream metrics exclude ties from optimal-fraction
    accuracy while still counting them in regret (which is 0 there).
    """
    reward_chosen = np.where(chosen_alloc == int(Allocations.red), reward_A, reward_B)
    reward_opt = np.maximum(reward_A, reward_B)
    regret = reward_opt - reward_chosen
    is_tie = np.abs(reward_A - reward_B) <= _REWARD_TIE_TOL
    is_optimal = np.where(is_tie, True, regret <= _REWARD_TIE_TOL)
    return is_optimal, regret, is_tie


# --------------------------------------------------------------------------- #
# Rollout                                                                     #
# --------------------------------------------------------------------------- #

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
    # AUTHORITATIVE experimental switches restored from the SAVED config,
    # not inferred from ENV_KWARGS alone. This prevents accidentally
    # evaluating a no-influence checkpoint under influence=True (or vice
    # versa) if the ENV_KWARGS block happens to be stale.
    env_kwargs["influence"] = bool(config.get("INFLUENCE", True))
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
    # Deterministically pin the per-slot capability.
    states = states.replace(capability=cap_per_ep)
    obs = jax.vmap(env.get_obs)(states)

    hstate = init_hstate_fn(B)
    done_prev = jnp.zeros((B,), dtype=bool)

    def body(carry, _):
        obs, states, hstate, done_prev, key = carry
        obs_a = obs["agent_0"]
        obs_in = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, :], obs_a)
        done_in = done_prev[jnp.newaxis, :]
        # Convention: `new_hstate` is h_t (post-observation for step t); it
        # is the recurrent state used to produce this step's action. That is
        # what we save as "hidden_state" so downstream probes decode the
        # representation the policy was actually acting from.
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
            info["layout_idx"].astype(jnp.int32),
            info["capability"].astype(jnp.int32),
            info["partner_assignment"].astype(jnp.int32),
            info["ego_alloc_action"].astype(jnp.int32),
            info["partner_goal"].astype(jnp.int32),
            info["round_time"].astype(jnp.int32),
            (obs_a["is_t0"] > 0.5),
            (new_hstate if save_hidden else jnp.zeros((B, 1), dtype=jnp.float32)),
        )

    (_, _, _, _, _), traj = jax.lax.scan(
        body, (obs, states, hstate, done_prev, key_step), None, length=T,
    )

    (rewards, dones, succs, round_dones, r_idx, layout_idx, caps,
     partner_assignment, ego_alloc, partner_goal, round_times, is_t0, hidden) = [
        np.asarray(x) for x in traj
    ]

    # All fields are (T, B) — transpose to (B, T) for storage.
    return {
        "rewards":            rewards.T,
        "dones":              dones.T,
        "success":            succs.T,
        "round_done":         round_dones.T,
        "round_idx":          r_idx.T,
        "layout_idx":         layout_idx.T,
        "capability":         caps.transpose(1, 0, 2),
        "partner_assignment": partner_assignment.T,
        "ego_alloc_action":   ego_alloc.T,
        "partner_goal":       partner_goal.T,
        "round_time":         round_times.T,
        "is_t0":              is_t0.T,
        "hidden_state":       hidden.transpose(1, 0, 2),
        "capability_index_per_ep": np.asarray(cap_index_per_ep),
        "capability_pool":         np.asarray(env.partner_capability_pairs),
    }, env


def _first_done_alive_mask(dones: np.ndarray) -> np.ndarray:
    """(B, T) dones -> (B, T) alive mask up to and including first done."""
    B, T = dones.shape
    first = np.argmax(dones.astype(np.int32), axis=1)
    never = ~dones.any(axis=1)
    T_ax = np.arange(T)[None, :]
    return ((T_ax <= first[:, None]) | never[:, None]).astype(np.bool_)


# --------------------------------------------------------------------------- #
# Summarization                                                               #
# --------------------------------------------------------------------------- #

def summarize(
    record: Dict[str, np.ndarray],
    env: CoordinationGrid,
    max_steps: int,
    step_penalty: float,
    success_reward: float,
) -> Dict:
    alive = _first_done_alive_mask(record["dones"])
    rd = record["round_done"] & alive
    succ = record["success"] & rd
    r_idx = record["round_idx"]
    reward_env = record["rewards"] * alive

    n_rounds = int(rd.sum())
    round_success_rate = float(succ.sum() / max(n_rounds, 1))
    ep_return = reward_env.sum(axis=1)
    completion_times = record["round_time"][succ]
    mean_completion = float(completion_times.mean()) if completion_times.size else float("nan")

    R = int(r_idx.max()) + 1 if r_idx.size else 0
    per_round_success = np.zeros(R, dtype=np.float32)
    per_round_n = np.zeros(R, dtype=np.int64)
    for r in range(R):
        m = (r_idx == r) & rd
        n = int(m.sum())
        per_round_n[r] = n
        if n:
            per_round_success[r] = float(succ[m].sum() / n)

    # ---- Optimal-allocation analysis (per t=0 sample) ----
    is_t0 = record["is_t0"] & alive
    # Sampled per round at t=0. layout, capability, and ego alloc are all
    # frozen for that (round, episode) at these steps.
    layout_flat = record["layout_idx"][is_t0]
    cap_flat = record["capability"][is_t0]
    ego_alloc_flat = record["ego_alloc_action"][is_t0]
    round_idx_flat = record["round_idx"][is_t0]

    dists = per_layout_bfs(env)
    rew_A, rew_B = analytical_alloc_rewards(
        dists, layout_flat.astype(np.int32),
        cap_flat.astype(np.int32),
        max_steps=max_steps, step_penalty=step_penalty,
        success_reward=success_reward,
    )
    is_opt, regret, is_tie = optimal_allocation_and_regret(
        rew_A, rew_B, ego_alloc_flat.astype(np.int32),
    )

    # Overall optimal-allocation fraction (excluding ties).
    non_tie_mask = ~is_tie
    if non_tie_mask.any():
        frac_optimal_overall = float(is_opt[non_tie_mask].mean())
    else:
        frac_optimal_overall = float("nan")
    regret_mean = float(regret.mean()) if regret.size else float("nan")

    # By round-index breakdown.
    frac_optimal_by_round = np.full(R, np.nan, dtype=np.float64)
    n_nontie_by_round = np.zeros(R, dtype=np.int64)
    regret_by_round = np.full(R, np.nan, dtype=np.float64)
    n_by_round = np.zeros(R, dtype=np.int64)
    for r in range(R):
        m = (round_idx_flat == r)
        if m.any():
            regret_by_round[r] = float(regret[m].mean())
            n_by_round[r] = int(m.sum())
            m_nt = m & non_tie_mask
            if m_nt.any():
                frac_optimal_by_round[r] = float(is_opt[m_nt].mean())
                n_nontie_by_round[r] = int(m_nt.sum())

    # Secondary: relative-capability -> partner-assigned-RED.
    d_r = cap_flat[:, 0]
    d_b = cap_flat[:, 1]
    rel_adv = d_b - d_r
    partner_asg_flat = record["partner_assignment"][is_t0]
    # pending==BLUE => partner takes RED
    partner_assigned_red = (partner_asg_flat == int(Allocations.blue))
    p_red_pos = float(partner_assigned_red[rel_adv > 0].mean()) if (rel_adv > 0).any() else float("nan")
    p_red_neg = float(partner_assigned_red[rel_adv < 0].mean()) if (rel_adv < 0).any() else float("nan")
    p_red_tie = float(partner_assigned_red[rel_adv == 0].mean()) if (rel_adv == 0).any() else float("nan")

    return {
        "n_episodes": int(record["dones"].shape[0]),
        "n_rounds": n_rounds,
        "round_success_rate": round_success_rate,
        "mean_ep_return": float(ep_return.mean()),
        "mean_reward_per_round": float(reward_env.sum() / max(n_rounds, 1)),
        "mean_successful_completion_time": mean_completion,
        "per_round_success_rate": per_round_success.tolist(),
        "per_round_n": per_round_n.tolist(),
        "fraction_optimal_allocation_overall": frac_optimal_overall,
        "fraction_optimal_allocation_by_round": frac_optimal_by_round.tolist(),
        "n_nontie_by_round": n_nontie_by_round.tolist(),
        "allocation_regret_mean": regret_mean,
        "allocation_regret_by_round": regret_by_round.tolist(),
        "n_allocations_by_round": n_by_round.tolist(),
        "fraction_ties_overall": float(is_tie.mean()) if is_tie.size else float("nan"),
        "partner_assigned_red_given_red_adv":   p_red_pos,
        "partner_assigned_red_given_blue_adv":  p_red_neg,
        "partner_assigned_red_given_tie":       p_red_tie,
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
    # Force the standalone eval to honor the checkpoint's true training
    # switches, regardless of whatever's in ENV_KWARGS.
    config.setdefault("MODEL_TYPE", "rnn")
    config.setdefault("PARTNER_REGIME", "diverse")
    config.setdefault("INFLUENCE", True)
    config.setdefault("SINGLE_PARTNER", [1, 4])

    params = load_params(args.params)

    # Analytical primitive parameters — read the training env's reward
    # settings so blend / regret math matches what the policy was trained on.
    env_kwargs = config["ENV_KWARGS"]
    max_steps = int(env_kwargs.get("max_steps", 100))
    step_penalty = float(env_kwargs.get("step_penalty", 0.01))
    success_reward = float(env_kwargs.get("success_reward", 1.0))

    summaries: Dict[str, Dict] = {}
    for slice_name, pool in (
        ("train", TRAIN_CAPABILITY_PAIRS),
        ("test",  TEST_CAPABILITY_PAIRS),
    ):
        print(f"[eval] slice={slice_name}, |pool|={len(pool)}, "
              f"n_eps/cap={args.n_episodes_per_capability}", flush=True)
        record, env = rollout_condition(
            params, config, pool,
            n_episodes_per_capability=args.n_episodes_per_capability,
            layouts_dir=args.layouts_dir,
            seed=args.seed,
            save_hidden=args.save_hidden and config.get("MODEL_TYPE", "rnn") == "rnn",
        )
        _save_hdf5(f"{args.out_prefix}_{slice_name}.h5", record)
        summaries[slice_name] = summarize(
            record, env,
            max_steps=max_steps, step_penalty=step_penalty,
            success_reward=success_reward,
        )
        s = summaries[slice_name]
        print(f"    round_success={s['round_success_rate']:.3f} "
              f"reward/round={s['mean_reward_per_round']:+.4f} "
              f"opt_alloc={s['fraction_optimal_allocation_overall']:.3f} "
              f"regret={s['allocation_regret_mean']:+.4f}",
              flush=True)

    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"[eval] wrote {args.out_prefix}_summary.json", flush=True)


if __name__ == "__main__":
    main()
