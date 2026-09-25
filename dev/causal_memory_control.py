"""Causal memory control for the Stage C+D policy.

Keep the trained model fixed; evaluate three variants of the *same* rollout,
differing only in what happens to the GRU hidden state at every round
boundary within a partner episode:

  normal        no intervention -- hidden carries information forward.
  round_reset   zero the GRU hidden at every round boundary.
  shuffle       permute hidden states across parallel episodes at every
                round boundary; because z is uniformly distributed across
                the parallel slots, most permutations swap between episodes
                with different z, so each ego receives a partner-history
                that doesn't match its own partner.

Predictions:
  * normal:      per-round-index success curve r0 -> r1 jumps sharply.
  * round_reset: r_k stays close to r_0 for all k (no accumulated info).
  * shuffle:     r_1 onward is systematically wrong (info from someone
                 else's partner).

Also fits a **linear probe** from the GRU hidden state after each round to
the true z. Decodability should rise sharply after round 0 and then plateau,
matching the behavioral curve.

Usage:
    python dev/causal_memory_control.py \\
        --params dev/train_logs/stage_cd_R20_seed1_<TS>.safetensors \\
        --n-eps-per-z 32 \\
        --out dev/train_logs/stage_cd_causal_memory.json
"""

from __future__ import annotations

import argparse
import functools
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


VARIANTS = ("normal", "round_reset", "shuffle", "cross_z_shuffle")


def _build_eval_env(config: dict, layouts_dir: str) -> CoordinationGrid:
    env_kwargs = dict(config["ENV_KWARGS"])
    env_kwargs["layouts_dir"] = layouts_dir
    env_kwargs["augment_symmetries"] = False
    env_kwargs.pop("layout_path", None)
    env_kwargs.pop("layout_paths", None)
    return CoordinationGrid(**env_kwargs)


def _make_rollout(env, network, mode: str, config: dict, B: int, T: int,
                   Z: int, N: int):
    """Return a jitted function that rolls one full partner episode under
    the specified intervention mode. Compiled once per mode.

    ``Z`` = number of partner-z values, ``N`` = eps per z. Assumes the
    parallel-slot layout is ``z_index_per_ep = repeat(arange(Z), N)`` so
    within-z groups are contiguous — used by ``cross_z_shuffle`` to
    guarantee cross-z swaps.
    """

    @jax.jit
    def rollout(params, obs, states, hstate, done_prev, key):
        def body(carry, _):
            obs, states, hstate, done_prev, key = carry
            obs_in = jax.tree_util.tree_map(
                lambda x: x[None, :], obs["agent_0"]
            )
            done_in = done_prev[None, :]
            hstate, pi, _ = network.apply(params, hstate, (obs_in, done_in))
            key, ka, ks, kp, kzs = jax.random.split(key, 5)
            action = pi.sample(seed=ka).squeeze(0)                # (B,)
            step_keys = jax.random.split(ks, B)
            obs, states, reward, done, info = jax.vmap(
                env.step_env, in_axes=(0, 0, {"agent_0": 0})
            )(step_keys, states, {"agent_0": action})

            round_done = info["round_done"]                       # (B,) bool
            done_all = done["__all__"]                            # (B,) bool

            # Intervene on hstate AT round boundaries (i.e., steps where this
            # step ended a round). We do this AFTER the current step so the
            # ego already used its "true" hidden for the terminal action of
            # the round; the intervention affects the NEXT round's forward pass.
            rd_mask = round_done[:, None]                         # (B, 1)
            if mode == "round_reset":
                hstate = jnp.where(rd_mask, jnp.zeros_like(hstate), hstate)
            elif mode == "shuffle":
                # Random permutation over all B slots. Some in-z swaps sneak
                # through because within-z groups are 1/Z of the slots.
                perm = jax.random.permutation(kp, B)
                shuffled = hstate[perm]
                hstate = jnp.where(rd_mask, shuffled, hstate)
            elif mode == "cross_z_shuffle":
                # STRICT cross-z: for each slot b, pick a target z' != z[b],
                # then a random slot within that z-group. Guaranteed the
                # incoming hidden was accumulated with a different z.
                # slots [zi*N, (zi+1)*N) belong to z index zi.
                slot_ids = jnp.arange(B)
                own_z = slot_ids // N                              # (B,)
                key_off, key_slot = jax.random.split(kzs)
                # Sample offset in [1, Z), forcing z' = (own_z + off) mod Z.
                offsets = jax.random.randint(
                    key_off, (B,), minval=1, maxval=Z
                )
                target_z = (own_z + offsets) % Z
                # Random slot within the target z-group.
                in_group = jax.random.randint(
                    key_slot, (B,), minval=0, maxval=N
                )
                target_slot = target_z * N + in_group              # (B,)
                shuffled = hstate[target_slot]
                hstate = jnp.where(rd_mask, shuffled, hstate)
            # mode == "normal" — leave hstate alone.

            return (obs, states, hstate, done_all, key), (
                reward["agent_0"],
                done_all,
                info["success"].astype(jnp.float32),
                info["round_done"].astype(jnp.float32),
                info["round_idx"].astype(jnp.int32),
                hstate,   # post-intervention hidden, recorded per step
            )

        init_carry = (obs, states, hstate, done_prev, key)
        _, out = jax.lax.scan(body, init_carry, None, length=T)
        return out

    return rollout


def _run_one_variant(params, config, env, key, n_eps_per_z, mode):
    """Run one full-episode eval under the specified intervention mode."""
    Z = int(env.n_partner_z)
    N = int(n_eps_per_z)
    B = Z * N
    R = int(env.rounds_per_episode)
    T = R * env.max_steps

    z_index_per_ep = jnp.repeat(jnp.arange(Z, dtype=jnp.int32), N)
    z_per_ep = env.partner_z_values[z_index_per_ep]

    key_reset, key_step = jax.random.split(key)
    reset_keys = jax.random.split(key_reset, B)
    obs, states = jax.vmap(env.reset)(reset_keys)
    states = states.replace(z=z_per_ep)   # force per-slot z
    obs = jax.vmap(env.get_obs)(states)

    hstate0 = ScannedRNN.initialize_carry(B, config["GRU_HIDDEN_DIM"])
    done_prev0 = jnp.zeros((B,), dtype=bool)

    network = ActorCriticCommRNN(action_dim=env.n_ego_actions, config=config)
    rollout = _make_rollout(env, network, mode, config, B, T, Z, N)
    (rewards, dones, successes, round_dones, round_idxs, hstates) = rollout(
        params, obs, states, hstate0, done_prev0, key_step
    )
    # rewards, dones, successes, round_dones, round_idxs: (T, B)
    # hstates: (T, B, H)

    dones_np = np.asarray(dones)
    first_done_idx = np.argmax(dones_np.astype(np.int32), axis=0)
    never_done = ~dones_np.any(axis=0)
    T_ax = np.arange(dones_np.shape[0])[:, None]
    alive_mask_np = (
        (T_ax <= first_done_idx[None, :]) | never_done[None, :]
    ).astype(np.float32)

    rd_alive = np.asarray(round_dones) * alive_mask_np             # (T, B)
    s_np = np.asarray(successes)                                   # (T, B)
    r_idx_np = np.asarray(round_idxs)                              # (T, B)
    z_idx_np = np.asarray(z_index_per_ep)                          # (B,)

    # Overall + per-z + per-round-index round-success.
    n_rounds = float(rd_alive.sum())
    overall = float((s_np * rd_alive).sum() / max(n_rounds, 1.0))
    per_z = np.zeros(Z, dtype=np.float32)
    per_z_n = np.zeros(Z, dtype=np.int64)
    for zi in range(Z):
        cols = (z_idx_np == zi)
        rd_z = rd_alive[:, cols]
        s_z = s_np[:, cols]
        nrd = int(rd_z.sum())
        per_z[zi] = float((s_z * rd_z).sum() / max(nrd, 1))
        per_z_n[zi] = nrd

    per_round = np.zeros(R, dtype=np.float32)
    per_round_n = np.zeros(R, dtype=np.int64)
    for r in range(R):
        mask = (r_idx_np == r) & (rd_alive > 0.5)
        n = int(mask.sum())
        if n > 0:
            per_round[r] = float(s_np[mask].sum() / n)
        per_round_n[r] = n

    # ---- Collect (round_idx, z, hstate) triples at every round boundary ----
    # We use the round_idx of the step that JUST ended (that's `info["round_idx"]`
    # for that step). The hstate is the post-intervention one, so under
    # 'shuffle' it's the SHUFFLED hidden the next round would see; under
    # 'normal' / 'round_reset' it's the actual hidden fed to the next round.
    hstates_np = np.asarray(hstates)                                # (T, B, H)
    round_hstates_by_r: Dict[int, np.ndarray] = {}
    round_zs_by_r: Dict[int, np.ndarray] = {}
    for r in range(R):
        mask = (r_idx_np == r) & (rd_alive > 0.5)                  # (T, B)
        # Pull each (t, b) hit into a flat list.
        t_hits, b_hits = np.where(mask)
        if len(t_hits) == 0:
            continue
        h_hits = hstates_np[t_hits, b_hits, :]                     # (n_hits, H)
        z_hits = np.asarray(env.partner_z_values)[z_idx_np[b_hits]]
        round_hstates_by_r[r] = h_hits.astype(np.float32)
        round_zs_by_r[r] = z_hits.astype(np.float32)

    return {
        "mode": mode,
        "n_episodes": int(B),
        "n_partner_z": Z,
        "n_eps_per_z": int(N),
        "rounds_per_episode": R,
        "partner_z_values": [float(v) for v in np.asarray(env.partner_z_values)],
        "round_success_overall": overall,
        "per_z_success": per_z.tolist(),
        "per_z_rounds": per_z_n.tolist(),
        "per_round_idx_success": per_round.tolist(),
        "per_round_idx_n": per_round_n.tolist(),
        "hstates_by_round": round_hstates_by_r,
        "zs_by_round": round_zs_by_r,
    }


# --------------------------------------------------------------------------- #
# Linear probe of z from the recorded GRU hidden state                        #
# --------------------------------------------------------------------------- #

def _linear_probe_z(hstates: np.ndarray, z_true: np.ndarray,
                    z_pool: np.ndarray, train_frac: float = 0.7,
                    ridge_alpha: float = 1.0,
                    rng: np.random.Generator | None = None):
    """Fit a *ridge-regularized* linear regression on hstate -> z_true.

    Hidden dim can be > sample size on smaller runs, so ordinary lstsq
    overfits catastrophically. Ridge (small L2) is the appropriate probe.
    Reports test R², MSE, and snap-to-pool classification accuracy.
    Chance-level accuracy = 1 / len(z_pool).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    N = hstates.shape[0]
    if N < 8:
        return {"n": int(N), "r2_test": float("nan"),
                "mse_test": float("nan"),
                "acc_test": float("nan")}
    idx = rng.permutation(N)
    n_train = max(1, int(N * train_frac))
    tr, te = idx[:n_train], idx[n_train:]
    X_tr, X_te = hstates[tr], hstates[te]
    y_tr, y_te = z_true[tr], z_true[te]

    # Ridge regression with a bias column. Standardize features so alpha is
    # comparable across models with different hidden-dim norms.
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-6
    Xtr_s = (X_tr - mu) / sd
    Xte_s = (X_te - mu) / sd
    Xa = np.concatenate([Xtr_s, np.ones((Xtr_s.shape[0], 1), dtype=Xtr_s.dtype)], axis=1)
    Xb = np.concatenate([Xte_s, np.ones((Xte_s.shape[0], 1), dtype=Xte_s.dtype)], axis=1)
    D = Xa.shape[1]
    reg = ridge_alpha * np.eye(D, dtype=Xa.dtype)
    reg[-1, -1] = 0.0                   # don't regularize the bias term
    w = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ y_tr)
    y_te_pred = Xb @ w
    ss_tot_te = float(((y_te - y_te.mean()) ** 2).sum())
    ss_res_te = float(((y_te - y_te_pred) ** 2).sum())
    r2_te = 1.0 - ss_res_te / max(ss_tot_te, 1e-12)
    mse_te = float(((y_te - y_te_pred) ** 2).mean())

    # Classification accuracy: snap the continuous prediction to the nearest
    # pool value, then compare.
    def _snap(vals):
        dists = np.abs(vals[:, None] - z_pool[None, :])
        return z_pool[np.argmin(dists, axis=1)]
    acc_te = float(np.mean(np.isclose(_snap(y_te_pred), y_te, atol=1e-6)))
    chance = float(1.0 / len(z_pool))

    return {
        "n": int(N),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "r2_test": float(r2_te),
        "mse_test": float(mse_te),
        "acc_test": float(acc_te),
        "chance_acc": chance,
    }


def _probe_all_rounds(round_hstates, round_zs, z_pool, rng):
    out = {}
    for r in sorted(round_hstates.keys()):
        out[str(r)] = _linear_probe_z(
            round_hstates[r], round_zs[r], z_pool, rng=rng
        )
    return out


# --------------------------------------------------------------------------- #
# Pretty print                                                                #
# --------------------------------------------------------------------------- #

def _print_curve(name: str, per_round, per_round_n, max_r=None):
    R = len(per_round) if max_r is None else min(len(per_round), max_r)
    print(f"  {name:>14s}:")
    for r in range(R):
        s = per_round[r]
        n = per_round_n[r]
        bar = "#" * int(round(s * 40))
        print(f"    r{r:>2d}  succ={s:.3f}  n={n:>4d}  |{bar}")


def _print_probe(name: str, probes: dict):
    print(f"  {name:>18s}: linear probe (z from hidden), test R² / snap-acc")
    for r_str, p in probes.items():
        r = int(r_str)
        r2 = p["r2_test"]
        acc = p["acc_test"]
        n = p["n"]
        bar = "#" * int(round(max(0.0, min(1.0, r2)) * 40))
        print(f"    r{r:>2d}  r2={r2:+.3f}  acc={acc:.3f}  n={n:>4d}  |{bar}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--params", required=True)
    p.add_argument("--config",
                   default=str(_REPO / "baselines/IPPO/config"
                                        / "ippo_rnn_coordination_grid.yaml"))
    p.add_argument("--val-dir",
                   default=str(_REPO / "dev/grids/layouts/val"))
    p.add_argument("--test-dir",
                   default=str(_REPO / "dev/grids/layouts/test"))
    p.add_argument("--split", choices=("val", "test", "both"), default="both")
    p.add_argument("--n-eps-per-z", type=int, default=32)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config))
    params = load_params(args.params)
    print(f"loaded params from {args.params}")

    splits = [args.split] if args.split != "both" else ["val", "test"]
    dirs = {"val": args.val_dir, "test": args.test_dir}

    key = jax.random.PRNGKey(args.seed)
    all_out: Dict[str, dict] = {}
    for split in splits:
        ldir = dirs[split]
        env = _build_eval_env(cfg, ldir)
        z_pool = np.asarray(env.partner_z_values)
        print(f"\n============ split = {split} "
              f"(K={env.n_layouts} layouts, "
              f"Z={env.n_partner_z} z-values, "
              f"R={env.rounds_per_episode} rounds/episode) ============")

        rng_np = np.random.default_rng(args.seed + hash(split) % (2**31))
        split_out: Dict[str, dict] = {}
        for mode in VARIANTS:
            key, sub = jax.random.split(key)
            print(f"\n[{split}] running mode = {mode} ...")
            r = _run_one_variant(params, cfg, env, sub, args.n_eps_per_z, mode)
            print(f"  overall round success: {r['round_success_overall']:.3f}")
            print(f"  per-z round success:")
            for zv, sv, nv in zip(r["partner_z_values"], r["per_z_success"],
                                   r["per_z_rounds"]):
                print(f"    z={zv:.2f}  succ={sv:.3f}  n_rounds={nv}")
            _print_curve("per-round r", r["per_round_idx_success"],
                          r["per_round_idx_n"])
            probes = _probe_all_rounds(
                r["hstates_by_round"], r["zs_by_round"], z_pool, rng_np
            )
            _print_probe(f"probe:{mode}", probes)
            r["probes_by_round"] = probes
            # Drop the raw hidden-state arrays before JSON serialization —
            # they'd be huge.
            del r["hstates_by_round"], r["zs_by_round"]
            split_out[mode] = r
        all_out[split] = split_out

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
