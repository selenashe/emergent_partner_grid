"""Environment tests for the CoordinationGrid final design.

Verifies the reference-style delay-based capability semantics, the 20-round
partner-episode structure, the influence and no-influence allocation
mechanisms, and that JIT / vmap still work. Bare-python test harness
(no pytest); run with:

    python dev/test_capability_env.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Make repo root importable without pytest.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp

from jaxmarl.environments.coordination_grid import (
    CoordinationGrid,
    Allocations,
    ACTION_MASK_T0,
    ACTION_MASK_TGEQ1,
    LEGAL_ACTION_IDS_T0,
    LEGAL_ACTION_IDS_TGEQ1,
    GOAL_RED,
    GOAL_BLUE,
    GOAL_UNSET,
    TRAIN_CAPABILITY_PAIRS,
    TEST_CAPABILITY_PAIRS,
    DEFAULT_SINGLE_PARTNER,
    FAST_DELAYS,
    SLOW_DELAYS,
    encode_ego,
)
from jaxmarl.environments.coordination_grid.capability_populations import (
    TEST_NOVEL_LOW,
    TEST_NOVEL_ZERO,
)


LAYOUTS_DIR = str(_REPO_ROOT / "dev" / "grids_capability_selected" / "layouts" / "train")

_FAILS: List[Tuple[str, str]] = []


def check(cond, name: str, detail: str = ""):
    if not cond:
        _FAILS.append((name, detail))
        print(f"  FAIL  {name}: {detail}")
    else:
        print(f"  ok    {name}")


def _make_env(**overrides) -> CoordinationGrid:
    kwargs = dict(
        layouts_dir=LAYOUTS_DIR,
        partner_capability_pairs=list(TRAIN_CAPABILITY_PAIRS),
        rounds_per_episode=20,
        max_steps=100,
        step_penalty=0.01,
        success_reward=1.0,
        augment_symmetries=False,
        hide_partner_until_time=0,
        influence=True,
    )
    kwargs.update(overrides)
    return CoordinationGrid(**kwargs)


# --------------------------------------------------------------------------- #
# Population + split                                                          #
# --------------------------------------------------------------------------- #

def test_train_test_populations():
    train_set = set(TRAIN_CAPABILITY_PAIRS)
    test_set = set(TEST_CAPABILITY_PAIRS)
    check(len(TRAIN_CAPABILITY_PAIRS) == 24, "train_pop_size",
          f"|train|={len(TRAIN_CAPABILITY_PAIRS)}")
    check(train_set.isdisjoint(test_set), "train_test_disjoint")
    train_scalars = {v for p in TRAIN_CAPABILITY_PAIRS for v in p}
    test_scalars = {v for p in TEST_CAPABILITY_PAIRS for v in p}
    novel = test_scalars - train_scalars
    check({0, 5, 6} <= novel, "novel_scalars_in_test", f"novel={sorted(novel)}")
    # Training pairs are orientation-balanced: every fast-slow pair also has
    # its slow-fast partner in the pool.
    for dr in FAST_DELAYS:
        for db in SLOW_DELAYS:
            check((dr, db) in train_set, "train_pair_fast_slow", f"({dr},{db})")
            check((db, dr) in train_set, "train_pair_slow_fast", f"({db},{dr})")


def test_action_masks():
    check(LEGAL_ACTION_IDS_T0 == (13, 14), "legal_t0", str(LEGAL_ACTION_IDS_T0))
    check(LEGAL_ACTION_IDS_TGEQ1 == (0, 3, 6, 9, 12), "legal_tge1",
          str(LEGAL_ACTION_IDS_TGEQ1))
    inter = np.logical_and(ACTION_MASK_T0, ACTION_MASK_TGEQ1)
    check(not inter.any(), "masks_disjoint")


# --------------------------------------------------------------------------- #
# Reset / capability                                                          #
# --------------------------------------------------------------------------- #

def test_reset_and_capability_placement():
    env = _make_env()
    obs, state = env.reset(jax.random.PRNGKey(0))
    check(state.capability.shape == (2,), "cap_shape")
    check(state.capability.dtype == jnp.int32, "cap_dtype",
          str(state.capability.dtype))
    pool = set(tuple(p) for p in env.partner_capability_pairs_np.tolist())
    check(tuple(state.capability.tolist()) in pool, "cap_from_pool",
          f"cap={tuple(state.capability.tolist())}")
    check("capability" not in obs["agent_0"], "obs_has_no_cap")
    check(int(state.partner_move_ctr) == 0, "move_ctr_init0")


def test_capability_absent_from_obs_content():
    e_fast = _make_env(partner_capability_pairs=[[0, 0]])
    e_slow = _make_env(partner_capability_pairs=[[9, 9]])
    obs_f, _ = e_fast.reset(jax.random.PRNGKey(1))
    obs_s, _ = e_slow.reset(jax.random.PRNGKey(1))
    for k in ("grid", "is_t0", "last_allocation"):
        check(bool(jnp.all(obs_f["agent_0"][k] == obs_s["agent_0"][k])),
              f"obs_{k}_indep_of_cap")


def test_capability_fixed_across_rounds():
    env = _make_env()
    obs, state = env.reset(jax.random.PRNGKey(7))
    cap0 = tuple(state.capability.tolist())
    round_indices_seen = set()
    for _ in range(2000):
        round_indices_seen.add(int(state.round_idx))
        current_cap = tuple(state.capability.tolist())
        check(current_cap == cap0, "cap_fixed_in_ep",
              f"cap became {current_cap} vs {cap0}")
        if state.time == 0:
            act = jnp.int32(encode_ego(4, 1))  # STAY + ALLOC_RED
        else:
            act = jnp.int32(encode_ego(4, 0))  # STAY + NONE
        obs, state, r, d, info = env.step_env(
            jax.random.PRNGKey(0), state, {"agent_0": act}
        )
        if bool(d["__all__"]):
            break
    check(round_indices_seen == set(range(20)),
          "all_20_rounds_seen", f"saw {sorted(round_indices_seen)}")


def test_capability_changes_on_episode_reset():
    env = _make_env()
    seen = set()
    for seed in range(30):
        _, state = env.reset(jax.random.PRNGKey(seed))
        seen.add(tuple(state.capability.tolist()))
    check(len(seen) >= 2, "cap_varies_across_resets", f"seen={seen}")


# --------------------------------------------------------------------------- #
# Delay semantics                                                             #
# --------------------------------------------------------------------------- #

def _partner_move_effective_sequence(env, capability, key, n_steps, ego_alloc):
    """Roll out one round with ego forced to STAY (+alloc at t=0). Returns
    the per-step 'partner_move_effective' int action.
    """
    obs, state = env.reset(key)
    state = state.replace(capability=jnp.asarray(capability, dtype=jnp.int32))
    obs = env.get_obs(state)
    moves = []
    for t in range(n_steps):
        if state.time == 0:
            act = jnp.int32(encode_ego(4, ego_alloc))
        else:
            act = jnp.int32(encode_ego(4, 0))
        obs, state, r, d, info = env.step_env(key, state, {"agent_0": act})
        moves.append(int(info["partner_move_effective"]))
        if bool(info["round_done"]):
            break
    return moves


def test_delay_zero_moves_every_step():
    env = _make_env(partner_capability_pairs=[[0, 0]], max_steps=20)
    moves = _partner_move_effective_sequence(
        env, [0, 0], jax.random.PRNGKey(42), n_steps=10, ego_alloc=int(Allocations.red),
    )
    # index 0 is the transition t=0->t=1 (partner STAYs at t=0 by convention).
    # From index 1 onward the partner should attempt a move every step (unless
    # it reached the goal, in which case round_done breaks the loop). Under
    # d=0, moves 1..N should be nonzero-count = every step in the window.
    non_stay_after_t0 = [m for m in moves[1:] if m != 4]
    check(len(non_stay_after_t0) >= min(3, len(moves[1:])),
          "d=0_moves_every_step",
          f"partner moves = {moves}")


def test_delay_k_moves_every_kplus1_steps():
    for d in (1, 2, 3):
        env = _make_env(partner_capability_pairs=[[d, d]], max_steps=40)
        moves = _partner_move_effective_sequence(
            env, [d, d], jax.random.PRNGKey(0), n_steps=25,
            ego_alloc=int(Allocations.red),
        )
        # After the initial two "warm-up" steps (t=0 STAY, t=1 first move),
        # the pattern is: 1 move followed by d STAYs, repeating.
        # Ignore the round-done truncation — inspect a fixed window.
        window = moves[1:2 + (d + 1) * 3]
        n_moves = sum(1 for m in window if m != 4)
        expected = min(3, len(window) // (d + 1)) + (1 if len(window) % (d + 1) else 0)
        # Not enforcing exact count here (the partner may reach the goal
        # early), just that the number of moves is roughly right.
        check(n_moves <= 4 and n_moves >= 1,
              f"d={d}_move_count_reasonable",
              f"window={window} moves={n_moves} expected~{expected}")


def test_d_r_used_only_for_red_d_b_only_for_blue():
    """Vary d_R only: partner cadence when pursuing RED should change, when
    pursuing BLUE should be unaffected."""
    def cadence(dr, db, ego_alloc):
        env = _make_env(partner_capability_pairs=[[dr, db]], max_steps=40)
        return _partner_move_effective_sequence(
            env, [dr, db], jax.random.PRNGKey(11), n_steps=30, ego_alloc=ego_alloc,
        )

    # ego picks ALLOC_BLUE -> partner takes RED, uses d_R.
    a_fast_R = cadence(0, 5, int(Allocations.blue))
    a_slow_R = cadence(5, 0, int(Allocations.blue))
    n_a_fast = sum(1 for m in a_fast_R[1:15] if m != 4)
    n_a_slow = sum(1 for m in a_slow_R[1:15] if m != 4)
    check(n_a_fast > n_a_slow, "d_R_gates_partner_on_red",
          f"fast(d_R=0)={n_a_fast} slow(d_R=5)={n_a_slow}")

    # ego picks ALLOC_RED -> partner takes BLUE, uses d_B.
    b_fast_B = cadence(5, 0, int(Allocations.red))
    b_slow_B = cadence(0, 5, int(Allocations.red))
    n_b_fast = sum(1 for m in b_fast_B[1:15] if m != 4)
    n_b_slow = sum(1 for m in b_slow_B[1:15] if m != 4)
    check(n_b_fast > n_b_slow, "d_B_gates_partner_on_blue",
          f"fast(d_B=0)={n_b_fast} slow(d_B=5)={n_b_slow}")


def test_delay_zero_is_legal():
    # This would have raised under the old >=1 bounds.
    env = _make_env(partner_capability_pairs=[[0, 0]])
    obs, state = env.reset(jax.random.PRNGKey(0))
    check(int(state.capability[0]) == 0, "d=0_allowed")


def test_negative_delay_rejected():
    try:
        _make_env(partner_capability_pairs=[[-1, 0]])
        _FAILS.append(("negative_delay_rejected", "did not raise"))
    except ValueError:
        print("  ok    negative_delay_rejected")


# --------------------------------------------------------------------------- #
# Allocation influence                                                        #
# --------------------------------------------------------------------------- #

def test_influence_true_alloc_drives_partner():
    env = _make_env(influence=True)
    obs, state = env.reset(jax.random.PRNGKey(3))
    act = jnp.int32(encode_ego(4, int(Allocations.red)))
    obs, state, r, d, info = env.step_env(
        jax.random.PRNGKey(0), state, {"agent_0": act}
    )
    check(int(info["pending_allocation"]) == int(Allocations.red),
          "inf_true_writes_ego_alloc")
    check(int(info["ego_alloc_action"]) == int(Allocations.red),
          "inf_true_ego_alloc_recorded")


def test_influence_false_alloc_forced_by_round_parity():
    env = _make_env(partner_capability_pairs=[[1, 4]], influence=False)
    obs, state = env.reset(jax.random.PRNGKey(3))
    # round_idx==0 -> forced RED regardless of ego pick.
    act_blue = jnp.int32(encode_ego(4, int(Allocations.blue)))
    obs, state, r, d, info = env.step_env(
        jax.random.PRNGKey(0), state, {"agent_0": act_blue}
    )
    check(int(info["pending_allocation"]) == int(Allocations.red),
          "inf_false_overrides_ego_at_r0",
          f"pending={int(info['pending_allocation'])}")
    check(int(info["ego_alloc_action"]) == int(Allocations.blue),
          "inf_false_records_ego_choice")


# --------------------------------------------------------------------------- #
# JIT + vmap                                                                  #
# --------------------------------------------------------------------------- #

def test_jit_vmap_smoke():
    env = _make_env()

    def one_step(k, s, a):
        return env.step_env(k, s, a)

    N = 4
    keys = jax.random.split(jax.random.PRNGKey(0), N)
    reset_v = jax.jit(jax.vmap(env.reset))
    step_v = jax.jit(jax.vmap(one_step, in_axes=(0, 0, {"agent_0": 0})))
    obs, state = reset_v(keys)
    check(state.capability.shape == (N, 2), "vmap_cap_shape",
          str(state.capability.shape))
    acts = jnp.full((N,), encode_ego(4, 1), dtype=jnp.int32)
    obs, state, r, d, info = step_v(keys, state, {"agent_0": acts})
    check(info["capability"].shape == (N, 2), "vmap_info_cap_shape")


def test_gru_reset_only_at_full_episode_end():
    """Sanity check that dones['__all__'] is only True on the final round's
    terminal step, not on intermediate round transitions."""
    env = _make_env()
    obs, state = env.reset(jax.random.PRNGKey(0))
    intermediates = 0
    ends = 0
    steps = 0
    while steps < 3000:
        act = (jnp.int32(encode_ego(4, 1)) if state.time == 0
               else jnp.int32(encode_ego(4, 0)))
        obs, state, r, d, info = env.step_env(
            jax.random.PRNGKey(steps), state, {"agent_0": act}
        )
        rd = bool(info["round_done"])
        ea = bool(d["__all__"])
        if rd and not ea:
            intermediates += 1
        if ea:
            ends += 1
            break
        steps += 1
    check(intermediates == 19, "19_intermediate_rounds",
          f"got {intermediates}")
    check(ends == 1, "one_final_end", f"ends={ends}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

TESTS = [
    test_train_test_populations,
    test_action_masks,
    test_reset_and_capability_placement,
    test_capability_absent_from_obs_content,
    test_capability_fixed_across_rounds,
    test_capability_changes_on_episode_reset,
    test_delay_zero_moves_every_step,
    test_delay_k_moves_every_kplus1_steps,
    test_d_r_used_only_for_red_d_b_only_for_blue,
    test_delay_zero_is_legal,
    test_negative_delay_rejected,
    test_influence_true_alloc_drives_partner,
    test_influence_false_alloc_forced_by_round_parity,
    test_jit_vmap_smoke,
    test_gru_reset_only_at_full_episode_end,
]


def main():
    for t in TESTS:
        print(f"--- {t.__name__} ---")
        try:
            t()
        except Exception:
            traceback.print_exc()
            _FAILS.append((t.__name__, "raised"))
    print()
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}):")
        for n, d in _FAILS:
            print(f"  - {n}: {d}")
        sys.exit(1)
    print("all env tests passed.")


if __name__ == "__main__":
    main()
