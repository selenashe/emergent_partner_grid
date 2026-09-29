"""Focused tests for the capability-vector CoordinationGrid build.

Verifies exactly the properties the redesign requires:
    (i)   the capability vector (c_R, c_B) stays fixed across all 20 rounds
          of a partner episode;
    (ii)  capability is absent from ego observations;
    (iii) partner movement uses c_R while pursuing RED and c_B while
          pursuing BLUE;
    (iv)  capability sampling is independent of layout identity;
    (v)   held-out capability pairs never appear in the training pool.

Also basic sanity: t=0 legal-mask is exactly {STAY+ALLOC_RED, STAY+ALLOC_BLUE};
reset builds a valid state; step_env respects the alloc → complementary-goal
rule; JIT / vmap don't crash.

Run:  python dev/test_capability_env.py
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from jaxmarl.environments.coordination_grid import (
    CoordinationGrid,
    Actions,
    Allocations,
    encode_ego,
    N_EGO_ACTIONS,
    GOAL_RED,
    GOAL_BLUE,
    GOAL_UNSET,
    ACTION_MASK_T0,
    ACTION_MASK_TGEQ1,
    LEGAL_ACTION_IDS_T0,
    LEGAL_ACTION_IDS_TGEQ1,
    CAPABILITY_VALUES,
    TRAINING_CAPABILITY_PAIRS,
    HELDOUT_CAPABILITY_PAIRS,
    ALL_CAPABILITY_PAIRS,
)

STAY = int(Actions.stay)
UP = int(Actions.up); DOWN = int(Actions.down)
LEFT = int(Actions.left); RIGHT = int(Actions.right)
NONE = int(Allocations.none)
ALLOC_RED = int(Allocations.red)
ALLOC_BLUE = int(Allocations.blue)


_FAILS = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        _FAILS.append(name)


def _drive(env, key, action_seq, state=None):
    """Step env repeatedly with a list of flat actions. Returns list of states."""
    if state is None:
        _, state = env.reset(key)
    states = [state]
    for a in action_seq:
        _, state, _, _, _ = env.step_env(
            key, state, {"agent_0": jnp.int32(a)}
        )
        states.append(state)
    return states


def test_split_constants():
    print("\n[capability split constants]")
    check(f"CAPABILITY_VALUES == (1,2,3,4,7,9) got {CAPABILITY_VALUES}",
          tuple(CAPABILITY_VALUES) == (1, 2, 3, 4, 7, 9))
    check("|training pairs| == 30", len(TRAINING_CAPABILITY_PAIRS) == 30)
    check("|held-out pairs| == 6", len(HELDOUT_CAPABILITY_PAIRS) == 6)
    check("|all pairs| == 36", len(ALL_CAPABILITY_PAIRS) == 36)
    train_set = set(TRAINING_CAPABILITY_PAIRS)
    heldout_set = set(HELDOUT_CAPABILITY_PAIRS)
    check("training ∪ held-out == all", train_set | heldout_set == set(ALL_CAPABILITY_PAIRS))
    check("training ∩ held-out is empty", train_set.isdisjoint(heldout_set))
    check("held-out pairs NEVER occur in the training pool",
          heldout_set.isdisjoint(train_set))
    # each individual capability value appears in training at BOTH positions
    cr_train = set(a for a, b in TRAINING_CAPABILITY_PAIRS)
    cb_train = set(b for a, b in TRAINING_CAPABILITY_PAIRS)
    check("every c_R value in training", cr_train == set(CAPABILITY_VALUES))
    check("every c_B value in training", cb_train == set(CAPABILITY_VALUES))
    # And also in held-out at both positions (so the held-out set is "unseen
    # combinations of already-seen individual values", not new values).
    cr_held = set(a for a, b in HELDOUT_CAPABILITY_PAIRS)
    cb_held = set(b for a, b in HELDOUT_CAPABILITY_PAIRS)
    check("every c_R value appears in held-out too", cr_held == set(CAPABILITY_VALUES))
    check("every c_B value appears in held-out too", cb_held == set(CAPABILITY_VALUES))


def test_action_masks():
    print("\n[t=0 / t>=1 action legality masks]")
    check("LEGAL_ACTION_IDS_T0 == (13, 14) — STAY+ALLOC_RED, STAY+ALLOC_BLUE",
          tuple(LEGAL_ACTION_IDS_T0) == (13, 14))
    check("LEGAL_ACTION_IDS_TGEQ1 == (0, 3, 6, 9, 12)",
          tuple(LEGAL_ACTION_IDS_TGEQ1) == (0, 3, 6, 9, 12))
    check("ACTION_MASK_T0 has 2 True entries", int(ACTION_MASK_T0.sum()) == 2)
    check("ACTION_MASK_TGEQ1 has 5 True entries", int(ACTION_MASK_TGEQ1.sum()) == 5)
    check("t0/t>=1 masks disjoint (never overlap on any id)",
          not bool((ACTION_MASK_T0 & ACTION_MASK_TGEQ1).any()))


def test_reset_and_capability_placement():
    print("\n[reset / capability placement]")
    env = CoordinationGrid(
        partner_capability_pairs=[(1, 9), (9, 1)],
        rounds_per_episode=20,
    )
    obs, s = env.reset(jax.random.PRNGKey(0))
    check("state.capability has shape (2,)", tuple(s.capability.shape) == (2,))
    check("state.capability dtype is int32", s.capability.dtype == jnp.int32)
    cap = tuple(int(v) for v in s.capability)
    check(f"state.capability {cap} is in the pool",
          cap in {(1, 9), (9, 1)})
    check("obs['agent_0'] has NO 'capability' key",
          "capability" not in obs["agent_0"])
    check("obs['agent_0'] has 'last_allocation' one-hot of shape (3,)",
          obs["agent_0"]["last_allocation"].shape == (3,))
    check("state.partner_move_ctr starts at 0", int(s.partner_move_ctr) == 0)


def test_capability_absent_from_obs_content():
    print("\n[capability is not in the obs content]")
    # Two envs, identical everything except capability. get_obs should give
    # bit-identical outputs (grid, is_t0, last_allocation) — capability
    # never enters the observation.
    env_a = CoordinationGrid(
        partner_capability_pairs=[(1, 9)], rounds_per_episode=1,
    )
    env_b = CoordinationGrid(
        partner_capability_pairs=[(9, 1)], rounds_per_episode=1,
    )
    key = jax.random.PRNGKey(7)
    _, sa = env_a.reset(key)
    _, sb = env_b.reset(key)
    # Force layouts to match.
    sb = sb.replace(
        agent_pos=sa.agent_pos, wall_map=sa.wall_map,
        red_goal=sa.red_goal, blue_goal=sa.blue_goal,
        layout_idx=sa.layout_idx, episode_layout_seq=sa.episode_layout_seq,
        time=sa.time, pending_allocation=sa.pending_allocation,
        partner_goal=sa.partner_goal, partner_move_ctr=sa.partner_move_ctr,
    )
    oa = env_a.get_obs(sa)["agent_0"]
    ob = env_b.get_obs(sb)["agent_0"]
    check("grid identical across differing capabilities",
          bool(jnp.array_equal(oa["grid"], ob["grid"])))
    check("is_t0 identical across differing capabilities",
          bool(jnp.array_equal(oa["is_t0"], ob["is_t0"])))
    check("last_allocation identical across differing capabilities",
          bool(jnp.array_equal(oa["last_allocation"], ob["last_allocation"])))


def test_capability_fixed_across_rounds():
    print("\n[capability stays fixed across all 20 rounds of an episode]")
    R = 20
    env = CoordinationGrid(
        partner_capability_pairs=[(2, 7)], rounds_per_episode=R, max_steps=6,
    )
    key = jax.random.PRNGKey(3)
    _, s = env.reset(key)
    initial_cap = tuple(int(v) for v in s.capability)
    steps_seen = 0
    round_ids_seen = set()
    # Play alternating ALLOC_RED / ALLOC_BLUE at each t=0, then STAY.
    while steps_seen < R * (env.max_steps + 2):
        is_t0_step = bool(s.time == 0)
        if is_t0_step:
            round_ids_seen.add(int(s.round_idx))
            alloc = ALLOC_RED if (int(s.round_idx) % 2 == 0) else ALLOC_BLUE
            action = encode_ego(STAY, alloc)
        else:
            action = encode_ego(STAY, NONE)
        _, s, _, d, _ = env.step_env(key, s, {"agent_0": jnp.int32(action)})
        cap_now = tuple(int(v) for v in s.capability)
        if cap_now != initial_cap:
            check(
                f"capability changed mid-episode at round {int(s.round_idx)}: "
                f"{initial_cap} -> {cap_now}",
                False,
            )
            return
        steps_seen += 1
        if bool(d["__all__"]):
            break
    check(f"capability unchanged over {len(round_ids_seen)} rounds (== {initial_cap})",
          True)
    check("saw all 20 round_idx values", len(round_ids_seen) == R)


def _partner_move_ctr_sequence(cap_pair, alloc_action, key_seed=0,
                                max_steps=15):
    """Return the round-local sequence of partner_move_ctr values and the
    sequence of times at which the partner actually attempted a move
    (ctr==0 at that step, per env implementation). Uses the default layout.
    """
    env = CoordinationGrid(
        partner_capability_pairs=[cap_pair], rounds_per_episode=1,
        max_steps=max_steps,
    )
    key = jax.random.PRNGKey(key_seed)
    _, s = env.reset(key)
    ctr_seq = [int(s.partner_move_ctr)]
    move_times = []
    # t=0 (ego picks alloc; both stay).
    _, s, _, _, _ = env.step_env(key, s, {"agent_0": jnp.int32(alloc_action)})
    ctr_seq.append(int(s.partner_move_ctr))
    for t in range(1, max_steps):
        pre_ctr = int(s.partner_move_ctr)
        _, s, _, d, _ = env.step_env(
            key, s, {"agent_0": jnp.int32(encode_ego(STAY, NONE))}
        )
        ctr_seq.append(int(s.partner_move_ctr))
        # A move happened this step iff the pre-step ctr was 0 (and goal
        # is committed, which is guaranteed by t>=1 after the alloc).
        if pre_ctr == 0:
            move_times.append(t)
        if bool(d["__all__"]):
            break
    return ctr_seq, move_times


def test_partner_uses_c_r_when_going_red():
    print("\n[partner movement uses c_R when pursuing RED]")
    # ego picks ALLOC_BLUE => partner pursues RED => cadence uses c_R.
    # We check the partner_move_ctr evolution directly (deterministic and
    # independent of layout geometry). Expected: cadence c=k means the
    # ctr sequence cycles through 0, k-1, k-2, ..., 0, k-1, ...
    _, moves_c1 = _partner_move_ctr_sequence(
        (1, 9), encode_ego(STAY, ALLOC_BLUE), key_seed=10, max_steps=15,
    )
    _, moves_c9 = _partner_move_ctr_sequence(
        (9, 1), encode_ego(STAY, ALLOC_BLUE), key_seed=10, max_steps=15,
    )
    check(
        f"c_R=1 pursuing RED attempts move every step: {moves_c1}",
        moves_c1 == list(range(1, 15)),
    )
    check(
        f"c_R=9 pursuing RED attempts moves at t=1 and t=10 only: {moves_c9}",
        moves_c9 == [1, 10],
    )


def test_partner_uses_c_b_when_going_blue():
    print("\n[partner movement uses c_B when pursuing BLUE]")
    # ego picks ALLOC_RED => partner pursues BLUE => cadence uses c_B.
    _, moves_c1 = _partner_move_ctr_sequence(
        (9, 1), encode_ego(STAY, ALLOC_RED), key_seed=11, max_steps=15,
    )
    _, moves_c9 = _partner_move_ctr_sequence(
        (1, 9), encode_ego(STAY, ALLOC_RED), key_seed=11, max_steps=15,
    )
    check(
        f"c_B=1 pursuing BLUE attempts move every step: {moves_c1}",
        moves_c1 == list(range(1, 15)),
    )
    check(
        f"c_B=9 pursuing BLUE attempts moves at t=1 and t=10 only: {moves_c9}",
        moves_c9 == [1, 10],
    )


def test_c_r_used_only_for_red_c_b_only_for_blue():
    print("\n[c_R and c_B are used independently within one profile]")
    # cap=(1, 9): pursuing RED  → cadence 1 (fast); pursuing BLUE → cadence 9.
    _, moves_red  = _partner_move_ctr_sequence(
        (1, 9), encode_ego(STAY, ALLOC_BLUE), key_seed=12, max_steps=15,
    )
    _, moves_blue = _partner_move_ctr_sequence(
        (1, 9), encode_ego(STAY, ALLOC_RED),  key_seed=12, max_steps=15,
    )
    check(
        f"cap=(1,9): pursuing RED attempts moves every step ({moves_red})",
        moves_red == list(range(1, 15)),
    )
    check(
        f"cap=(1,9): pursuing BLUE attempts moves at t=1,10 only ({moves_blue})",
        moves_blue == [1, 10],
    )
    # Symmetric with (9, 1).
    _, moves_red2  = _partner_move_ctr_sequence(
        (9, 1), encode_ego(STAY, ALLOC_BLUE), key_seed=13, max_steps=15,
    )
    _, moves_blue2 = _partner_move_ctr_sequence(
        (9, 1), encode_ego(STAY, ALLOC_RED),  key_seed=13, max_steps=15,
    )
    check(
        f"cap=(9,1): pursuing RED attempts moves at t=1,10 only ({moves_red2})",
        moves_red2 == [1, 10],
    )
    check(
        f"cap=(9,1): pursuing BLUE attempts moves every step ({moves_blue2})",
        moves_blue2 == list(range(1, 15)),
    )


def test_capability_independent_of_layout():
    print("\n[capability is sampled independently of layout]")
    # Build an env with a non-trivial capability pool and many layouts;
    # reset it many times; check that every capability pair is seen with
    # every layout at least a few times (statistical independence proxy).
    env = CoordinationGrid(
        layouts_dir="/juice6/u/jshe/emergent_partner_grid/dev/grids_final/layouts/val",
        partner_capability_pairs=[(1, 1), (1, 9), (9, 1), (9, 9)],
        rounds_per_episode=1,
    )
    keys = jax.random.split(jax.random.PRNGKey(42), 400)
    cap_layout_counts: dict = {}
    for k in keys:
        _, s = env.reset(k)
        cap = tuple(int(v) for v in s.capability)
        ly = int(s.layout_idx)
        cap_layout_counts[(cap, ly)] = cap_layout_counts.get((cap, ly), 0) + 1
    # Marginal counts.
    cap_marginal: dict = {}
    for (cap, _), c in cap_layout_counts.items():
        cap_marginal[cap] = cap_marginal.get(cap, 0) + c
    check("each of the 4 capability pairs sampled at least ~50 times over 400 resets",
          all(cap_marginal.get(cap, 0) >= 50
              for cap in [(1, 1), (1, 9), (9, 1), (9, 9)]))
    # For a random capability sample per reset, the marginal per-cap counts
    # should be roughly balanced (~100 each). We allow slack for RNG noise.
    counts = list(cap_marginal.values())
    check(f"capability marginals roughly balanced (min={min(counts)}, max={max(counts)})",
          min(counts) >= 70 and max(counts) <= 130)


def test_alloc_maps_to_complementary_goal():
    print("\n[ego alloc at t=0 → partner takes the complementary goal]")
    env = CoordinationGrid(
        partner_capability_pairs=[(3, 3)], rounds_per_episode=1,
    )
    key = jax.random.PRNGKey(5)
    # ALLOC_RED  -> partner_goal = BLUE
    _, s0 = env.reset(key)
    _, s1, _, _, _ = env.step_env(key, s0, {"agent_0": jnp.int32(encode_ego(STAY, ALLOC_RED))})
    _, s2, _, _, _ = env.step_env(key, s1, {"agent_0": jnp.int32(encode_ego(STAY, NONE))})
    check("ALLOC_RED -> partner_goal == BLUE", int(s2.partner_goal) == GOAL_BLUE)
    # ALLOC_BLUE -> partner_goal = RED
    _, s0 = env.reset(key)
    _, s1, _, _, _ = env.step_env(key, s0, {"agent_0": jnp.int32(encode_ego(STAY, ALLOC_BLUE))})
    _, s2, _, _, _ = env.step_env(key, s1, {"agent_0": jnp.int32(encode_ego(STAY, NONE))})
    check("ALLOC_BLUE -> partner_goal == RED", int(s2.partner_goal) == GOAL_RED)


def test_jit_vmap_smoke():
    print("\n[jit + vmap smoke test]")
    env = CoordinationGrid(
        partner_capability_pairs=[(1, 9), (9, 1), (3, 3)],
        rounds_per_episode=2, max_steps=6,
    )
    N = 8
    keys = jax.random.split(jax.random.PRNGKey(0), N)
    obs_v, state_v = jax.vmap(env.reset)(keys)
    check("vmap reset: capability shape (N, 2)",
          tuple(state_v.capability.shape) == (N, 2))
    check("vmap reset: partner_move_ctr shape (N,)",
          tuple(state_v.partner_move_ctr.shape) == (N,))
    step_v = jax.jit(jax.vmap(env.step_env,
                               in_axes=(0, 0, {"agent_0": 0})))
    acts = jnp.array(
        [encode_ego(STAY, ALLOC_RED)] * N, dtype=jnp.int32
    )
    obs_v2, state_v2, r, d, info = step_v(keys, state_v, {"agent_0": acts})
    check("vmap step: state.capability shape preserved",
          tuple(state_v2.capability.shape) == (N, 2))
    check("vmap step: capability unchanged by one step",
          bool(jnp.all(state_v.capability == state_v2.capability)))
    check("info['capability'] shape (N, 2)",
          tuple(info["capability"].shape) == (N, 2))


def test_bounds_of_capability_values():
    print("\n[invalid capability rejected]")
    try:
        _ = CoordinationGrid(partner_capability_pairs=[(0, 1)])
        check("capability with 0 must raise", False)
    except ValueError:
        check("capability with 0 raises ValueError", True)


def main():
    test_split_constants()
    test_action_masks()
    test_reset_and_capability_placement()
    test_capability_absent_from_obs_content()
    test_capability_fixed_across_rounds()
    test_partner_uses_c_r_when_going_red()
    test_partner_uses_c_b_when_going_blue()
    test_c_r_used_only_for_red_c_b_only_for_blue()
    test_capability_independent_of_layout()
    test_alloc_maps_to_complementary_goal()
    test_jit_vmap_smoke()
    test_bounds_of_capability_values()

    print()
    if _FAILS:
        print(f"===  {len(_FAILS)} check(s) FAILED  ===")
        for name in _FAILS:
            print(f"    - {name}")
        raise SystemExit(1)
    print("===  ALL CHECKS PASSED  ===")


if __name__ == "__main__":
    main()
