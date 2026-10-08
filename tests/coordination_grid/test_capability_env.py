"""Environment tests for the CoordinationGrid final design.

Verifies the reference-style delay-based capability semantics, the 20-round
partner-episode structure, the influence and no-influence allocation
mechanisms, and that JIT / vmap still work. Bare-python test harness
(no pytest); run with:

    python tests/coordination_grid/test_capability_env.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Make repo root importable without pytest.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp

from jaxmarl.environments.coordination_grid import (
    CoordinationGrid,
    Actions,
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
    ego_action_mask,
)
from jaxmarl.environments.coordination_grid.capability_populations import (
    TEST_NOVEL_LOW,
    TEST_NOVEL_ZERO,
)


LAYOUTS_DIR = str(_REPO_ROOT / "data_prep" / "grids_capability_selected_balanced_1096" / "layouts" / "train")

_FAILS: List[Tuple[str, str]] = []


def check(cond, name: str, detail: str = ""):
    if not cond:
        _FAILS.append((name, detail))
        print(f"  FAIL  {name}: {detail}")
    else:
        print(f"  ok    {name}")


_ENV_CACHE = {}


def _force_initial(state, alloc):
    return state.replace(last_ego_allocation=jnp.int32(alloc),
        partner_assignment=jnp.int32(alloc),
        partner_goal=jnp.int32(GOAL_BLUE if int(alloc)==1 else GOAL_RED))


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
    import json
    cache_key = json.dumps(kwargs, sort_keys=True)
    if cache_key not in _ENV_CACHE:
        env = CoordinationGrid(**kwargs)
        env.step_env = jax.jit(env.step_env)
        _ENV_CACHE[cache_key] = env
    return _ENV_CACHE[cache_key]


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
    expected = tuple(3*m+a for m in range(5) for a in (1,2))
    check(LEGAL_ACTION_IDS_TGEQ1 == expected, "ten_online_actions")
    check(not ACTION_MASK_TGEQ1[::3].any(), "none_excluded")
    for a in (1,2):
        obs = {"is_t0": jnp.float32(1), "last_allocation": jax.nn.one_hot(a,3)}
        mask = np.asarray(ego_action_mask(obs))
        check(np.flatnonzero(mask).tolist() == [encode_ego(4,a)], "forced_default", str(a))
        obs["is_t0"] = jnp.float32(0)
        check(np.array_equal(np.asarray(ego_action_mask(obs)), ACTION_MASK_TGEQ1), "online_mask")



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
            act = jnp.int32(encode_ego(4, 1))  # STAY + NONE
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
    state = _force_initial(state, ego_alloc)
    obs = env.get_obs(state)
    moves = []
    for t in range(n_steps):
        if state.time == 0:
            act = jnp.int32(encode_ego(4, ego_alloc))
        else:
            act = jnp.int32(encode_ego(4, ego_alloc))
        obs, state, r, d, info = env.step_env(key, state, {"agent_0": act})
        moves.append(int(info["partner_move_effective"]))
        if bool(info["round_done"]):
            break
    return moves


def _pick_layout_with_longest_partner_path(env, goal_alloc):
    """Return (layout_idx, partner_path_len) for the layout with the
    longest partner-start -> partner-goal BFS path in the loaded corpus.
    """
    partner_goal_np_key = ("blue_goals" if int(goal_alloc) == int(Allocations.red)
                           else "red_goals")
    from jaxmarl.environments.coordination_grid.coordination_grid import bfs_distance_map
    best_i, best_d = -1, -1
    for i in range(env.n_layouts):
        walls = np.asarray(env.wall_maps[i])
        goal_xy = np.asarray(getattr(env, partner_goal_np_key)[i])
        pstart = np.asarray(env.partner_starts[i])
        dist = bfs_distance_map(walls, goal_xy)
        d = int(dist[pstart[1], pstart[0]])
        if d > best_d:
            best_d = d
            best_i = i
    return best_i, best_d


def _exact_move_sequence_on_layout(env, layout_idx, capability, goal_alloc, n_steps):
    """Force a specific layout via reset_from_schedule and record the exact
    per-step partner_move_effective for a round where the ego STAYs and holds
    ``goal_alloc`` at t=0.
    """
    R = env.rounds_per_episode
    layout_seq = jnp.full((R,), layout_idx, dtype=jnp.int32)
    obs, state = env.reset_from_schedule(
        jnp.asarray(capability, dtype=jnp.int32), layout_seq, jax.random.PRNGKey(0),
    )
    state = _force_initial(state, goal_alloc)
    moves = []
    for _ in range(n_steps):
        if state.time == 0:
            act = jnp.int32(encode_ego(4, int(goal_alloc)))
        else:
            act = jnp.int32(encode_ego(4, int(goal_alloc)))
        obs, state, r, d, info = env.step_env(jax.random.PRNGKey(0), state, {"agent_0": act})
        moves.append(int(info["partner_move_effective"]))
        if bool(info["round_done"]):
            break
    return moves


def _expected_move_step_indices(d: int, n_slots: int):
    """Under new delay semantics, partner move happens whenever partner_move_ctr==0.
    Timeline (state.time -> action taken during that step):
        t=0  : STAY (is_time0)
        t=1  : MOVE  (ctr=0 -> reset to d)
        t=2  : STAY if d>=1 else MOVE
        ...
    Move at absolute step-index k (1-indexed among the moves[] list from
    _exact_move_sequence_on_layout) iff (k-1) % (d+1) == 0 AND k >= 1.
    Return the list of expected move-slot indices (relative to moves[]).
    """
    return [k for k in range(1, n_slots) if (k - 1) % (d + 1) == 0]


def _n_steps_for(d, path_len, n_move_cycles=3):
    """Number of env steps to observe ``n_move_cycles`` partner moves
    without letting the partner reach the goal (which stops the round).
    """
    # k-th move happens at env-step index 1 + (k-1)*(d+1). We want to see
    # cycles until either n_move_cycles or path_len-1 moves have occurred.
    n_visible = min(n_move_cycles, max(path_len - 1, 1))
    return 1 + (n_visible - 1) * (d + 1) + 1  # +1 slot to confirm the last move slot


def test_exact_cooldown_cadence_symmetric():
    """For each d, pick the layout with the longest partner path so the
    partner can't finish before we've seen the intended cadence, then
    verify the exact per-step move-timing.
    """
    for d in (0, 1, 2, 3, 5):
        env = _make_env(partner_capability_pairs=[[d, d]], max_steps=100)
        # ego alloc = RED  =>  partner takes BLUE  =>  BLUE path counts.
        li, path_len = _pick_layout_with_longest_partner_path(
            env, int(Allocations.red),
        )
        check(li >= 0 and path_len >= 3,
              f"d={d}_has_usable_layout", f"li={li} path_len={path_len}")
        n_steps = _n_steps_for(d, path_len, n_move_cycles=3)
        moves = _exact_move_sequence_on_layout(
            env, li, [d, d], int(Allocations.red), n_steps=n_steps,
        )
        observed = [k for k, m in enumerate(moves) if m != int(Actions.stay)]
        expected = [k for k in range(1, len(moves)) if (k - 1) % (d + 1) == 0]
        check(observed == expected,
              f"d={d}_exact_cadence",
              f"path_len={path_len} obs={observed} exp={expected}")


def test_exact_cooldown_asymmetric_uses_correct_delay():
    """d_R != d_B: ensure RED goal uses d_R and BLUE goal uses d_B."""
    # d=(2, 5) => ego picks ALLOC_BLUE => partner takes RED, uses d_R=2, cadence 3
    env_r = _make_env(partner_capability_pairs=[[2, 5]], max_steps=100)
    li_r, plen_r = _pick_layout_with_longest_partner_path(env_r, int(Allocations.blue))
    check(li_r >= 0 and plen_r >= 3, "asym_R_usable_layout", f"plen={plen_r}")
    n_r = _n_steps_for(2, plen_r, n_move_cycles=3)
    moves = _exact_move_sequence_on_layout(
        env_r, li_r, [2, 5], int(Allocations.blue), n_steps=n_r,
    )
    observed = [k for k, m in enumerate(moves) if m != int(Actions.stay)]
    expected = [k for k in range(1, len(moves)) if (k - 1) % 3 == 0]
    check(observed == expected, "d_R=2_used_on_RED",
          f"obs={observed} exp={expected}")

    # d=(2, 5) => ego picks ALLOC_RED => partner takes BLUE, uses d_B=5, cadence 6
    env_b = _make_env(partner_capability_pairs=[[2, 5]], max_steps=100)
    li_b, plen_b = _pick_layout_with_longest_partner_path(env_b, int(Allocations.red))
    check(li_b >= 0 and plen_b >= 3, "asym_B_usable_layout", f"plen={plen_b}")
    n_b = _n_steps_for(5, plen_b, n_move_cycles=3)
    moves = _exact_move_sequence_on_layout(
        env_b, li_b, [2, 5], int(Allocations.red), n_steps=n_b,
    )
    observed = [k for k, m in enumerate(moves) if m != int(Actions.stay)]
    expected = [k for k in range(1, len(moves)) if (k - 1) % 6 == 0]
    check(observed == expected, "d_B=5_used_on_BLUE",
          f"obs={observed} exp={expected}")


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
    _, state = env.reset(jax.random.PRNGKey(3))
    initial = int(state.partner_assignment)
    pos = np.asarray(state.agent_pos).copy()
    opposite = 3-initial
    obs, state, _, _, info = env.step_env(jax.random.PRNGKey(0), state,
        {"agent_0": encode_ego(0,opposite)})
    check(int(state.partner_assignment) == initial, "t0_cannot_override_default")
    check(np.array_equal(pos,np.asarray(state.agent_pos)), "both_stay_t0")
    obs, state, _, _, info = env.step_env(jax.random.PRNGKey(1), state,
        {"agent_0": encode_ego(4,opposite)})
    check(int(info["partner_assignment"]) == opposite, "t1_ego_controls_goal")
    check(bool(info["assignment_changed"]), "switch_recorded")



def test_influence_false_alloc_keeps_random_default():
    env = _make_env(partner_capability_pairs=[[1,4]], influence=False)
    _, state = env.reset(jax.random.PRNGKey(3))
    initial = int(state.partner_assignment)
    for t in range(4):
        _, state, _, _, info = env.step_env(jax.random.PRNGKey(t),state,
            {"agent_0": encode_ego(4,3-initial)})
        check(int(info["partner_assignment"]) == initial, "noinf_keeps_default")
        check(not bool(info["assignment_changed"]), "noinf_no_switch")



def test_influence_false_partner_goal_invariant_to_ego_alloc():
    env = _make_env(partner_capability_pairs=[[1,4]], influence=False)
    cap = jnp.asarray([1,4]); layouts = jnp.zeros((20,),dtype=jnp.int32)
    def run(a):
        _, state = env.reset_from_schedule(cap,layouts,jax.random.PRNGKey(0))
        for t in range(3):
            _, state, _, _, info = env.step_env(jax.random.PRNGKey(t),state,
                {"agent_0": encode_ego(4,a)})
        return int(info["partner_goal"]),int(state.partner_assignment),np.asarray(state.agent_pos[1])
    red, blue = run(1),run(2)
    check(red[:2] == blue[:2] and np.array_equal(red[2],blue[2]), "noinf_goal_and_motion_invariant")



def test_influence_true_partner_goal_flips_with_ego_alloc():
    env = _make_env(partner_capability_pairs=[[1,4]], influence=True)
    _, state = env.reset(jax.random.PRNGKey(0))
    _, state, *_ = env.step_env(jax.random.PRNGKey(0),state,{"agent_0": encode_ego(4,1)})
    for a in (1,2,1,2):
        _, state, _, _, info = env.step_env(jax.random.PRNGKey(a),state,
            {"agent_0": encode_ego(4,a)})
        expected = GOAL_BLUE if a == 1 else GOAL_RED
        check(int(info["partner_goal"]) == expected, "online_reassignment",str(a))



def test_observation_schema_matches_across_influence():
    cap = jnp.asarray([1,4]); layouts = jnp.zeros((20,),dtype=jnp.int32)
    def run(influence):
        env = _make_env(partner_capability_pairs=[[1,4]],influence=influence)
        obs,state = env.reset_from_schedule(cap,layouts,jax.random.PRNGKey(0))
        traces = [np.asarray(obs["agent_0"]["last_allocation"])]
        for t,a in enumerate((1,1,2)):
            obs,state,*_ = env.step_env(jax.random.PRNGKey(t),state,{"agent_0": encode_ego(4,a)})
            traces.append(np.asarray(obs["agent_0"]["last_allocation"]))
        return traces
    left,right = run(True),run(False)
    check(np.array_equal(left,right), "allocation_obs_same_semantics")
    check(np.argmax(left[-1]) == 2, "observation_echoes_online_request")
    check(all(v[0] == 0 for v in left), "obs_never_none")



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
               else jnp.int32(encode_ego(4, 1)))
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

# --------------------------------------------------------------------------- #
# Analytical optimal-allocation primitive                                     #
# --------------------------------------------------------------------------- #

def test_info_round_time_is_post_transition():
    """Regression: info["round_time"] must be the POST-transition round-
    local time, not the pre-step time. The pre-step time is state.time
    entering the step; round_time must equal state.time + 1 = new_time.
    """
    env = _make_env()
    obs, state = env.reset(jax.random.PRNGKey(0))
    for i in range(5):
        pre_step_time = int(state.time)
        act = (jnp.int32(encode_ego(4, 1)) if state.time == 0
               else jnp.int32(encode_ego(4, 1)))
        obs, state, r, d, info = env.step_env(
            jax.random.PRNGKey(i), state, {"agent_0": act}
        )
        check(int(info["round_time"]) == pre_step_time + 1,
              f"info_round_time_step{i}",
              f"got {int(info['round_time'])} expected {pre_step_time + 1}")
        if bool(info["round_done"]):
            break


def test_analytical_optimal_allocation_hand_crafted():
    """Sanity check the analytical primitive used by the evaluator.

    Hand-computed cases:
      * partner-fast on RED (d_R=0) and slow on BLUE (d_B=9), with roughly
        equal BFS distances => optimal is ego->BLUE, partner->RED (alloc B).
      * symmetric swap => optimal flips to alloc A.

    Uses capability_selection.completion_time (single primitive shared
    between validation, selection, and the evaluator).
    """
    from data_prep.capability_selection import completion_time, _reward_from_time
    max_steps = 100
    step_penalty = 0.01
    success_reward = 1.0
    ego_r, ego_b, part_r, part_b = 4, 4, 4, 4      # symmetric geometry

    def reward_pair(d_R, d_B):
        # A: ego RED, partner BLUE (partner uses d_B)
        tA = completion_time(ego_r, part_b, d_B, max_steps)
        # B: ego BLUE, partner RED (partner uses d_R)
        tB = completion_time(ego_b, part_r, d_R, max_steps)
        rA = _reward_from_time(tA, max_steps, step_penalty, success_reward)
        rB = _reward_from_time(tB, max_steps, step_penalty, success_reward)
        return rA, rB, tA, tB

    rA, rB, tA, tB = reward_pair(0, 9)  # partner fast at RED, slow at BLUE
    check(rB > rA, "opt_alloc_B_when_partner_fast_at_RED",
          f"rA={rA:.3f} tA={tA}, rB={rB:.3f} tB={tB}")
    rA, rB, tA, tB = reward_pair(9, 0)  # partner slow at RED, fast at BLUE
    check(rA > rB, "opt_alloc_A_when_partner_fast_at_BLUE",
          f"rA={rA:.3f} tA={tA}, rB={rB:.3f} tB={tB}")
    # Symmetric d => tie
    rA, rB, tA, tB = reward_pair(2, 2)
    check(abs(rA - rB) < 1e-9, "opt_alloc_tie_when_d_symmetric",
          f"rA={rA:.5f} rB={rB:.5f}")

    # Allocation regret: reward gap between chosen and optimal.
    rA, rB, *_ = reward_pair(0, 9)
    optimal = max(rA, rB)
    regret_wrong = optimal - min(rA, rB)
    regret_right = optimal - optimal
    check(regret_wrong > 0.0 and regret_right == 0.0,
          "alloc_regret_sign", f"wrong={regret_wrong:.4f} right={regret_right}")


TESTS = [
    test_train_test_populations,
    test_action_masks,
    test_reset_and_capability_placement,
    test_capability_absent_from_obs_content,
    test_capability_fixed_across_rounds,
    test_capability_changes_on_episode_reset,
    test_exact_cooldown_cadence_symmetric,
    test_exact_cooldown_asymmetric_uses_correct_delay,
    test_d_r_used_only_for_red_d_b_only_for_blue,
    test_delay_zero_is_legal,
    test_negative_delay_rejected,
    test_influence_true_alloc_drives_partner,
    test_influence_false_alloc_keeps_random_default,
    test_influence_false_partner_goal_invariant_to_ego_alloc,
    test_influence_true_partner_goal_flips_with_ego_alloc,
    test_observation_schema_matches_across_influence,
    test_info_round_time_is_post_transition,
    test_analytical_optimal_allocation_hand_crafted,
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
