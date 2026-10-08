"""V3 chooses once at initialization and then matches v2 transition by transition."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmarl.environments.coordination_grid import (
    CoordinationGrid, Actions, Allocations, GOAL_BLUE, GOAL_RED, GOAL_UNSET,
    encode_ego, ego_action_mask,
)


@pytest.fixture
def layout(tmp_path):
    grid = np.ones((7, 7), dtype=int)
    grid[1:-1, 1:-1] = 0
    path = tmp_path / 'layout.json'
    path.write_text(json.dumps(dict(grid=grid.tolist(), ego_start=[5, 1], partner_start=[1, 1],
                                   red_goal=[1, 5], blue_goal=[5, 5])))
    return str(path)


def env(layout, version='online_v3', **kwargs):
    return CoordinationGrid(layout_path=layout, allocation_protocol=version,
                            partner_capability_pairs=[[2, 4]], **kwargs)


def step(environment, state, allocation, move=Actions.stay, seed=0):
    return environment.step_env(jax.random.PRNGKey(seed), state,
                                {'agent_0': encode_ego(move, allocation)})


def assert_trees_equal(left, right):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('allocation,goal', [(Allocations.red, GOAL_BLUE), (Allocations.blue, GOAL_RED)])
def test_initial_choice_stays_has_zero_cooldown_and_moves_first_at_t1(layout, allocation, goal):
    environment = env(layout)
    obs, state = environment.reset(jax.random.PRNGKey(0))
    assert int(state.partner_goal) == GOAL_UNSET
    assert int(state.partner_assignment) == Allocations.none
    assert np.flatnonzero(ego_action_mask(obs['agent_0'])).tolist() == [13, 14]
    after_obs, after, _, _, info = step(environment, state, allocation, Actions.right)
    np.testing.assert_array_equal(after.agent_pos, state.agent_pos)
    assert int(after.partner_assignment) == allocation
    assert int(after.partner_goal) == goal
    assert int(after.partner_move_ctr) == 0
    assert not bool(info['assignment_changed'])
    assert int(jnp.argmax(after_obs['agent_0']['last_allocation'])) == allocation
    assert np.flatnonzero(ego_action_mask(after_obs['agent_0'])).tolist() == [1, 2, 4, 5, 7, 8, 10, 11, 13, 14]
    _, moved, _, _, _ = step(environment, after, allocation)
    assert not np.array_equal(moved.agent_pos[1], after.agent_pos[1])
    assert int(moved.partner_move_ctr) == (4 if goal == GOAL_BLUE else 2)


def test_t1_switch_loads_destination_delay_but_t0_does_not(layout):
    environment = env(layout)
    _, state = environment.reset(jax.random.PRNGKey(0))
    _, state, _, _, _ = step(environment, state, Allocations.red)
    _, switched, _, _, info = step(environment, state, Allocations.blue)
    assert bool(info['assignment_changed'])
    np.testing.assert_array_equal(switched.agent_pos[1], state.agent_pos[1])
    assert int(switched.partner_move_ctr) == 1  # destination RED delay 2, then one wait tick


@pytest.mark.parametrize('seed', range(4))
def test_no_influence_is_identical_to_v2_through_round_resets(layout, seed):
    v2 = env(layout, 'online_v2', influence=False, rounds_per_episode=3, max_steps=5)
    v3 = env(layout, influence=False, rounds_per_episode=3, max_steps=5)
    key = jax.random.PRNGKey(seed)
    o2, s2 = v2.reset(key)
    o3, s3 = v3.reset(key)
    assert_trees_equal((o2, s2), (o3, s3))
    for tick in range(15):
        requested = 3 - int(s2.partner_assignment)  # explicitly contradict initial assignment
        mask = ego_action_mask(o3['agent_0'])
        if int(s3.time) == 0:
            assert np.flatnonzero(mask).tolist() == [12 + int(s3.partner_assignment)]
        left = step(v2, s2, requested, tick % 5, seed=tick + 10)
        right = step(v3, s3, requested, tick % 5, seed=tick + 10)
        assert_trees_equal(left, right)
        o2, s2 = left[:2]
        o3, s3 = right[:2]


@pytest.mark.parametrize('initial', [Allocations.red, Allocations.blue])
def test_all_post_initialization_dynamics_and_observations_match_v2(layout, initial):
    v2 = env(layout, 'online_v2', max_steps=25)
    v3 = env(layout, max_steps=25)
    _, s2 = v2.reset(jax.random.PRNGKey(17))
    _, s3 = v3.reset(jax.random.PRNGKey(17))
    s2 = s2.replace(last_ego_allocation=jnp.int32(initial), partner_assignment=jnp.int32(initial),
                    partner_goal=jnp.int32(GOAL_BLUE if initial == Allocations.red else GOAL_RED))
    left, right = step(v2, s2, initial), step(v3, s3, initial)
    assert_trees_equal(left, right)
    s2, s3 = left[1], right[1]
    for tick in range(1, 25):
        allocation = Allocations.blue if tick % 4 == 0 else Allocations.red
        move = tick % 5
        left, right = step(v2, s2, allocation, move, tick), step(v3, s3, allocation, move, tick)
        assert_trees_equal(left, right)
        s2, s3 = left[1], right[1]


def test_each_round_starts_with_fresh_choice_and_retains_capability(layout):
    environment = env(layout, rounds_per_episode=2, max_steps=2)
    _, start = environment.reset(jax.random.PRNGKey(0))
    _, state, _, _, _ = step(environment, start, Allocations.red)
    obs, state, _, dones, info = step(environment, state, Allocations.red)
    assert bool(info['round_done']) and not bool(dones['__all__'])
    assert int(state.round_idx) == 1 and int(state.time) == 0
    assert int(state.partner_move_ctr) == 0 and int(state.partner_goal) == GOAL_UNSET
    assert int(state.partner_assignment) == Allocations.none
    np.testing.assert_array_equal(state.capability, start.capability)
    assert np.flatnonzero(ego_action_mask(obs['agent_0'])).tolist() == [13, 14]


def test_batched_mask_handles_v3_choice_v2_forced_and_movement():
    observations = dict(last_allocation=jnp.eye(3)[jnp.array([[0, 1, 2], [0, 1, 2]])],
                        is_t0=jnp.array([[1, 1, 1], [0, 0, 0]]))
    masks = jax.jit(ego_action_mask)(observations)
    assert masks.shape == (2, 3, 15)
    np.testing.assert_array_equal(masks.sum(axis=-1), [[2, 1, 1], [10, 10, 10]])
    assert not np.asarray(masks[..., ::3]).any()
