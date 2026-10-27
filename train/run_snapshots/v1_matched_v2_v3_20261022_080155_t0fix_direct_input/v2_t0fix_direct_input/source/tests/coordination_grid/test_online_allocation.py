"""Behavioral regressions for random defaults and within-round allocation."""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines/IPPO"))
from jaxmarl.environments.coordination_grid import (
    CoordinationGrid, Actions, Allocations, GOAL_BLUE, GOAL_RED,
    encode_ego, ego_action_mask, ALLOCATION_PROTOCOL,
)
import ippo_rnn_coordination_grid as trainer
from analysis import evaluate_partner_modelling as evaluator


@pytest.fixture
def layout(tmp_path):
    grid = np.ones((7, 7), dtype=int)
    grid[1:-1, 1:-1] = 0
    path = tmp_path / "layout.json"
    path.write_text(json.dumps({"grid": grid.tolist(), "ego_start": [5, 1],
        "partner_start": [1, 1], "red_goal": [1, 5], "blue_goal": [5, 5]}))
    return str(path)


def make_env(layout, **kwargs):
    return CoordinationGrid(layout_path=layout, rounds_per_episode=2,
                            partner_capability_pairs=[[0, 2]], **kwargs)


def initial(state, alloc):
    return state.replace(last_ego_allocation=jnp.int32(alloc),
        partner_assignment=jnp.int32(alloc),
        partner_goal=jnp.int32(GOAL_BLUE if alloc == 1 else GOAL_RED))


def step(env, state, alloc, move=Actions.stay, seed=0):
    return env.step_env(jax.random.PRNGKey(seed), state,
                        {"agent_0": encode_ego(move, alloc)})


def test_random_initialization_is_visible_reproducible_and_capability_independent(layout):
    env = make_env(layout)
    layouts = jnp.zeros((2,), dtype=jnp.int32)
    seen = set()
    for seed in range(24):
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_from_schedule(jnp.array([0, 2]), layouts, key)
        obs2, state2 = env.reset_from_schedule(jnp.array([9, 9]), layouts, key)
        a = int(state.partner_assignment)
        seen.add(a)
        assert a in (1, 2)
        assert int(jnp.argmax(obs["agent_0"]["last_allocation"])) == a
        assert int(state2.partner_assignment) == a
        for field in obs["agent_0"]:
            np.testing.assert_array_equal(obs["agent_0"][field], obs2["agent_0"][field])
    assert seen == {1, 2}


def test_both_stay_at_zero_then_movement_and_assignment_are_joint_choices(layout):
    env = make_env(layout)
    _, state = env.reset(jax.random.PRNGKey(0))
    old_positions = np.asarray(state.agent_pos)
    old_alloc = int(state.partner_assignment)
    _, state, _, _, info = step(env, state, 3-old_alloc, Actions.up)
    np.testing.assert_array_equal(state.agent_pos, old_positions)
    assert int(info["ego_alloc_action"]) == old_alloc
    assert not bool(info["allocation_decision"])
    _, state, _, _, info = step(env, state, Allocations.blue, Actions.up)
    assert int(info["partner_goal"]) == GOAL_RED
    np.testing.assert_array_equal(state.agent_pos[0], [1, 4])
    np.testing.assert_array_equal(state.agent_pos[1], [2, 1])
    assert bool(info["allocation_decision"])


def test_repeating_assignment_preserves_cadence_and_switch_uses_destination_delay(layout):
    env = make_env(layout)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = initial(state, 1)  # partner BLUE: delay 2
    movements = []
    for _ in range(5):
        _, state, _, _, info = step(env, state, 1)
        movements.append(int(info["partner_move_effective"]) != Actions.stay)
    assert movements == [False, True, False, False, True]
    # Switching to RED (delay 0) immediately changes direction and can move.
    _, state, _, _, info = step(env, state, 2)
    assert bool(info["assignment_changed"])
    assert int(info["partner_goal"]) == GOAL_RED
    # Switching back loads BLUE's delay, instead of giving a free movement.
    _, state, _, _, info = step(env, state, 1)
    assert int(info["partner_move_effective"]) == Actions.stay
    assert int(state.partner_move_ctr) == 1
    _, state, _, _, info = step(env, state, 1)
    assert int(info["partner_move_effective"]) == Actions.stay
    _, state, _, _, info = step(env, state, 1)
    assert int(info["partner_move_effective"]) != Actions.stay


def test_no_influence_freezes_default_but_observation_echoes_requests(layout):
    env = make_env(layout, influence=False)
    obs, state = env.reset(jax.random.PRNGKey(0))
    a = int(state.partner_assignment)
    for t, request in enumerate((a, 3-a, a, 3-a)):
        obs, state, _, _, info = step(env, state, request)
        assert int(info["partner_assignment"]) == a
        assert not bool(info["assignment_changed"])
        if t:
            assert int(jnp.argmax(obs["agent_0"]["last_allocation"])) == request


def test_each_round_resamples_default_and_only_full_episode_terminates_under_jit(layout):
    env = make_env(layout, max_steps=2)
    keys = jax.random.split(jax.random.PRNGKey(0), 16)
    reset = jax.jit(jax.vmap(env.reset))
    advance = jax.jit(jax.vmap(env.step_env, in_axes=(0, 0, {"agent_0": 0})))
    obs, state = reset(keys)
    cap = np.asarray(state.capability)
    actions = {"agent_0": jnp.full((16,), encode_ego(4, 1))}
    for _ in range(2):
        obs, state, _, done, info = advance(keys, state, actions)
    assert np.all(np.asarray(info["round_done"]))
    assert not np.any(np.asarray(done["__all__"]))
    assert np.all(np.asarray(state.time) == 0)
    np.testing.assert_array_equal(state.capability, cap)
    assert set(np.asarray(state.partner_assignment).tolist()) == {1, 2}
    assert np.all(np.asarray(state.partner_assignment) == np.argmax(obs["agent_0"]["last_allocation"], axis=-1))
    for _ in range(2):
        obs, state, _, done, _ = advance(keys, state, actions)
    assert np.all(np.asarray(done["__all__"]))


@pytest.mark.parametrize("model_type", ["rnn", "mlp"])
def test_network_excludes_none_and_has_no_initial_assignment_choice(layout, model_type):
    env = make_env(layout)
    obs, _ = env.reset(jax.random.PRNGKey(0))
    config = dict(MODEL_TYPE=model_type, ACTIVATION="relu", GRU_HIDDEN_DIM=128,
                  FC_DIM_SIZE=128, GRID_EMB_DIM=64, MSG_EMB_DIM=8)
    net, init, _ = trainer.build_network(config, env.n_ego_actions)
    obs = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs["agent_0"])
    resets = jnp.zeros((1, 1), dtype=bool)
    params = net.init(jax.random.PRNGKey(0), init(1), (obs, resets))
    _, pi, _ = net.apply(params, init(1), (obs, resets))
    prob = np.asarray(pi.probs)[0, 0]
    assert np.count_nonzero(prob) == 1
    assert float(pi.entropy()[0, 0]) == 0
    obs["is_t0"] = jnp.zeros((1, 1))
    _, pi, _ = net.apply(params, init(1), (obs, resets))
    prob = np.asarray(pi.probs)[0, 0]
    assert np.count_nonzero(prob) == 10
    assert np.all(prob[::3] == 0)
    assert np.isfinite(np.asarray(pi.entropy())).all()


def test_legacy_config_is_rejected():
    with pytest.raises(ValueError, match="fixed-allocation"):
        trainer.require_current_protocol({})


def test_unchanged_policy_replays_collection_across_episode_reset(layout):
    env = make_env(layout)
    obs, _ = env.reset(jax.random.PRNGKey(0))
    config = dict(MODEL_TYPE="rnn", ACTIVATION="relu", GRU_HIDDEN_DIM=128, FC_DIM_SIZE=128)
    net, init, _ = trainer.build_network(config, env.n_ego_actions)
    observations = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (6, 1) + x.shape), obs["agent_0"])
    observations["is_t0"] = jnp.array([[1], [0], [0], [1], [0], [0]], dtype=jnp.float32)
    resets = jnp.array([[False], [False], [False], [True], [False], [False]])
    done = jnp.array([[False], [False], [True], [False], [False], [True]])
    h0 = init(1)
    params = net.init(jax.random.PRNGKey(0), h0, (observations, resets))
    h, actions, old_logprobs, old_values = h0, [], [], []
    for t in range(6):
        o = jax.tree_util.tree_map(lambda x: x[t:t+1], observations)
        h, pi, value = net.apply(params, h, (o, resets[t:t+1]))
        action = pi.mode()
        actions.append(action); old_logprobs.append(pi.log_prob(action)); old_values.append(value)
    trajectory = trainer.Transition(done=done, reset=resets, action=jnp.concatenate(actions),
        value=jnp.concatenate(old_values), reward=jnp.zeros((6, 1)),
        log_prob=jnp.concatenate(old_logprobs), obs=observations, info={},
        pre_step_time=jnp.array([[0], [1], [2], [0], [1], [2]]))
    _, pi, value = net.apply(params, h0, (trajectory.obs, trajectory.reset))
    np.testing.assert_allclose(pi.log_prob(trajectory.action), trajectory.log_prob, atol=1e-6)
    np.testing.assert_allclose(value, trajectory.value, atol=1e-6)
    _, _, wrong_value = net.apply(params, h0, (trajectory.obs, trajectory.done))
    assert float(jnp.max(jnp.abs(wrong_value-trajectory.value))) > 1e-5


def test_ppo_update_and_evaluation_use_online_decisions_and_finite_losses(layout):
    config = dict(ALLOCATION_PROTOCOL=ALLOCATION_PROTOCOL, ENV_NAME="coordination_grid",
        ENV_KWARGS=dict(layout_path=layout, rounds_per_episode=2, max_steps=3),
        MODEL_TYPE="rnn", PARTNER_REGIME="single", SINGLE_PARTNER=[0, 2], INFLUENCE=True,
        NUM_ENVS=2, NUM_STEPS=8, TOTAL_TIMESTEPS=16, NUM_MINIBATCHES=2,
        UPDATE_EPOCHS=1, N_EPS_TOTAL=8, SCHEDULE_SEED=0, SEED=0,
        LR=5e-4, ANNEAL_LR=False, MAX_GRAD_NORM=0.25, GAMMA=0.99, GAE_LAMBDA=0.95,
        CLIP_EPS=0.2, VF_COEF=1., ENT_COEF=0.01, GRU_HIDDEN_DIM=128,
        FC_DIM_SIZE=128, ACTIVATION="relu")
    import wandb
    with wandb.init(mode="disabled"):
        out = jax.jit(trainer.make_train(config))(jax.random.PRNGKey(0))
        for name in ("actor_loss", "value_loss", "entropy", "entropy_t0", "entropy_tge1"):
            assert np.isfinite(np.asarray(out["metrics"][name])).all(), name
        assert float(out["metrics"]["entropy_t0"][0]) == 0
    params = out["runner_state"][0].params
    # A directory with a single layout is the evaluation corpus here.
    config["ENV_KWARGS"].pop("layout_path")
    record, env = evaluator.rollout_condition(params, config, [[0, 2]],
        n_episodes_per_capability=2, layouts_dir=str(Path(layout).parent),
        seed=0, save_hidden=True)
    summary = evaluator.summarize(record, env, 3, .01, 1.)
    assert summary["n_allocation_decisions"] == 8  # 2 episodes x 2 rounds x 2 decisions
    assert summary["allocation_protocol"] == ALLOCATION_PROTOCOL
    assert "fraction_optimal_allocation_overall" not in summary
    assert np.all(record["ego_alloc_action"] > 0)


def test_summary_distinguishes_requested_and_realized_assignments_and_masks_padding():
    # The default is RED; the ego asks for BLUE, but this is a no-influence trace.
    record = dict(dones=np.array([[False, False, True, True]]),
        is_t0=np.array([[True, False, False, False]]),
        round_done=np.array([[False, False, True, True]]),
        success=np.array([[False, False, True, True]]),
        round_idx=np.zeros((1, 4), dtype=int), round_time=np.array([[1, 2, 3, 4]]),
        rewards=np.array([[-.01, -.01, 1., 100.]]),
        capability=np.broadcast_to([0, 2], (1, 4, 2)),
        partner_assignment=np.array([[1, 1, 1, 2]]),
        ego_alloc_action=np.array([[1, 2, 2, 1]]),
        assignment_changed=np.array([[False, False, False, True]]))
    summary = evaluator.summarize(record, None, 3, .01, 1.)
    assert summary["n_allocation_decisions"] == 2
    assert summary["fraction_partner_assigned_faster"] == 0
    assert summary["fraction_ego_requested_faster_partner_goal"] == 1
    assert summary["assignment_switch_count"] == 0
    assert summary["mean_ep_return"] == pytest.approx(.98)
