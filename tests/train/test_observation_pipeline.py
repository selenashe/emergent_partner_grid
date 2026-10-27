"""Phase information, direct-input memory, and historical checkpoint fidelity."""
import ast
import functools
from pathlib import Path
from typing import Any, Callable, Dict

import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from train.observation_pipeline import upgrade_trainer, limit_training

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "v1_matched": "v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v1_matched_influence_t0fix_direct_input",
    "v2": "v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v2_t0fix_direct_input",
    "v3": "v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v3_t0fix_direct_input",
}


def frozen(version):
    return (ROOT / "train/run_snapshots" / SOURCES[version] /
            "source/baselines/IPPO/ippo_rnn_coordination_grid.py").read_text()


def modules(source):
    # Execute the actual network definitions without importing an unrelated env.
    names = {"ScannedRNN", "CNN", "CommObsEncoder", "DirectObsEncoder",
             "encode_observation", "ActorCriticCommRNN", "ActorCriticCommMLP", "build_network"}
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in names]
    mask0 = jnp.asarray([i in (13, 14) for i in range(15)])
    mask1 = jnp.asarray([i % 3 != 0 for i in range(15)])
    scope = dict(nn=nn, jax=jax, jnp=jnp, np=np, functools=functools, distrax=distrax,
                 Any=Any, Callable=Callable, Dict=Dict, orthogonal=orthogonal, constant=constant,
                 _ACTION_MASK_T0_J=mask0, _ACTION_MASK_TGEQ1_J=~mask1)
    def ego_action_mask(obs):
        return jnp.where((obs["is_t0"] > .5)[..., None], mask0, mask1)
    scope["ego_action_mask"] = ego_action_mask
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<protocol networks>", "exec"), scope)
    return scope


def observation():
    # All features match except the two phase flags, reproducing the v2 ambiguity.
    grid = jnp.zeros((1, 2, 7, 7, 5)).at[:, :, 1, 1, 0].set(1)
    return dict(grid=grid, last_allocation=jnp.tile(jnp.array([0., 1., 0.]), (1, 2, 1)),
                is_t0=jnp.array([[1., 0.]]))


def config(model="mlp", encoder="cnn", phase=True):
    return dict(MODEL_TYPE=model, GRID_ENCODER=encoder, INCLUDE_IS_T0=phase,
                GRU_HIDDEN_DIM=8, FC_DIM_SIZE=8, GRID_EMB_DIM=4, MSG_EMB_DIM=3, ACTIVATION="relu")


@pytest.mark.parametrize("version", SOURCES)
@pytest.mark.parametrize("encoder", ["cnn", "flat"])
@pytest.mark.parametrize("model", ["mlp", "rnn"])
def test_value_can_distinguish_phases_in_every_protocol(version, encoder, model):
    scope = modules(upgrade_trainer(frozen(version)))
    network, initialize, _ = scope["build_network"](config(model, encoder), 15)
    obs, memory = observation(), initialize(2)
    reset = jnp.zeros((1, 2), bool)
    params = network.init(jax.random.PRNGKey(31), memory, (obs, reset))
    carry, _, values = network.apply(params, memory, (obs, reset))
    assert carry.shape == (2, 8)
    assert np.isfinite(np.asarray(values)).all()
    assert abs(float(values[0, 0] - values[0, 1])) > 1e-7
    gradient = jax.grad(lambda phase: network.apply(params, memory,
        (dict(obs, is_t0=phase), reset))[2].sum())(obs["is_t0"])
    assert np.any(np.abs(np.asarray(gradient)) > 1e-7)


@pytest.mark.parametrize("version", SOURCES)
@pytest.mark.parametrize("model", ["mlp", "rnn"])
def test_legacy_weights_and_outputs_are_preserved(version, model):
    original, current = modules(frozen(version)), modules(upgrade_trainer(frozen(version)))
    cfg = config(model)
    del cfg["INCLUDE_IS_T0"]
    del cfg["GRID_ENCODER"]
    old, init, _ = original["build_network"](cfg, 15)
    new, _, _ = current["build_network"](cfg, 15)
    obs, reset, memory = observation(), jnp.zeros((1, 2), bool), init(2)
    params = old.init(jax.random.PRNGKey(11), memory, (obs, reset))
    expected = old.apply(params, memory, (obs, reset))
    actual = new.apply(params, memory, (obs, reset))
    for a, b in [(actual[0], expected[0]), (actual[1].logits, expected[1].logits),
                 (actual[2], expected[2])]:
        np.testing.assert_array_equal(a, b)
    if model == "mlp":
        np.testing.assert_array_equal(actual[2][:, 0], actual[2][:, 1])


def test_direct_grid_is_lossless_and_replay_resets_keep_hidden_width():
    scope = modules(upgrade_trainer(frozen("v2")))
    obs = {k: v[0] for k, v in observation().items()}
    encoder = scope["DirectObsEncoder"]()
    embedded = encoder.apply({}, obs)
    assert embedded.shape == (2, 249)
    np.testing.assert_array_equal(embedded[:, :245], np.asarray(obs["grid"]).reshape(2, -1))
    np.testing.assert_array_equal(embedded[:, 245:248], obs["last_allocation"])
    np.testing.assert_array_equal(embedded[:, -1], obs["is_t0"])
    rnn = scope["ScannedRNN"]()
    memory = jnp.ones((2, 8))
    sequence = jnp.tile(embedded[None], (3, 1, 1))
    resets = jnp.array([[0, 0], [1, 0], [0, 1]], bool)
    params = rnn.init(jax.random.PRNGKey(4), memory, (sequence, resets))
    carry, replay = rnn.apply(params, memory, (sequence, resets))
    collected = []
    for tick in range(3):
        memory, output = rnn.apply(params, memory, (sequence[tick:tick+1], resets[tick:tick+1]))
        collected.append(output)
    assert carry.shape == (2, 8)
    np.testing.assert_allclose(replay, jnp.concatenate(collected), atol=1e-6)


def test_upgrade_is_idempotent_and_only_run_length_changes_legacy_pilot():
    for version in SOURCES:
        original = frozen(version)
        upgraded = upgrade_trainer(original)
        assert upgrade_trainer(upgraded) == upgraded
        assert 'config.setdefault("INCLUDE_IS_T0", True)' in upgraded
        limited = limit_training(original)
        assert limit_training(limited) == limited
        # The LR function still reads the full NUM_UPDATES schedule.
        assert 'update_steps = config["NUM_UPDATES"]' in limited
        assert '_update_step, runner_state, None, config["RUN_UPDATES"]' in limited
        call = next(n for n in ast.walk(ast.parse(limited)) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "write_audit")
        cfg = dict(NUM_UPDATES=915, RUN_UPDATES=16)
        passed = eval(compile(ast.Expression(body=call.args[3]), "<audit config>", "eval"), {"config": cfg})
        assert passed["NUM_UPDATES"] == 16 and cfg["NUM_UPDATES"] == 915
