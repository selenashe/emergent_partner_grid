"""V1_matched replays collection memory and submits a complete resumable batch."""

import ast
import functools
from pathlib import Path
import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.v1_matched_protocol import TRAINER_FILE, validate_recurrent_replay


ROOT = Path(__file__).resolve().parents[2]
def source(version):
    from train.experiment import TRAINING_MANIFEST
    import json
    manifest = json.loads(TRAINING_MANIFEST.read_text())
    variant = 'v1_matched_influence' if version == 'v1_matched' else version
    return (Path(manifest['source_variants'][variant]['frozen_source_root']) / TRAINER_FILE).read_text()


def test_replay_matches_collection_across_episode_and_rollout_boundaries():
    # Run the actual frozen GRU, isolated from the environment's heavy imports.
    tree = ast.parse(source('v1_matched'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ScannedRNN')
    scope = dict(nn=nn, jax=jax, jnp=jnp, functools=functools)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<frozen ScannedRNN>', 'exec'), scope)
    network = scope['ScannedRNN']()
    inputs = jax.random.normal(jax.random.PRNGKey(7), (7, 2, 8))
    initial = jnp.ones((2, 8)) * .4  # memory carried into this rollout chunk
    post_done = jnp.asarray([[0, 0], [1, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 1]], bool)
    # Slot 1 ended its previous episode just before the chunk started.
    prior_done = jnp.asarray([[0, 1]], bool)
    resets = jnp.concatenate((prior_done, post_done[:-1]), axis=0)
    params = network.init(jax.random.PRNGKey(9), initial, (inputs, resets))

    def collect(parameters):
        carry, outputs = initial, []
        for index in range(len(inputs)):
            carry, output = network.apply(parameters, carry,
                                          (inputs[index:index + 1], resets[index:index + 1]))
            outputs.append(output)
        return jnp.concatenate(outputs, axis=0)

    # Read the replay input from the transformed PPO call rather than inventing
    # a test-only implementation of its selection between done and reset.
    replay_call = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and node.func.attr == 'apply' and len(node.args) == 3
                       and isinstance(node.args[2], ast.Tuple)
                       and isinstance(node.args[2].elts[0], ast.Attribute)
                       and node.args[2].elts[0].attr == 'obs')
    assert replay_call.args[2].elts[1].attr == 'reset'
    transition_call = next(node for node in ast.walk(tree)
                           if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                           and node.func.id == 'Transition')
    assert next(k.value.id for k in transition_call.keywords if k.arg == 'reset') == 'last_done'
    assert next(k.value.id for k in transition_call.keywords if k.arg == 'done') == 'done_all'
    replay = network.apply(params, initial, (inputs, resets))[1]
    np.testing.assert_allclose(replay, collect(params), atol=1e-6)
    old_replay = network.apply(params, initial, (inputs, post_done))[1]
    assert not np.allclose(old_replay, collect(params), atol=1e-6)
    # The corrected replay also reproduces gradients of the collected sequence.
    collection_grad = jax.grad(lambda p: jnp.square(collect(p)).sum())(params)
    replay_grad = jax.grad(lambda p: jnp.square(network.apply(p, initial, (inputs, resets))[1]).sum())(params)
    for actual, expected in zip(jax.tree_util.tree_leaves(replay_grad),
                                jax.tree_util.tree_leaves(collection_grad)):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_matched_replay_and_gae_use_separate_flags():
    text = source('v1_matched')
    assert validate_recurrent_replay(text)
    assert validate_recurrent_replay(source('v2'))
    gae = next(n for n in ast.walk(ast.parse(text))
               if isinstance(n, ast.FunctionDef) and n.name == '_calculate_gae')
    fields = {n.attr for n in ast.walk(gae) if isinstance(n, ast.Attribute)}
    assert 'done' in fields and 'reset' not in fields
    with pytest.raises(ValueError, match='replay'):
        validate_recurrent_replay(text.replace('(traj_batch.obs, traj_batch.reset),',
                                              '(traj_batch.obs, traj_batch.done),'))


