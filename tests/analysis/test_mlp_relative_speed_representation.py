"""The MLP probe must see actor features, never its interface-only carry."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import mlp_relative_speed_representation as analysis


def test_capture_reads_post_relu_actor_layer_and_preserves_sampled_actions():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    nn = pytest.importorskip("flax.linen")
    distrax = pytest.importorskip("distrax")

    class MLP(nn.Module):
        @nn.compact
        def __call__(self, hidden, inputs):
            obs, _ = inputs
            trunk = nn.relu(nn.Dense(128)(obs))
            actor = nn.relu(nn.Dense(128)(trunk))
            logits = nn.Dense(3)(actor)
            logits = logits.at[..., 2].set(-jnp.inf)
            return hidden, distrax.Categorical(logits=logits), jnp.zeros(obs.shape[:2])

    network = MLP()
    obs = jnp.ones((1, 2, 4))
    dummy = jnp.full((2, 128), 999.)
    inputs = (obs, jnp.zeros((1, 2), dtype=bool))
    params = network.init(jax.random.PRNGKey(42), dummy, inputs)
    # Distinct known trunk and actor values detect selecting the wrong layer,
    # retaining its pre-ReLU values, or exporting the dummy carry.
    params["params"]["Dense_0"]["kernel"] = jnp.zeros((4, 128))
    params["params"]["Dense_0"]["bias"] = jnp.ones(128)
    params["params"]["Dense_1"]["kernel"] = jnp.eye(128)
    params["params"]["Dense_1"]["bias"] = jnp.linspace(-3., 1., 128)
    audit = []
    captured = analysis.capture_network(network, audit)
    activation, pi, value = captured.apply(params, dummy, inputs)
    expected = np.maximum(1 + np.linspace(-3., 1., 128), 0)
    np.testing.assert_allclose(activation, np.broadcast_to(expected, (2, 128)), atol=3e-7)
    assert np.any(activation == 0) and np.any(activation > 0)
    _, plain_pi, plain_value = network.apply(params, dummy, inputs)
    np.testing.assert_array_equal(pi.logits, plain_pi.logits)
    np.testing.assert_array_equal(value, plain_value)
    key = jax.random.PRNGKey(55)
    np.testing.assert_array_equal(pi.sample(seed=key), plain_pi.sample(seed=key))
    compiled = jax.jit(captured.apply)(params, dummy, inputs)
    np.testing.assert_allclose(compiled[0], activation, atol=3e-7)
    assert audit[0]["logits_reconstructed_from_final_layer"]
