"""Greedy collection must never explore a lower-probability action."""
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from train.action_selection import select_action, GREEDY_RANDOM_TIES


def choices(probabilities, n=1024):
    keys = jax.random.split(jax.random.PRNGKey(2026), n)
    policy = distrax.Categorical(probs=jnp.asarray(probabilities))
    return np.asarray(jax.jit(jax.vmap(
        lambda key: select_action(policy, key, GREEDY_RANDOM_TIES)))(keys))


def test_unique_maxima_do_not_depend_on_seed_or_choose_less_likely_actions():
    actions = choices([[[.7, .3, 0], [.1, .8, .1], [0, 0, 1]]])
    np.testing.assert_array_equal(actions, np.broadcast_to([[0, 1, 2]], actions.shape))


def test_exact_ties_are_reproducible_uniform_and_exclude_other_actions():
    actions = choices([.45, .1, .45, 0], n=4096)
    assert set(actions) == {0, 2}
    assert .46 < np.mean(actions == 0) < .54
    np.testing.assert_array_equal(actions, choices([.45, .1, .45, 0], n=4096))


def test_nearly_equal_probabilities_are_not_treated_as_ties():
    actions = choices([.50000006, .49999994])
    assert np.all(actions == 0)


def test_mixed_tied_and_unique_batch_preserves_unique_choices():
    actions = choices([[.5, .5, 0], [.1, .7, .2]])
    assert set(actions[:, 0]) == {0, 1}
    assert np.all(actions[:, 1] == 1)


def test_legacy_checkpoint_action_sampling_is_preserved():
    policy = distrax.Categorical(probs=jnp.array([.7, .3]))
    key = jax.random.PRNGKey(11)
    assert int(select_action(policy, key)) == int(policy.sample(seed=key))


def test_unknown_mode_fails_before_training():
    with pytest.raises(ValueError, match="Unknown action selection"):
        select_action(distrax.Categorical(probs=jnp.array([.7, .3])),
                      jax.random.PRNGKey(0), "invalid")
