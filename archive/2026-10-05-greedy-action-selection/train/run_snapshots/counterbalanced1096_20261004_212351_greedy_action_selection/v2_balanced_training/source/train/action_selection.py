"""Policy action selection shared by training and checkpoint evaluation."""
import jax
import jax.numpy as jnp

GREEDY_RANDOM_TIES = "greedy_random_ties"
CATEGORICAL = "categorical"
ACTION_SELECTION_MODES = (GREEDY_RANDOM_TIES, CATEGORICAL)


def select_action(distribution, key, mode=CATEGORICAL):
    """Select the exact probability maximum; randomize only equal maxima.

    Equality is exact in the policy's probability dtype, with no tolerance.
    Unique maxima are independent of ``key``. Legacy checkpoints without an
    ACTION_SELECTION setting retain their original categorical behavior.
    """
    if mode == CATEGORICAL:
        return distribution.sample(seed=key)
    if mode != GREEDY_RANDOM_TIES:
        raise ValueError(f"Unknown action selection mode: {mode}")
    probabilities = distribution.probs
    maxima = probabilities == probabilities.max(axis=-1, keepdims=True)
    unique_choice = jnp.argmax(probabilities, axis=-1)

    def break_ties(_):
        # Uniform over exact maxima, assigning zero probability to others.
        logits = jnp.where(maxima, 0.0, -jnp.inf)
        return jax.random.categorical(key, logits, axis=-1)

    return jax.lax.cond(jnp.any(maxima.sum(axis=-1) > 1), break_ties,
                        lambda _: unique_choice, operand=None)
