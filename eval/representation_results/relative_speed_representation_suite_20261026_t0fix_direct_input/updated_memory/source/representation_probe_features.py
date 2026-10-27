"""Frozen policy feature extraction with explicit episode-relative timing."""
from pathlib import Path

import numpy as np


def checkpoint_heads(path):
    from safetensors import safe_open
    with safe_open(str(Path(path)), framework="numpy") as handle:
        return {name: handle.get_tensor("params," + name) for layer in range(4)
                for name in (f"Dense_{layer},kernel", f"Dense_{layer},bias")}


def frozen_heads(hidden, weights):
    actor = np.maximum(hidden @ weights["Dense_0,kernel"] + weights["Dense_0,bias"], 0)
    critic = np.maximum(hidden @ weights["Dense_2,kernel"] + weights["Dense_2,bias"], 0)
    logits = actor @ weights["Dense_1,kernel"] + weights["Dense_1,bias"]
    value = critic @ weights["Dense_3,kernel"] + weights["Dense_3,bias"]
    return actor, critic, logits, value


def masked_probabilities(logits, legal):
    legal = np.asarray(legal, dtype=bool)
    if logits.shape != legal.shape or not legal.any(axis=-1).all():
        raise ValueError("Every observation needs a nonempty action mask")
    masked = np.where(legal, logits, -np.inf)
    shifted = masked - masked.max(axis=-1, keepdims=True)
    probability = np.exp(shifted)
    return probability / probability.sum(axis=-1, keepdims=True)


def representation_features(kind, hidden, observation, legal, lengths, steps, weights=None):
    """Return [time,episode,feature], holding each final valid feature forward.

    Previous memory is indexed BEFORE terminal holding: an episode ending at
    index L-1 retains h_(L-2) for that representation, rather than h_(L-1).
    Intermediate round boundaries never reset previous memory.
    """
    lengths = np.asarray(lengths, dtype=int)
    steps = np.asarray(steps, dtype=int)
    if np.any(lengths < 1) or np.any(steps < 0):
        raise ValueError("Invalid episode lengths or observation indices")
    index = np.minimum(steps[None, :], lengths[:, None] - 1)
    rows = np.arange(len(lengths))[:, None]
    if kind == "current_observation":
        values = observation[rows, index]
    elif kind == "previous_memory":
        previous = np.maximum(index - 1, 0)
        values = hidden[rows, previous].copy()
        values[index == 0] = 0
    else:
        state = hidden[rows, index]
        if kind == "updated_memory":
            values = state
        else:
            if weights is None:
                raise ValueError("Policy head weights are required")
            actor, critic, logits, value = frozen_heads(state, weights)
            if kind == "actor_features": values = actor
            elif kind == "critic_features": values = critic
            elif kind == "action_logits": values = logits
            elif kind == "value_estimate": values = value
            elif kind == "action_probabilities": values = masked_probabilities(logits, legal[rows, index])
            else: raise ValueError("Unknown representation: " + kind)
    return np.ascontiguousarray(values.transpose(1,0,2), dtype=np.float32)
