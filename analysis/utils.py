import os
from typing import Dict

import chex
import jax
import jax.numpy as jnp
import json
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pcax
import umap
import h5py

from tsnex import transform as tsne_transform


# Copied from: https://stackoverflow.com/a/23689767
class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def replace(self, **kwargs):
        return dotdict({**self, **kwargs})


def load_rollout(artifact_name):
    fp = os.path.join("artifacts", artifact_name)
    with open(fp) as f:
        raw_data = json.load(f)

    out = {}
    for k in [
        "agent1_mode_seq",
        "agent1_speed_seq",
        "hidden_state_0_seq",
        "rewards",
    ]:
        if k in raw_data:
            out[k] = np.array(raw_data[k][-400:])

    return out
    # return dotdict(out)


def load_rollouts_batch(artifact_name):
    rollouts = []
    f = h5py.File(os.path.join("artifacts", artifact_name), "r")
    for k1 in f.keys():
        r = {}
        for k2 in f[k1]:
            r[k2] = np.array(f[k1][k2])
        rollouts.append(dotdict(r))
    return rollouts


def save_fig(fig, filename, extensions=["png", "svg"]):
    fig.tight_layout()
    dir = os.path.join("analysis", "plots")
    if "/" in filename:
        dir = os.path.join(dir, *filename.split("/")[:-1])
        filename = filename.split("/")[-1]
    os.makedirs(dir, exist_ok=True)
    for ext in extensions:
        fig.savefig(
            os.path.join(dir, f"{filename}.{ext}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def do_regression(data, x, y, add_const=True):
    X = data[x]
    Y = data[y]
    if add_const:
        X = sm.add_constant(X)

    model = sm.OLS(Y, X)
    res = model.fit()

    conf_int = res.conf_int()

    intercept = res.params["const"] if add_const else 0
    intercept_conf_int = conf_int.loc["const"].values if add_const else [0, 0]

    return {
        "slope": res.params[x],
        "slope_conf_int": conf_int.loc[x].values,
        "intercept": intercept,
        "intercept_conf_int": intercept_conf_int,
        "r2": res.rsquared,
        "p_value": res.pvalues[x],
    }


def extract_sliding_windows(x: chex.Array, window_size: int):
    half = window_size // 2
    x_padded = jnp.pad(x, (half, half), mode="edge")
    get_window = lambda i: jax.lax.dynamic_slice(x_padded, (i,), (window_size,))
    return jax.vmap(get_window)(jnp.arange(x.shape[0]))


def estimate_throughput(
    rewards: chex.Array, window_size: int = 50, norm=False
) -> jax.Array:
    x_positions = jnp.arange(-window_size // 2, window_size // 2)  # (window_size,)

    # (T, window_size)
    windows = extract_sliding_windows(jnp.cumsum(rewards), window_size)

    # Mean center x_positions once
    x_mean = jnp.mean(x_positions)
    x_centered = x_positions - x_mean
    x_var = jnp.mean(x_centered**2)

    def compute_slope(window):
        y_mean = jnp.mean(window)
        y_centered = window - y_mean
        cov = jnp.mean(x_centered * y_centered)
        slope = jnp.where(x_var > 0, cov / x_var, 0.0)
        return slope

    slopes = jax.vmap(compute_slope)(windows)

    if norm:
        slopes = (slopes - jnp.min(slopes)) / (jnp.max(slopes) - jnp.min(slopes))

    slopes = (
        slopes.at[: window_size // 2].set(jnp.nan).at[-window_size // 2 :].set(jnp.nan)
    )
    return slopes


def get_time_to_first_reward(rewards: chex.Array) -> chex.Scalar:
    if rewards.max() == 0:
        return jnp.nan
    return int(jnp.where(rewards > 0)[0][0])


def embed_hidden_states(
    hidden_states: chex.Array, method: str, method_kwargs: Dict = {}
):
    if method == "pca":
        state = pcax.fit(hidden_states, n_components=2, **method_kwargs)
        return np.array(pcax.transform(state, hidden_states)), state
    elif method == "tsne":
        return (
            np.array(tsne_transform(hidden_states, perplexity=50, **method_kwargs)),
            None,
        )
    elif method == "UMAP":
        reducer = umap.UMAP(**method_kwargs, n_epochs=5000, random_state=42)
        transform = reducer.fit(hidden_states)
        return transform.embedding_, transform
    else:
        raise ValueError(f"Unsupported method: {method}")


def relabel_speeds(speeds):
    for i in [0, 1]:
        unique_speeds = sorted(np.unique(speeds[:, i]))
        mapping = {speed: j for j, speed in enumerate(unique_speeds)}
        speeds[:, i] = np.array([mapping[speed] for speed in speeds[:, i]])
    return speeds
