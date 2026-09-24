from collections import Counter
import functools
import json
import os

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_rollout, estimate_throughput, save_fig, embed_hidden_states

colors = ["xkcd:green blue", "xkcd:periwinkle", "xkcd:pumpkin"]


class LinearProbe(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.num_classes)(x)
        return x


@functools.partial(jax.jit, static_argnums=(3,))
def train_step(ts, embeddings, labels, n_cats):
    def loss_fn(params):
        logits = ts.apply_fn(params, embeddings)
        loss = jnp.mean(
            optax.softmax_cross_entropy(logits=logits, labels=jnp.eye(n_cats)[labels])
        )
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(ts.params)
    return ts.apply_gradients(grads=grads), loss


def train_probe(rng, x, y, n_iterations=int(1e3), lr=1e-3):
    n_classes = y.max() + 1
    model = LinearProbe(num_classes=n_classes)
    params = model.init(rng, jnp.empty([1, x.shape[1]]))
    tx = optax.adam(learning_rate=lr)
    ts = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    for itr in range(n_iterations):
        rng, _ = jax.random.split(rng)
        # x_train, y_train = shuffle(rng, x, y)
        ts, loss = train_step(ts, x, y, n_classes)

    return ts, loss


def estimate_throughput_simple(rewards, window_size=50):
    return np.convolve(rewards, np.ones(window_size), "same") / window_size


def make_throughput_df(rollout_dict, metadata_dict):
    data = {
        "model": [],
        "layout": [],
        "timestep": [],
        "throughput": [],
        "old_speed_1": [],
        "old_speed_2": [],
        "new_speed_1": [],
        "new_speed_2": [],
    }
    for an, rollout in rollout_dict.items():
        if metadata_dict[an]["run_type"] == "basic":
            continue

        speeds = metadata_dict[an]["speeds"]
        if speeds[0] == speeds[2] and speeds[1] == speeds[3]:
            continue

        tp = np.array(estimate_throughput(rollout.rewards, window_size=50, norm=True))
        data["model"].extend([metadata_dict[an]["model_name"]] * len(tp))
        data["layout"].extend([metadata_dict[an]["layout"]] * len(tp))
        data["timestep"].extend(np.arange(len(tp)))
        data["throughput"].extend(tp)
        data["old_speed_1"].extend([speeds[0]] * len(tp))
        data["old_speed_2"].extend([speeds[1]] * len(tp))
        data["new_speed_1"].extend([speeds[2]] * len(tp))
        data["new_speed_2"].extend([speeds[3]] * len(tp))

    return pd.DataFrame(data)


def make_embeddings_df(rollout_dict, metadata_dict, layout):
    base_rollouts, switch_rollouts, speeds = [], [], []
    for an in rollout_dict:
        if metadata_dict[an]["layout"] == layout:
            speeds_ = metadata_dict[an]["speeds"]
            if speeds_[0] == speeds_[2] and speeds_[1] == speeds_[3]:
                base_rollouts.append(rollout_dict[an])
            else:
                switch_rollouts.append(rollout_dict[an])
                speeds.append(speeds_)
    speeds = np.array(speeds)

    base_hs = np.concatenate(
        [r.hidden_state_0_seq.mean(axis=0).reshape(1, -1) for r in base_rollouts]
    )
    base_labels = np.concatenate(
        [r.agent1_speed_seq[0].reshape(1, -1) for r in base_rollouts]
    )

    switch_h1s = np.concatenate(
        [
            r.hidden_state_0_seq[100:300].mean(axis=0).reshape(1, -1)
            for r in switch_rollouts
        ]
    )
    switch_h2s = np.concatenate(
        [
            r.hidden_state_0_seq[400:].mean(axis=0).reshape(1, -1)
            for r in switch_rollouts
        ]
    )

    base_embeddings, transform = embed_hidden_states(
        base_hs, "UMAP", {"n_neighbors": base_hs.shape[0] - 1, "min_dist": 1.0}
    )
    embed_1s = transform.transform(switch_h1s)
    embed_2s = transform.transform(switch_h2s)

    return (
        pd.DataFrame(
            {
                "z_1_0": embed_1s[:, 0],
                "z_1_1": embed_1s[:, 1],
                "z_2_0": embed_2s[:, 0],
                "z_2_1": embed_2s[:, 1],
                "old_speed_1": speeds[:, 0],
                "old_speed_2": speeds[:, 1],
                "new_speed_1": speeds[:, 2],
                "new_speed_2": speeds[:, 3],
            }
        ),
        base_embeddings,
        base_labels,
    )


def make_task_proportion_df(rollout_dict, metadata_dict):
    data = []
    for an, rollout in rollout_dict.items():
        speeds = metadata_dict[an]["speeds"]
        if speeds[0] == speeds[2] and speeds[1] == speeds[3]:
            continue

        old_0 = (rollout.agent1_mode_seq[100:300] == 0).sum()
        old_1 = (rollout.agent1_mode_seq[100:300] == 1).sum()
        new_0 = (rollout.agent1_mode_seq[400:] == 0).sum()
        new_1 = (rollout.agent1_mode_seq[400:] == 1).sum()
        old_0, old_1 = old_0 / (old_0 + old_1), old_1 / (old_0 + old_1)
        new_0, new_1 = new_0 / (new_0 + new_1), new_1 / (new_0 + new_1)

        data.append(
            {
                "layout": metadata_dict[an]["layout"],
                "old_prop_0": old_0,
                "old_prop_1": old_1,
                "new_prop_0": new_0,
                "new_prop_1": new_1,
                "old_speed_1": speeds[0],
                "old_speed_2": speeds[1],
                "new_speed_1": speeds[2],
                "new_speed_2": speeds[3],
            }
        )

    return pd.DataFrame(data)


def plot_all_combos_for_layout(df, layout):
    speed_combos = [(0, 6), (6, 0), (1, 4), (4, 1)]
    df_ = df[df["layout"] == layout]
    fig, axs = plt.subplots(
        len(speed_combos), 2, figsize=(8, 16), sharex=True, sharey=True
    )
    for i, old in enumerate(speed_combos):
        for j, new in enumerate(speed_combos[:2]):
            sns.lineplot(
                df_[
                    (df_["old_speed_1"] == old[0])
                    & (df_["old_speed_2"] == old[1])
                    & (df_["new_speed_1"] == new[0])
                    & (df_["new_speed_2"] == new[1])
                ],
                x="timestep",
                y="throughput",
                ax=axs[i, j],
            )
            # add a vertical dashed red line at x = 300
            axs[i, j].axvline(x=300, color="red", linestyle="--")
            axs[i, j].set_title(f"{old} -> {new}")
            axs[i, j].set(
                xlabel="",
                ylabel="",
                xticks=[],
                yticks=[],
                xticklabels=[],
                yticklabels=[],
            )
            sns.despine(ax=axs[i, j], bottom=True, left=True)
    fig.suptitle(layout)
    return fig, axs


def plot_combo_for_all_layouts(df, combo):
    old, new = combo
    df_ = df[
        (df["old_speed_1"] == old[0])
        & (df["old_speed_2"] == old[1])
        & (df["new_speed_1"] == new[0])
        & (df["new_speed_2"] == new[1])
    ]

    layouts = sorted(list(df["layout"].unique())) + ["All"]
    fig, axs = plt.subplots(1, (len(layouts)), figsize=(16, 3.5))

    for i, layout in enumerate(layouts):
        if layout == "All":
            _df_ = df_
        else:
            _df_ = df_[df_["layout"] == layout]

        pre_df, post_df = _df_[_df_["timestep"] < 300], _df_[_df_["timestep"] >= 300]

        sns.lineplot(
            data=pre_df,
            x="timestep",
            y="throughput",
            ax=axs[i],
            linewidth=3,
            color=colors[0],
        )
        sns.lineplot(
            data=post_df,
            x="timestep",
            y="throughput",
            ax=axs[i],
            linewidth=3,
            color=colors[1],
        )
        # add a vertical dashed red line at x = 300
        axs[i].axvline(x=300, color="red", linestyle="--", linewidth=3)
        axs[i].set(
            xlabel="Timestep", ylabel="Throughput" if i == 0 else "", title=f"{layout}"
        )
        if i > 0:
            axs[i].set(yticks=[], yticklabels=[])
        sns.despine(ax=axs[i], left=i > 0)
    return fig, axs


def plot_embeddings(rollout_dict, metadata_dict, layouts, combo):
    old, new = combo
    sns.set_style("dark")
    fig, axs = plt.subplots(1, len(layouts), figsize=(10, 3.5))
    for i, layout in enumerate(layouts):
        df, base_embeddings, base_labels = make_embeddings_df(
            rollout_dict, metadata_dict, layout
        )
        df = df[
            (df["old_speed_1"] == old[0])
            & (df["old_speed_2"] == old[1])
            & (df["new_speed_1"] == new[0])
            & (df["new_speed_2"] == new[1])
        ]

        # first plot base embedding distributions
        scalar_labels = 1 - (base_labels[:, 0] == old[0]).astype(int)
        sns.kdeplot(
            x=base_embeddings[:, 0],
            y=base_embeddings[:, 1],
            hue=scalar_labels,
            palette=colors,
            alpha=0.3,
            fill=True,
            levels=10,
            ax=axs[i],
            legend=False,
        )

        # then plot new embeddings
        sns.scatterplot(
            df,
            x="z_1_0",
            y="z_1_1",
            c=colors[0],
            alpha=0.8,
            s=100,
            ax=axs[i],
        )
        sns.scatterplot(
            df,
            x="z_2_0",
            y="z_2_1",
            c=colors[1],
            alpha=0.8,
            s=100,
            ax=axs[i],
        )
        axs[i].set(
            xlabel="",
            ylabel="",
            xticks=[],
            yticks=[],
            xticklabels=[],
            yticklabels=[],
        )

        sns.despine(ax=axs[i], bottom=True, left=True)
    return fig, axs


with open("analysis/metadata.json", "r") as f:
    metadata_dict = json.load(f)

artifact_names = [
    dir_name
    for dir_name in os.listdir("artifacts")
    if "rollout" in dir_name
    and metadata_dict[dir_name]["run_type"] in ["changes-switch2"]
    and metadata_dict[dir_name]["model_name"] == "rnn"
]

rollout_dict, layouts = {}, set()
for an in tqdm(artifact_names, desc="loading rollouts"):
    try:
        rollout_dict[an] = load_rollout(an)
        tmp = an.split("overcooked_v2_")[1]
        speeds = [int(tmp[0]), int(tmp[1]), int(tmp[4]), int(tmp[5])]
        metadata_dict[an] = {**metadata_dict[an], "speeds": speeds}
        layouts.add(metadata_dict[an]["layout"])
    except Exception as e:
        print(f"Error loading rollout {an}: {e}")

df = make_throughput_df(rollout_dict, metadata_dict)

fig, _ = plot_combo_for_all_layouts(df, [(6, 0), (0, 6)])
fig.tight_layout()
plt.show()
save_fig(fig, "switching/6006")

layouts = sorted(list(set(v["layout"] for v in metadata_dict.values())))
layouts = [l for l in layouts if "cramped_room" in l]

fig, axs = plot_embeddings(rollout_dict, metadata_dict, layouts, [(6, 0), (0, 6)])
save_fig(fig, "switching/embeddings-6006")

df = make_task_proportion_df(rollout_dict, metadata_dict)
old, new = [(6, 0), (0, 6)]
df = df[
    (df["old_speed_1"] == old[0])
    & (df["old_speed_2"] == old[1])
    & (df["new_speed_1"] == new[0])
    & (df["new_speed_2"] == new[1])
]

layouts += ["All"]
fig, axs = plt.subplots(1, len(layouts), figsize=(16, 2.5))
for i, layout in enumerate(layouts):
    if layout == "All":
        df_ = df
    else:
        df_ = df[df["layout"] == layout]

    old_prop_0 = df_["old_prop_0"].mean()
    old_prop_1 = df_["old_prop_1"].mean()
    new_prop_0 = df_["new_prop_0"].mean()
    new_prop_1 = df_["new_prop_1"].mean()

    axs[i].bar(
        ["Pre-switch", "Post-switch"],
        [old_prop_0, new_prop_0],
        color=["#06B48B", "#8E82FE"],
    )
    axs[i].bar(
        ["Pre-switch", "Post-switch"],
        [old_prop_1, new_prop_1],
        bottom=[old_prop_0, new_prop_0],
        color=["#53F8D1", "#BBB4FF"],
    )
    axs[i].set(
        ylabel="Time spent by partner on each subtask" if i == 0 else "",
        xlabel="",
        xticks=[],
        xticklabels=[],
    )
    if i > 0:
        axs[i].set(yticks=[], yticklabels=[])
    sns.despine(ax=axs[i], bottom=True, left=i > 0)

save_fig(fig, "switching/bars")
