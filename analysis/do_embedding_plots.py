import json
import os
from typing import Dict, Optional, List, Tuple

import jax
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

from utils import (
    load_rollout,
    save_fig,
    embed_hidden_states,
    relabel_speeds,
    dotdict,
)

sns.set_theme(style="dark")

METHOD = "UMAP"  # pca, tsne or UMAP


def make_embedding_df(
    hidden_states: np.ndarray,
    speeds: np.ndarray,
    layouts: Optional[List] = None,
    method_kwargs: Dict = {},
) -> pd.DataFrame:
    embeddings_2d, _ = embed_hidden_states(hidden_states, METHOD, method_kwargs)
    speeds = relabel_speeds(speeds)
    embeddings_2d = np.array(embeddings_2d)

    if layouts is None:
        layouts = [None] * len(hidden_states)

    return pd.DataFrame(
        {
            "dim_1": embeddings_2d[:, 0],
            "dim_2": embeddings_2d[:, 1],
            "task_1_speed": speeds[:, 0],
            "task_2_speed": speeds[:, 1],
            "speed_diff": speeds[:, 0] - speeds[:, 1],
            "best_task": speeds.argmax(axis=1),
            "layout": layouts,
        }
    )


def do_layout_plot(df: pd.DataFrame) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="dim_1", y="dim_2", hue="layout", alpha=0.8, s=60, ax=ax)
    ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[], xlabel="", ylabel="")
    ax.set_title(f"{METHOD} embeddings coloured by layout")
    sns.despine(fig, ax, bottom=True, left=True)
    return fig, ax


def do_speed_plots(df: pd.DataFrame) -> Tuple[Figure, Axes]:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    layouts = sorted(df["layout"].unique())
    for i in range(2):
        cmap = sns.color_palette("crest", as_cmap=True)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize())
        sns.scatterplot(
            data=df,
            x="dim_1",
            y="dim_2",
            hue=f"task_{i+1}_speed",
            style="layout",
            style_order=layouts,
            alpha=0.8,
            s=60,
            ax=axs[i],
            palette=cmap,
            hue_norm=sm.norm,
            legend=False,
        )
        axs[i].set(
            xticks=[], yticks=[], xticklabels=[], yticklabels=[], xlabel="", ylabel=""
        )
        axs[i].set_title(f"{METHOD} embeddings coloured by task {i + 1} speed")
        sns.despine(fig, axs[i], bottom=True, left=True)
        if i == 1:
            plt.colorbar(sm, ax=axs[i])
    return fig, axs


def do_speed_diff_plot(df: pd.DataFrame) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    cmap = sns.color_palette("crest", as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize())
    layouts = sorted(df["layout"].unique())
    sns.scatterplot(
        data=df,
        x="dim_1",
        y="dim_2",
        hue="speed_diff",
        style="layout",
        style_order=layouts,
        alpha=0.8,
        s=60,
        ax=ax,
        palette=cmap,
        hue_norm=sm.norm,
        legend=False,
    )
    ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[], xlabel="", ylabel="")
    ax.set_title(f"{METHOD} embeddings coloured by task speed difference")
    sns.despine(fig, ax, bottom=True, left=True)
    plt.colorbar(sm, ax=ax)
    return fig, ax


def do_layout_specific_plots(rollout_dict):
    unique_layouts = sorted(list(rollout_dict.keys())) + ["random"]
    cmap = sns.color_palette("crest", as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize())
    fig, axs = plt.subplots(3, len(unique_layouts), figsize=(12, 6))
    hue_vars = ["task_1_speed", "task_2_speed", "speed_diff"]
    hue_labels = ["Task 1 speed", "Task 2 speed", "Speed difference"]
    for i, layout in enumerate(unique_layouts):
        if layout == "random":
            hs = np.random.normal(size=(460, 128))
            labels = np.concatenate(
                [
                    r.agent1_speed_seq[0].reshape(1, -1)
                    for r in rollout_dict["cramped_room"]
                ]
            )
        else:
            hs = np.concatenate(
                [
                    r.hidden_state_0_seq[-50:].mean(axis=0).reshape(1, -1)
                    for r in rollout_dict[layout]
                ]
            )
            labels = np.concatenate(
                [r.agent1_speed_seq[0].reshape(1, -1) for r in rollout_dict[layout]]
            )
        df = make_embedding_df(
            hs,
            labels,
            method_kwargs={"min_dist": 1.0, "n_neighbors": hs.shape[0] - 1},
        )
        for j in (0, 1, 2):
            sns.scatterplot(
                data=df,
                x="dim_1",
                y="dim_2",
                hue=hue_vars[j],
                alpha=0.8,
                s=35,
                linewidth=0,
                ax=axs[j, i],
                palette=cmap,
                hue_norm=sm.norm,
                legend=False,
            )
            axs[j, i].set(
                xticks=[],
                yticks=[],
                xticklabels=[],
                yticklabels=[],
                xlabel="",
                ylabel="",
            )
            if j == 0:
                axs[j, i].set_title(layout)
            if i == 0:
                axs[j, i].set_ylabel(f"hue = {hue_labels[j]}")
            sns.despine(ax=axs[j, i], bottom=True, left=True)
    fig.suptitle("UMAP projections of RNN hidden states")
    # fig.colorbar(sm, ax=axs, location="right")
    return fig, axs


with open("analysis/metadata.json", "r") as f:
    metadata_dict = json.load(f)
with open("analysis/influence_rollouts.pkl", "rb") as f:
    rollout_dict = pickle.load(f)
for layout in rollout_dict:
    rollout_dict[layout] = [dotdict(r) for r in rollout_dict[layout]]

fig, _ = do_layout_specific_plots(rollout_dict)
save_fig(fig, "embeddings/embeddings-layout-specific")
