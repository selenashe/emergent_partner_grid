from collections import Counter
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from tqdm import tqdm

from utils import load_rollout, save_fig, do_regression, estimate_throughput

scatter_kwargs = {"s": 40, "alpha": 0.1}
line_kwargs = {"linewidth": 4}
colors = ["xkcd:green blue", "xkcd:periwinkle", "xkcd:pumpkin"]


def make_total_reward_df(rollouts, model_names, layouts, speeds):
    data = []
    for rollout, model_name, layout, speed in zip(
        rollouts, model_names, layouts, speeds
    ):
        data.append(
            {
                "model": model_name,
                "reward": rollout.rewards.sum(),
                "layout": layout,
                "speeds": speed,
            }
        )
    return pd.DataFrame(data)


def make_throughput_df(rollouts, model_names, layouts):
    data = {"model": [], "layout": [], "timestep": [], "cr": [], "throughput": []}
    for rollout, model_name, layout in zip(rollouts, model_names, layouts):
        cr = np.cumsum(rollout.rewards)
        tp = np.array(estimate_throughput(rollout.rewards, window_size=50))
        data["model"].extend([model_name] * len(cr))
        data["layout"].extend([layout] * len(cr))
        data["timestep"].extend(np.arange(len(cr)))
        data["cr"].extend(cr)
        data["throughput"].extend(tp)
    return pd.DataFrame(data)


def make_task_df(rollouts, model_names, layouts):
    data = []
    for rollout, model_name, layout in zip(rollouts, model_names, layouts):
        prop_0 = (rollout.agent1_mode_seq[200:] == 0).sum()
        prop_1 = (rollout.agent1_mode_seq[200:] == 1).sum()
        prop_0, prop_1 = prop_0 / (prop_0 + prop_1), prop_1 / (prop_0 + prop_1)
        speed_0, speed_1 = rollout.agent1_speed_seq[0][::-1]
        speed_0, speed_1 = 1 / (speed_0 + 1), 1 / (speed_1 + 1)
        data.append(
            {
                "model": model_name,
                "layout": layout,
                "prop_0": prop_0,
                "prop_1": prop_1,
                "speed_0": speed_0,
                "speed_1": speed_1,
                "speed_diff": speed_0 - speed_1,
            }
        )
    return pd.DataFrame(data)


def pointplot(df, x, y, hue, hue_order, ax, kwargs={}):
    palette = sns.color_palette(colors)
    sns.pointplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        order=sorted(df[x].unique()),
        hue_order=hue_order,
        ax=ax,
        linestyle="none",
        dodge=False,
        markersize=10,
        palette=palette,
        **kwargs,
    )


def do_layout_and_partner_reward_plot(df: pd.DataFrame) -> Tuple[Figure, Axes]:
    unique_speeds = df["speeds"].unique()
    fig, axs = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(16, 8))
    for i, speed_str in enumerate(unique_speeds):
        sns.barplot(
            df[df["speeds"] == speed_str],
            x="model",
            y="reward",
            hue="layout",
            ax=axs[i // 4, i % 4],
            legend=i == 1,
        )
        axs[i // 4, i % 4].set_title(speed_str)
    return fig, axs


def do_layout_reward_plot(df: pd.DataFrame) -> Tuple[Figure, Axes]:
    layouts = df["layout"].unique()
    fig, axs = plt.subplots(1, len(layouts) + 1, figsize=(15, 3), sharey=False)
    y_min, y_max = df["reward"].min(), df["reward"].max()

    for i, layout in enumerate(layouts):
        pointplot(
            df[df["layout"] == layout],
            "model",
            "reward",
            None,
            None,
            axs[i],
        )
        axs[i].set(
            ylim=(y_min, y_max),
            xlabel="",
            xticks=[],
            xticklabels=[],
            ylabel="Average episode reward" if i == 0 else "",
        )
        axs[i].set_title(layout, fontstyle="italic")
        if i > 0:
            axs[i].set(yticks=[], yticklabels=[])
        sns.despine(ax=axs[i], left=i > 0, bottom=True)

    pointplot(df, "model", "reward", None, None, axs[-1])
    axs[-1].set(
        ylim=(y_min, y_max),
        yticks=[],
        yticklabels=[],
        ylabel="",
        xticks=[],
        xticklabels=[],
        xlabel="",
    )
    axs[-1].set_title("Overall", fontweight="bold")
    sns.despine(ax=axs[-1], left=True, bottom=True)

    return fig, axs


def do_overall_reward_plot(df: pd.DataFrame) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots()

    pointplot(df, "model", "reward", None, None, ax)

    ax.set_title("Average episode reward")
    ax.set_xlabel("Model")
    ax.set_ylabel("Reward")

    sns.despine(fig, ax)
    return fig, ax


def do_throughput_plot(df: pd.DataFrame) -> Tuple[Figure, Axes]:
    models = sorted(df["model"].unique())
    layouts = df["layout"].unique()
    fig, axs = plt.subplots(2, 3, figsize=(15, 5), sharex=False, sharey=False)
    palette = sns.color_palette(colors)

    for i, layout in enumerate(layouts):
        row, col = divmod(i, 3)
        df_layout = df[df["layout"] == layout]
        sns.lineplot(
            data=df_layout,
            x="timestep",
            y="throughput",
            hue="model",
            hue_order=models,
            legend=False,
            palette=palette,
            ax=axs[row, col],
            linewidth=2,
            alpha=0.7,
        )

        axs[row, col].set(
            xlabel="Timestep" if row == 1 else "",
            ylabel="Throughput" if col == 0 else "",
        )
        axs[row, col].set_title(layout, fontstyle="italic")

        if row == 0:
            axs[row, col].set(xticks=[], xticklabels=[])
        if col > 0:
            axs[row, col].set(yticks=[], yticklabels=[])

        sns.despine(ax=axs[row, col], left=col > 0, bottom=row == 0)

    sns.lineplot(
        data=df,
        x="timestep",
        y="throughput",
        hue="model",
        hue_order=models,
        palette=palette,
        ax=axs[1, 2],
        linewidth=4,
    )
    axs[1, 2].set(xlabel="Timestep", ylabel="", yticks=[], yticklabels=[])
    axs[1, 2].set_title("Overall", fontweight="bold")
    sns.despine(ax=axs[1, 2], left=True)

    fig.suptitle("Soup throughput")
    fig.tight_layout()
    return fig, axs


def do_task_proportion_plots(df: pd.DataFrame) -> Tuple[Figure, Axes]:
    model_names = sorted(df["model"].unique())
    fig, axs = plt.subplots(1, len(model_names), figsize=(14, 4))

    model_titles = {"mlp": "MLP", "rnn": "RNN", "rnn-single": "RNN (single-partner)"}
    for i, mn in enumerate(model_names):
        df_ = df[df["model"] == mn]

        reg_results = do_regression(df_, "speed_diff", "prop_0")
        sns.scatterplot(
            data=df_,
            x="speed_diff",
            y="prop_0",
            ax=axs[i],
            **scatter_kwargs,
            c=colors[i],
        )

        x_min, x_max = df_["speed_diff"].min(), df_["speed_diff"].max()
        x_vals = np.linspace(x_min, x_max, 100)
        y_mid = reg_results["intercept"] + reg_results["slope"] * x_vals
        sns.lineplot(x=x_vals, y=y_mid, ax=axs[i], **line_kwargs, color=colors[i])

        # plot 95% confidence intervals
        y_low, y_high = (
            reg_results["intercept_conf_int"][0]
            + reg_results["slope_conf_int"][0] * x_vals,
            reg_results["intercept_conf_int"][1]
            + reg_results["slope_conf_int"][1] * x_vals,
        )
        for y_ in [y_low, y_high]:
            axs[i].plot(x_vals, y_, color=colors[i], alpha=0.1)
        axs[i].fill_between(x_vals, y_low, y_high, color=colors[i], alpha=0.2)

        # annotate with r2 value
        axs[i].annotate(
            f"$R^2 = {reg_results['r2']:.2f}$",
            xy=(0.05, 0.9),
            xycoords="axes fraction",
            size=16,
            weight="bold",
        )

        # annotate with p-value
        p_val = reg_results["p_value"]
        p_str = "p < 1e-4" if p_val < 1e-4 else f"p = {p_val:.2e}".lower()
        axs[i].annotate(
            f"${p_str}$",
            xy=(0.05, 0.8),
            xycoords="axes fraction",
            size=16,
            weight="bold",
        )

        axs[i].set(
            xlabel="Task 1 speed $-$ task 2 speed",
            ylabel="Proportion of time spent on task 1",
            title=model_titles[mn],
        )

        if i > 0:
            axs[i].set(yticks=[], yticklabels=[], ylabel="")

        sns.despine(ax=axs[i], left=i > 0)

    fig.suptitle("Partner task allocation vs speed difference")
    return fig, axs


with open("analysis/metadata.json", "r") as f:
    metadata_dict = json.load(f)

artifact_names = [
    dir_name
    for dir_name in os.listdir("artifacts")
    if "rollout" in dir_name and metadata_dict[dir_name]["run_type"] == "basic"
]

rollouts, model_names, layouts, speeds, ego_speeds = [], [], [], [], []
for an in tqdm(artifact_names, desc="loading rollouts"):
    try:
        tmp = an.split(".json")[0].split("_")
        speed_combo = f"{tmp[-2]}_{tmp[-1]}"
        rollouts.append(load_rollout(an))
        model_names.append(metadata_dict[an]["model_name"])
        layouts.append(metadata_dict[an]["layout"])
        speeds.append(speed_combo)
        ego_speeds.append(metadata_dict[an]["ego_speed"])
    except Exception as e:
        print(f"Error loading rollout {an}: {e}")

reward_df = make_total_reward_df(rollouts, model_names, layouts, speeds)
reward_df.to_csv("reward_df.csv")

throughput_df = make_throughput_df(rollouts, model_names, layouts)
throughput_df.to_csv("throughput_df.csv")

task_df = make_task_df(rollouts, model_names, layouts)
task_df.to_csv("task_df.csv")

fig, _ = do_layout_reward_plot(reward_df)
save_fig(fig, "performance-new/reward")

fig, _ = do_throughput_plot(throughput_df)
save_fig(fig, "performance-new/throughput")

fig, _ = do_task_proportion_plots(task_df)
save_fig(fig, "performance-new/task_proportion")
