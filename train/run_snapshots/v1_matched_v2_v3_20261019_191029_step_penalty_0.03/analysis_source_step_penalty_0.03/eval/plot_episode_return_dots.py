"""Plot saved episode-return means and seed SDs as protocol-specific dot panels."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


# The paper reference uses filled circles and uncapped vertical error bars.
# Colors identify conditions, while panels identify allocation protocols.
CONDITION_STYLES = (
    ("mlp_diverse_influence", "MLP", "#00b894"),
    ("rnn_diverse_influence", "RNN diverse-partner", "#9580ff"),
    ("rnn_single_influence", "RNN single-partner", "#e64b35"),
    ("rnn_diverse_noinfluence", "RNN no influence", "#e5b82e"),
)


def plot_episode_return_dots(report, out_dir, version_order=("v1_matched", "v2", "v3")):
    """Reuse the comparison's equal-seed means and sample standard deviations."""
    population = report["population"]
    if population not in ("train", "test"):
        raise ValueError(f"Unknown evaluation population: {population}")
    seed_count = len(report["seeds"])
    if seed_count < 2:
        raise ValueError("At least two training seeds are needed for sample SD")
    aggregates = report["aggregates"]
    lookup = {(a["version"], a["condition"]): a for a in aggregates}
    if len(lookup) != len(aggregates):
        raise ValueError("Duplicate version/condition aggregates")
    versions = tuple(v for v in version_order
                     if any(a["version"] == v for a in aggregates))
    if not versions or any(a["version"] not in versions for a in aggregates):
        raise ValueError("Expected aggregates for the requested version order")

    bounds = []
    for version in versions:
        for condition, _, _ in CONDITION_STYLES:
            entry = lookup[(version, condition)]
            mean = entry["episode_return_mean"]
            sd = entry["episode_return_sd"]
            if entry["n_seeds"] != seed_count:
                raise ValueError("All conditions must use the same training seeds")
            if not np.isfinite([mean, sd]).all() or sd < 0:
                raise ValueError("Return means and SDs must be finite; SD must be nonnegative")
            bounds.extend((mean - sd, mean + sd))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    population_label = "Novel" if population == "test" else "Shared training"
    with plt.rc_context({"font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, len(versions), figsize=(4.1 * len(versions), 4.8),
                                 sharey=True, squeeze=False)
        fig.subplots_adjust(left=.075, right=.985, bottom=.16, top=.70, wspace=.16)
        span = max(bounds) - min(bounds)
        padding = max(.6, .08 * span)
        for i, (version, ax) in enumerate(zip(versions, axes.flat)):
            for x, (condition, _, color) in enumerate(CONDITION_STYLES):
                entry = lookup[(version, condition)]
                ax.errorbar(x, entry["episode_return_mean"],
                            yerr=entry["episode_return_sd"], fmt="o",
                            markersize=11, markeredgewidth=0, color=color,
                            ecolor=color, elinewidth=2.8, capsize=0, zorder=3)
            ax.set_title(version, fontsize=16, fontweight="bold", pad=15)
            ax.set_xlim(-.5, 3.5)
            ax.set_ylim(min(bounds) - padding, max(bounds) + padding)
            ax.set_xticks([])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            for spine in ("top", "right", "bottom"):
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_visible(i == 0)
            ax.spines["left"].set_color("#555555")
            ax.tick_params(axis="y", left=i == 0, labelleft=i == 0,
                           colors="#333333", width=1, length=4)
            if i == 0:
                ax.set_ylabel("Average episode return", labelpad=10)

        legend = [Line2D([], [], marker="o", linestyle="none", markersize=11,
                         markeredgewidth=0, color=color, label=label)
                  for _, label, color in CONDITION_STYLES]
        fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(.985, .855),
                   ncol=4, frameon=True, fancybox=True, framealpha=1,
                   facecolor="#eeeeee", edgecolor="none", fontsize=11,
                   handlelength=1, handletextpad=.5, columnspacing=1.2)
        fig.text(.075, .95, report.get("plot_title", f"{population_label} partner episode return"),
                 fontsize=18, va="top", fontweight="bold")
        if report.get("plot_subtitle"):
            fig.text(.075, .905, report["plot_subtitle"],
                     fontsize=11, va="top", color="#555555")
        fig.text(.075, .035,
                 f"Dots: mean across {seed_count} training seeds. Error bars: sample SD. "
                 "Each episode contains 20 rounds.", fontsize=10, color="#555555")
        paths = tuple(out_dir / f"episode_return_dots.{ext}" for ext in ("png", "pdf", "svg"))
        for path in paths:
            fig.savefig(path, dpi=240, facecolor="white")
        plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path, help="Saved comparison.json from compare_allocation_protocols.py")
    parser.add_argument("--out-dir", type=Path, help="Defaults to the comparison.json directory")
    args = parser.parse_args()
    report = json.loads(args.comparison.read_text())
    paths = plot_episode_return_dots(report, args.out_dir or args.comparison.parent)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
