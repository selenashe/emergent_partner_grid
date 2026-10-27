"""Plot sliding experiment episode returns from saved per-seed evaluations."""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CONDITIONS = (
    ("mlp_diverse_influence", "MLP", "#00b894"),
    ("rnn_diverse_influence", "RNN diverse-partner", "#9580ff"),
    ("rnn_single_influence", "RNN single-partner", "#e64b35"),
    ("rnn_diverse_noinfluence", "RNN no influence", "#e5b82e"),
)


def plot_episode_returns(results_dir, population="test", *, extra_results=(), out_dir=None,
                         adaptive_seeds=None, selection_population="test"):
    results_dir = Path(results_dir)
    if population not in ("train", "test"):
        raise ValueError("Population must be train or test")
    merged = {}
    for directory in (results_dir, *(Path(p) for p in extra_results)):
        with (directory / "behavior_per_seed.csv").open() as handle:
            for row in csv.DictReader(handle):
                if row["population"] != population:
                    continue
                key = (row["reversibility"], row["condition"], int(row["t"]), int(row["seed"]))
                if key in merged:
                    if key[2] != 0 or float(merged[key]["episode_return"]) != float(row["episode_return"]):
                        raise ValueError("Duplicate policy or inconsistent shared t=0 baseline")
                    continue
                merged[key] = row
    rows = list(merged.values())
    times = sorted({int(r["t"]) for r in rows})
    if not times or times[0] < 0:
        raise ValueError("Require nonnegative observation times")
    expected = {(mode, condition, t, seed)
                for mode in ("irreversible", "reversible")
                for condition, _, _ in CONDITIONS for t in times for seed in range(1, 6)}
    keys = [(r["reversibility"], r["condition"], int(r["t"]), int(r["seed"])) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Require both allocation modes, four conditions and learner seeds 1–5 at every t")
    if not all(math.isfinite(float(r["episode_return"])) for r in rows):
        raise ValueError("Non-finite episode return")

    eligible = None
    if adaptive_seeds is not None:
        if selection_population not in ("train", "test"):
            raise ValueError("Selection population must be train or test")
        audited = {}
        with Path(adaptive_seeds).open() as handle:
            for row in csv.DictReader(handle):
                if row["population"] != selection_population or int(row["t"]) not in times:
                    continue
                key = (row["reversibility"], row["condition"], int(row["t"]), int(row["seed"]))
                if key in audited:
                    raise ValueError("Duplicate strategy-audit policy")
                audited[key] = row
        if set(audited) != expected:
            raise ValueError("Strategy audit must cover every plotted policy")
        eligible = set()
        for key, row in audited.items():
            flag = row["initial_positive_specialization"]
            if flag not in ("True", "False"):
                raise ValueError("Invalid adaptation flag")
            calculated = (row["influence"] == "True"
                          and float(row["initial_late_adjusted_gap"]) >= .20
                          and float(row["initial_late_gap_ci_low"]) > 0)
            if (flag == "True") != calculated:
                raise ValueError("Adaptation flag disagrees with the audited criterion")
            if population == selection_population and not math.isclose(
                    float(row["episode_return"]), float(merged[key]["episode_return"]),
                    rel_tol=1e-6, abs_tol=2e-5):
                raise ValueError("Strategy audit and plotted return disagree")
            if calculated:
                eligible.add(key)

    # Retain the full five-seed plot's y limits for a direct visual comparison.
    bounds = []
    for mode in ("irreversible", "reversible"):
        for condition, _, _ in CONDITIONS:
            for t in times:
                values = [float(merged[mode, condition, t, seed]["episode_return"]) for seed in range(1, 6)]
                bounds.extend((mean(values) - stdev(values), mean(values) + stdev(values)))
    selected_cells = []

    with plt.rc_context({"font.size": 12, "pdf.fonttype": 42}):
        vertical = len(times) > 6
        fig, axes = plt.subplots(2 if vertical else 1, 1 if vertical else 2,
                                 figsize=(13, 8.2) if vertical else (12, 5.6), sharey=True)
        handles = [Line2D([], [], linestyle="none", marker="o", markersize=9,
                          markerfacecolor=color, markeredgecolor="none", label=label)
                   for _, label, color in CONDITIONS]
        offsets = (-.27, -.09, .09, .27)
        for ax, mode in zip(axes.flat, ("irreversible", "reversible")):
            for offset, (condition, label, color) in zip(offsets, CONDITIONS):
                cells = [sorted([r for r in rows if r["reversibility"] == mode
                                 and r["condition"] == condition and int(r["t"]) == t
                                 and (eligible is None or (mode, condition, t, int(r["seed"])) in eligible)],
                                key=lambda r: int(r["seed"])) for t in times]
                for t, cell in zip(times, cells):
                    values = [float(r["episode_return"]) for r in cell]
                    average = mean(values) if values else None
                    sd = stdev(values) if len(values) >= 2 else None
                    selected_cells.append(dict(population=population, selection_population=selection_population,
                                               allocation=mode, t=t, condition=condition, n_seeds=len(values),
                                               seed_ids=";".join(r["seed"] for r in cell),
                                               episode_return_mean=average, episode_return_sd=sd))
                    if not values:
                        continue
                    if sd is not None:
                        ax.errorbar(t + offset, average, yerr=sd, color=color,
                                    fmt="o", linestyle="none", markersize=8,
                                    markeredgewidth=0, elinewidth=2.5, capsize=0,
                                    zorder=3)
                    else:
                        ax.plot(t + offset, average, "o", color=color, markersize=8,
                                markeredgewidth=0, zorder=3)
                    if eligible is not None:
                        ax.annotate(f"n={len(values)}", (t + offset, average + (sd or 0)),
                                    xytext=(0, 7), textcoords="offset points",
                                    ha="center", va="bottom", fontsize=9, color=color)
            ax.set_title(mode.capitalize(), fontsize=15, fontweight="bold", pad=12)
            ax.set_xlabel("Pre-commit observed transitions t")
            ax.set_xticks(times)
            ax.set_xlim(times[0] - .6, times[-1] + .6)
            for spine in ("top", "right", "bottom"):
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#555555")
            ax.tick_params(axis="x", length=0)
            ax.set_ylabel("Mean episode return (20 rounds)")
        padding = max(.6, .06 * (max(bounds) - min(bounds)))
        axes.flat[0].set_ylim(min(bounds) - padding, max(bounds) + padding)
        partner = "Novel" if population == "test" else "Familiar"
        title = (f"{partner}-partner episode return — adaptive seeds only" if eligible is not None
                 else f"{partner}-partner episode return across observation time")
        fig.suptitle(title, fontsize=17, y=.985)
        fig.legend(handles=handles, labels=[label for _, label, _ in CONDITIONS],
                   loc="upper center", bbox_to_anchor=(.5, .94 if vertical else .91), ncol=4,
                   fontsize=11, frameon=True, fancybox=True, facecolor="#eeeeee",
                   edgecolor="none", handlelength=1, handletextpad=.4)
        fig.subplots_adjust(left=.085, right=.985, top=.84 if vertical else .76,
                            bottom=(.15 if eligible is not None else .12) if vertical else .18,
                            wspace=.12, hspace=.48)
        if eligible is None:
            caption = "Dots: five-seed mean; bars: sample SD. t=0 reuses counterbalanced sliding irreversible/v3."
        else:
            selector = "novel" if selection_population == "test" else "familiar"
            caption = ("Dots: eligible-seed mean; bars: sample SD for n≥2. n=1: dot only. Empty groups: no eligible seeds.\n"
                       f"Post hoc selection on {selector} profiles: late initial role gap ≥0.20 and bootstrap 95% lower bound >0.\n"
                       "Y limits match the original five-seed plot; t=0 reuses counterbalanced sliding irreversible/v3.")
        fig.text(.5, .015 if eligible is not None else .035, caption,
                 ha="center", va="bottom", fontsize=9 if eligible is not None else 10)
        destination = Path(out_dir) if out_dir is not None else results_dir
        destination.mkdir(parents=True, exist_ok=True)
        stem = f"episode_return_{population}" + ("_adaptive" if eligible is not None else "")
        paths = [destination / f"{stem}.{ext}" for ext in ("png", "pdf", "svg")]
        for path in paths:
            fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        if eligible is not None:
            with (destination / f"{stem}_selected_cells.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(selected_cells[0]))
                writer.writeheader(); writer.writerows(selected_cells)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--population", choices=("train", "test"), default="test")
    parser.add_argument("--extra-results", type=Path, nargs="*", default=[], help="Additional batches; matching t=0 baselines are reused once")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--adaptive-seeds", type=Path, help="seed_strategy_summary.csv from the strategy audit")
    parser.add_argument("--selection-population", choices=("train", "test"), default="test",
                        help="Use one population's adaptation flags consistently across return plots")
    args = parser.parse_args()
    for path in plot_episode_returns(args.results_dir, args.population,
                                     extra_results=args.extra_results, out_dir=args.out_dir,
                                     adaptive_seeds=args.adaptive_seeds,
                                     selection_population=args.selection_population):
        print(path)
