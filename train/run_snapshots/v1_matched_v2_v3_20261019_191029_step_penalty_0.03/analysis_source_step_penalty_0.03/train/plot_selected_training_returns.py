"""Plot training histories for the exact policies in a saved held-out selection.

Selections are read, never recomputed. Training stdout preserves per-update
means only, so these plots deliberately contain no episode confidence intervals.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_path
from eval.plot_overcooked_paper_match import (
    VERSIONS, STYLES, DEFAULT_OUT, LOG_LINE, sha256, write_csv,
)

DEFAULT_SELECTION = DEFAULT_OUT / "best_of_5/selection.json"
OUTPUT_ROOT = ROOT / "/juice6/u/jshe/emergent_partner_grid/train/training_curves/v1_matched_v2_v3_20261019_191029_step_penalty_0.03/best_training_seeds"


def read_history(selected):
    """Verify original selection evidence and retain missing-episode updates."""
    path = resolve_path(selected["training_log"])
    raw = path.read_bytes()
    if sha256(raw) != selected["training_log_sha256"]:
        raise ValueError(f"Training log changed since held-out selection: {path}")
    rows = []
    for line in raw.decode().splitlines():
        if not line.startswith("[u "):
            continue
        match = LOG_LINE.match(line)
        if match is None:
            raise ValueError(f"Unrecognized training metric: {path}: {line}")
        update, steps, _, _, episode_return, episode_length = map(float, match.groups())
        valid = episode_length > 0
        rows.append(dict(version=selected["version"], condition=selected["condition"],
            seed=selected["seed"], update=int(update), environment_steps=int(steps),
            episode_return=episode_return if valid else None,
            mean_episode_steps=episode_length if valid else None,
            completed_episode_mean_available=valid))
    if [r["update"] for r in rows] != list(range(1, selected["final_update"] + 1)):
        raise ValueError(f"Missing or duplicated training updates: {path}")
    final = rows[-1]
    if (final["environment_steps"] != selected["environment_steps"]
            or final["episode_return"] != selected["training_return"]):
        raise ValueError(f"Final training score disagrees with held-out seed selection: {path}")
    return rows


def trailing_mean(values, window):
    """Equal-update trailing means; no-completed-episode placeholders are omitted."""
    values = np.asarray([np.nan if x is None else x for x in values], dtype=float)
    smoothed = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        group = values[i - window + 1:i + 1]
        if np.isfinite(group).any():
            smoothed[i] = np.nanmean(group)
    return values, smoothed


def save_figure(fig, out, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=240, facecolor="white")
    plt.close(fig)


def setup_panels():
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.8), sharey=True)
    fig.subplots_adjust(left=.085, right=.98, top=.69, bottom=.20, wspace=.14)
    for vi, (version, ax) in enumerate(zip(VERSIONS, axes)):
        ax.set_title(version, fontstyle="italic", fontsize=15, pad=16)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_visible(vi == 0)
        ax.spines["left"].set_color("#555555")
        ax.tick_params(axis="y", left=vi == 0, labelleft=vi == 0, colors="#333333")
    return fig, axes


def legend(fig, curves=False):
    handles = [Line2D([], [], marker=None if curves else "o",
                      linestyle="-" if curves else "none", color=color,
                      markersize=10, markeredgewidth=0, label=label)
               for _, label, color in STYLES]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.98, .87),
               ncol=4, frameon=True, fancybox=True, framealpha=1,
               facecolor="#eeeeee", edgecolor="none", fontsize=11, handlelength=1)


def make_plots(selected, histories, out, window):
    lookup = {(r["version"], r["condition"]): r for r in selected}
    with plt.rc_context({"font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = setup_panels()
        values = [r["training_return"] for r in selected]
        pad = max(.6, (max(values) - min(values)) * .12)
        for version, ax in zip(VERSIONS, axes):
            for x, (condition, _, color) in enumerate(STYLES):
                row = lookup[(version, condition)]
                ax.plot(x, row["training_return"], "o", color=color,
                        markersize=10, markeredgewidth=0)
                ax.annotate(f"s{row['seed']}", (x, row["training_return"]), xytext=(0, 10),
                            textcoords="offset points", ha="center", fontsize=9, color="#555555")
            ax.set_xlim(-.5, 3.5)
            ax.set_ylim(min(values) - pad, max(values) + pad * 1.5)
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
        axes[0].set_ylabel("Average episode return", labelpad=12)
        legend(fig)
        fig.text(.085, .96, "Selected seeds: return at final training update",
                 fontsize=18, fontweight="bold", va="top")
        fig.text(.085, .045,
                 "Same seeds as the best-training-seed held-out plot. Dots: final logged training episode-return means.\n"
                 "No confidence intervals: individual training episode returns were not saved. Each episode contains 20 rounds.\n"
                 "Training partners: single-partner RNN uses (1,4); other conditions use 24 diverse profiles.",
                 fontsize=10, color="#555555")
        save_figure(fig, out, "final_training_return_dots")

        fig, axes = setup_panels()
        fig.subplots_adjust(bottom=.27)
        for version, ax in zip(VERSIONS, axes):
            for condition, _, color in STYLES:
                selected_row = lookup[(version, condition)]
                history = histories[(version, condition)]
                x = np.array([r["environment_steps"] for r in history]) / 1e6
                raw, smoothed = trailing_mean([r["episode_return"] for r in history], window)
                ax.plot(x, raw, color=color, alpha=.14, linewidth=.65)
                ax.plot(x, smoothed, color=color, linewidth=1.8)
                ax.plot(x[-1], raw[-1], "o", color=color, markersize=4)
                # A color-matched seed key avoids crowded labels at the endpoints.
                label_index = next(i for i, (c, _, _) in enumerate(STYLES) if c == condition)
                ax.text(2, 17.5 - label_index * 2.2, f"s{selected_row['seed']}", color=color,
                        fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=.8))
            ax.set_xlabel("Environment steps (millions)")
            ax.set_xlim(0, 61)
            ax.set_xticks([0, 20, 40, 60])
            ax.set_ylim(-61, 21)
            ax.grid(axis="y", alpha=.15)
        axes[0].set_ylabel("Average episode return", labelpad=12)
        legend(fig, curves=True)
        fig.text(.085, .96, "Training returns of the selected seeds",
                 fontsize=18, fontweight="bold", va="top")
        fig.text(.085, .045,
                 f"Same seeds as the held-out plot. Faint lines: per-update means; solid lines: trailing {window}-update means.\n"
                 "Dots at the end: unsmoothed selection scores. No uncertainty bands; missing-episode updates are omitted.",
                 fontsize=10, color="#555555")
        save_figure(fig, out, "selected_training_return_curves")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--smoothing-window", type=int, default=46)
    args = parser.parse_args()
    if args.smoothing_window < 1:
        parser.error("Smoothing window must be positive")
    selection_path = args.selection.resolve()
    selection_raw = selection_path.read_bytes()
    selected = json.loads(selection_raw)["results"]
    expected = {(v, c) for v in VERSIONS for c, _, _ in STYLES}
    if len(selected) != len(expected) or {(r["version"], r["condition"]) for r in selected} != expected:
        raise ValueError("Expected exactly the twelve selected protocol/condition policies")
    histories = {(r["version"], r["condition"]): read_history(r) for r in selected}
    if any(args.smoothing_window > len(h) for h in histories.values()):
        raise ValueError("Smoothing window exceeds training history")
    out = (args.out_dir or OUTPUT_ROOT / selection_path.parent.name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for h in histories.values() for r in h]
    write_csv(out / "selected_training_history.csv", rows)
    metrics = [dict(version=r["version"], condition=r["condition"], seed=r["seed"],
                    final_training_return=r["training_return"], held_out_return=r["mean"],
                    final_update=r["final_update"], environment_steps=r["environment_steps"])
               for r in selected]
    write_csv(out / "selected_seed_metrics.csv", metrics)
    report = dict(selection=str(selection_path), selection_sha256=sha256(selection_raw),
        source_sha256=sha256(Path(__file__).read_bytes()), selected=selected,
        smoothing_window_updates=args.smoothing_window,
        metric="Mean completed-episode return within each training rollout collection update",
        uncertainty="Unavailable: only per-update means were saved, without individual episode returns or their counts",
        limitations=["Final logged rollout precedes the final gradient update.",
                     "Single-partner RNN trains with partner (1,4); other conditions train with their diverse partner populations.",
                     "Printed training returns have three decimal places.",
                     "Smoothing weights updates equally; it is a display aid, not the seed-selection statistic."])
    (out / "training_returns.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    make_plots(selected, histories, out, args.smoothing_window)
    for r in metrics:
        print(f"{r['version']} {r['condition']} s{r['seed']}: training={r['final_training_return']:.3f}, held-out={r['held_out_return']:.3f}")
    print(out)


if __name__ == "__main__":
    main()
