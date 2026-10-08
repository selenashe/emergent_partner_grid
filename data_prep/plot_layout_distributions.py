"""Plot partner/ego BFS distances and realized wall density for a saved corpus.

Usage:
    python data_prep/plot_layout_distributions.py --corpus_dir data_prep/grids_capability_selected_2000

Distances are measured around walls. Equidistance plots use only layouts
whose partner is equally distant from red and blue. Signed partner-distance
differences, full-corpus ego distances, and wall density use all training
layouts. Density subplot bins include the lower bound and exclude the upper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from capability_selection import layout_bfs_distances


def _summary(values: np.ndarray) -> dict:
    return {
        "n": int(values.size), "min": float(values.min()),
        "max": float(values.max()), "mean": float(values.mean()),
        "median": float(np.median(values)),
    }


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18)


def _save(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.png", dpi=180)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)


def _csv(out_dir: Path, name: str, rows: list[dict]) -> None:
    with (out_dir / f"{name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # Audit guide:
    # Read saved geometry metadata and plot distance and wall-density distributions plus
    # balancing feasibility diagnostics. These are descriptive corpus checks, not
    # another training run. Inspect both goal marginals to verify a simultaneous
    # distance-balance claim.
    #
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    out_dir = args.output_dir or args.corpus_dir / "diagnostics" / "layout_distributions"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted((args.corpus_dir / "layouts" / "train").glob("*.json"))
    if not paths:
        raise ValueError("No training layouts found")

    records = []
    for path in paths:
        layout = json.loads(path.read_text())
        grid = np.asarray(layout["grid"], dtype=np.int32)
        # Recompute distances from the actual saved grid, checking metadata.
        distances = layout_bfs_distances(
            (grid == 1).astype(np.int32), tuple(layout["ego_start"]), tuple(layout["partner_start"]),
            tuple(layout["red_goal"]), tuple(layout["blue_goal"]),
        )
        for key, value in distances.items():
            assert value == layout["metadata"][key], (path, key)
        walls = int(np.sum(grid == 1))
        records.append({
            "layout_id": layout["layout_id"], **distances,
            "wall_count": walls, "grid_cells": int(grid.size),
            "wall_density_percent": 100.0 * walls / grid.size,
            "partner_equidistant": distances["partner_to_red"] == distances["partner_to_blue"],
            "partner_distance_delta_red_minus_blue": distances["partner_to_red"] - distances["partner_to_blue"],
        })
    _csv(out_dir, "layout_metrics", records)
    eq = [record for record in records if record["partner_equidistant"]]
    if not eq:
        raise ValueError("No partner-equidistant layouts found")
    density = np.array([r["wall_density_percent"] for r in records])
    eq_density = np.array([r["wall_density_percent"] for r in eq])
    distance = np.array([r["partner_to_red"] for r in eq])
    red = np.array([r["ego_to_red"] for r in eq])
    blue = np.array([r["ego_to_blue"] for r in eq])
    partner_steps = np.arange(min(3, int(distance.min())), int(distance.max()) + 1)
    partner_counts = np.array([np.sum(distance == d) for d in partner_steps])
    assert int(partner_counts.sum()) == len(eq)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(partner_steps, partner_counts, width=0.9, color="#4b78b8", edgecolor="white")
    for step, count in zip(partner_steps, partner_counts):
        ax.annotate(str(count), (step, count), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)
    ax.set(title=f"Partner distance to each goal — {len(eq):,} equidistant layouts",
           xlabel="Shortest-path steps to each goal", ylabel="Number of layouts",
           xticks=partner_steps, ylim=(0, max(partner_counts) * 1.17))
    _style(ax)
    fig.tight_layout()
    _save(fig, out_dir, "partner-goal-distance-histogram")
    _csv(out_dir, "partner-goal-distance-histogram", [
        {"steps_to_each_goal": int(d), "count": int(c), "percent_equidistant": 100 * int(c) / len(eq)}
        for d, c in zip(partner_steps, partner_counts)
    ])

    # Same five-wall bins as the earlier histogram (half-integer boundaries).
    cells = {r["grid_cells"] for r in records}
    if len(cells) != 1:
        raise ValueError("Wall-count bins require grids of the same size")
    grid_cells = cells.pop()
    max_walls = max(r["wall_count"] for r in records)
    wall_edges = np.arange(-0.5, (max_walls // 5 + 1) * 5 + 0.5, 5)
    density_edges = 100 * wall_edges / grid_cells
    counts, _ = np.histogram(density, bins=density_edges)
    assert int(counts.sum()) == len(records)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(density, bins=density_edges, color="#4b78b8", edgecolor="white")
    for left, right, count in zip(density_edges[:-1], density_edges[1:], counts):
        ax.annotate(str(count), ((left + right) / 2, count), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)
    ax.set(title=f"Wall density across all {len(records):,} training layouts",
           xlabel="Grid cells occupied by walls (%)", ylabel="Number of layouts",
           xlim=(density_edges[0], density_edges[-1]), ylim=(0, max(counts) * 1.17))
    _style(ax)
    fig.tight_layout()
    _save(fig, out_dir, "wall-density-histogram")
    _csv(out_dir, "wall-density-histogram", [
        {"min_walls": i * 5, "max_walls": i * 5 + 4, "count": int(c)}
        for i, c in enumerate(counts)
    ])

    # Cover every density bin present in the full corpus, including empty
    # equidistant subsets. The 100% endpoint cannot occur with four free roles.
    n_bins = int(density.max() // 10) + 1
    n_columns = min(4, n_bins)
    n_rows = math.ceil(n_bins / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(3.5 * n_columns, 3.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    density_rows = []
    for i, ax in enumerate(axes.flat):
        if i >= n_bins:
            ax.set_visible(False)
            continue
        low, high = i * 10, (i + 1) * 10
        mask = (eq_density >= low) & (eq_density < high)
        bin_counts = np.array([np.sum(distance[mask] == d) for d in partner_steps])
        assert int(bin_counts.sum()) == int(mask.sum())
        ax.bar(partner_steps, bin_counts, width=0.9, color="#4b78b8", edgecolor="white")
        ax.set_title(f"{low}–{high}% walls · n={int(mask.sum())}", fontsize=11)
        ax.set_xticks(partner_steps)
        ax.tick_params(labelbottom=True, labelleft=True, labelsize=9)
        _style(ax)
        density_rows.extend([
            {"wall_density_lower_percent": low, "wall_density_upper_percent_exclusive": high,
             "steps_to_each_goal": int(d), "count": int(c), "n_equidistant_in_bin": int(mask.sum())}
            for d, c in zip(partner_steps, bin_counts)
        ])
    assert sum(row["count"] for row in density_rows) == len(eq)
    fig.suptitle(f"Partner distance to each goal, grouped by wall density\n"
                 f"{len(eq):,} layouts with equal shortest-path distances to both goals", fontsize=14)
    fig.supxlabel("Shortest-path steps to each goal")
    fig.supylabel("Number of layouts")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.91))
    _save(fig, out_dir, "partner-distance-by-wall-density")
    _csv(out_dir, "partner-distance-by-wall-density", density_rows)

    partner_delta = np.array([r["partner_distance_delta_red_minus_blue"] for r in records])
    delta_extent = int(np.max(np.abs(partner_delta)))
    delta_steps = np.arange(-delta_extent, delta_extent + 1)
    delta_counts = np.array([np.sum(partner_delta == d) for d in delta_steps])
    assert int(delta_counts.sum()) == len(records)
    assert int(np.sum(partner_delta == 0)) == len(eq)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(delta_steps, delta_counts, width=0.9, color="#4b78b8", edgecolor="white")
    ax.set(title=f"Partner distance difference across all {len(records):,} layouts\n"
                 f"Mean {partner_delta.mean():.2f} · median {np.median(partner_delta):g}",
           xlabel="Partner distance to red − partner distance to blue (steps)",
           ylabel="Number of layouts", xticks=delta_steps)
    _style(ax)
    fig.tight_layout()
    _save(fig, out_dir, "partner-distance-delta-all-layouts")
    _csv(out_dir, "partner-distance-delta-all-layouts", [
        {"red_minus_blue_steps": int(d), "count": int(c), "percent_all_layouts": 100 * int(c) / len(records)}
        for d, c in zip(delta_steps, delta_counts)
    ])

    all_red = np.array([r["ego_to_red"] for r in records])
    all_blue = np.array([r["ego_to_blue"] for r in records])
    for red_values, blue_values, subset_label, name in (
            (red, blue, f"the {len(eq):,} partner-equidistant layouts", "ego-distances-partner-equidistant"),
            (all_red, all_blue, f"all {len(records):,} layouts", "ego-distances-all-layouts")):
        ego_steps = np.arange(1, int(max(red_values.max(), blue_values.max())) + 1)
        red_counts = np.array([np.sum(red_values == d) for d in ego_steps])
        blue_counts = np.array([np.sum(blue_values == d) for d in ego_steps])
        assert int(red_counts.sum()) == int(blue_counts.sum()) == len(red_values)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
        for ax, values, counts, color, label in zip(
                axes, [red_values, blue_values], [red_counts, blue_counts],
                ["#cc504b", "#4b78b8"], ["Red goal", "Blue goal"]):
            ax.bar(ego_steps, counts, width=0.9, color=color, edgecolor="white")
            ax.set_title(f"{label} · mean {values.mean():.2f} · median {np.median(values):g}")
            ax.set_xticks(ego_steps)
            ax.tick_params(labelleft=True)
            ax.set_xlabel("Ego shortest-path steps to goal")
            _style(ax)
        axes[0].set_ylabel("Number of layouts")
        fig.suptitle(f"Ego distances in {subset_label}", fontsize=14)
        fig.tight_layout()
        _save(fig, out_dir, name)
        _csv(out_dir, name, [
            {"steps": int(d), "red_count": int(r), "blue_count": int(b)}
            for d, r, b in zip(ego_steps, red_counts, blue_counts)
        ])

    summary = {
        "n_layouts": len(records), "n_partner_equidistant": len(eq),
        "percent_partner_equidistant": 100 * len(eq) / len(records),
        "wall_density_percent_all_layouts": _summary(density),
        "partner_distance_equidistant_layouts": _summary(distance),
        "ego_red_distance_partner_equidistant_layouts": _summary(red),
        "ego_blue_distance_partner_equidistant_layouts": _summary(blue),
        "partner_distance_delta_red_minus_blue_all_layouts": _summary(partner_delta),
        "partner_distance_delta_counts": {
            "negative_red_closer": int(np.sum(partner_delta < 0)),
            "zero_equidistant": int(np.sum(partner_delta == 0)),
            "positive_blue_closer": int(np.sum(partner_delta > 0)),
        },
        "ego_red_distance_all_layouts": _summary(all_red),
        "ego_blue_distance_all_layouts": _summary(all_blue),
        "density_subplot_bins": "[lower, upper), in percentage points, width 10",
        "wall_histogram_bins": "Five wall cells per bin, starting at 0–4 walls",
        "reproduce_command": f"python data_prep/plot_layout_distributions.py --corpus_dir {args.corpus_dir.resolve()} --output_dir {out_dir.resolve()}",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
