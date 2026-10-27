"""Replot the original protocol training returns as nine panels of five seeds.

Run: python train/plot_per_seed_training_returns.py
Uses the saved raw updates and the original trailing-mean implementation;
checks every plotted value against the original saved smoothed curves.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
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
from train.plot_protocol_training_curves import VERSIONS, trailing_mean

SOURCE = ROOT / "/juice6/u/jshe/emergent_partner_grid/train/training_curves/v1_matched_v2_v3_20261019_191029_step_penalty_0.03"
CONDITIONS = (
    ("mlp_diverse_influence", "MLP"),
    ("rnn_diverse_influence", "RNN diverse-partner"),
    ("rnn_single_influence", "RNN single-partner"),
)
SEEDS = tuple(range(1, 6))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_curves(path):
    curves = {}
    selected = {condition for condition, _ in CONDITIONS}
    with path.open() as file:
        for row in csv.DictReader(file):
            if row["condition"] not in selected or row["version"] not in VERSIONS:
                continue
            key = (row["version"], row["condition"], int(row["seed"]))
            curves.setdefault(key, []).append((int(row["update"]),
                int(row["env_steps"]), float(row["episode_return"])))
    expected = {(v, c, s) for v in VERSIONS for c, _ in CONDITIONS for s in SEEDS}
    if set(curves) != expected:
        raise ValueError(f"Unexpected seed coverage in {path}")
    return {key: np.asarray(rows) for key, rows in curves.items()}


def make_figure(curves, window, steps_per_update):
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(3, 3, figsize=(15.8, 11.6), sharex=True, sharey=True)
    for row, (condition, label) in enumerate(CONDITIONS):
        for col, version in enumerate(VERSIONS):
            ax = axes[row, col]
            for seed in SEEDS:
                data = curves[(version, condition, seed)]
                ax.plot(data[:, 1] / 1e6, data[:, 2], color=colors[seed - 1],
                        lw=1.5, alpha=.95, label=f"Seed {seed}")
            if len(ax.lines) != 5:
                raise ValueError("Every panel must contain exactly five curves")
            ax.set_xlim(0, 60)
            ax.set_xticks(np.arange(0, 61, 10))
            ax.grid(alpha=.18)
            if row == 0:
                ax.set_title(version, fontsize=17, weight="bold", pad=13)
            if col == 0:
                ax.set_ylabel(label, fontsize=13, labelpad=12)
    fig.suptitle("Training episode return · every training seed",
                 fontsize=21, weight="bold", y=.985)
    handles = [Line2D([], [], color=colors[s - 1], lw=2, label=f"Seed {s}")
               for s in SEEDS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .95),
               ncol=5, frameon=False, fontsize=11, handlelength=1.8,
               columnspacing=1.25)
    fig.supylabel("Training return per 20-round episode", x=.012, fontsize=13)
    fig.supxlabel("Collected environment steps (millions)", y=.067, fontsize=13)
    fig.text(.5, .025,
             f"Each line is one seed. Trailing {window}-update mean "
             f"({window * steps_per_update / 1e6:.2f}M steps); "
             "early windows use available updates.\n"
             "Training partners: single-partner RNN uses (1,4); "
             "MLP and diverse-partner RNN use 24 profiles.",
             ha="center", va="center", fontsize=10, color="#444444",
             linespacing=1.6)
    fig.subplots_adjust(left=.105, right=.985, top=.845, bottom=.13,
                        wspace=.08, hspace=.13)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=SOURCE / "per_seed_returns")
    args = parser.parse_args()
    source, out = args.source_dir.resolve(), args.out_dir.resolve()
    metadata = json.loads((source / "summary.json").read_text())
    window = metadata["window_updates"]
    steps_per_update = metadata["window_steps"] // window
    raw = read_curves(source / "updates.csv")
    original_smoothed = read_curves(source / "smoothed_per_seed.csv")
    runs = {(r["version"], r["condition"], r["seed"]): r for r in metadata["runs"]}
    curves, rows, validation = {}, [], []
    for key in sorted(raw):
        data, saved, run = raw[key], original_smoothed[key], runs[key]
        n = run["updates"]
        if (data.shape != (n, 3) or saved.shape != data.shape
                or not np.array_equal(data[:, 0], np.arange(1, n + 1))
                or not np.array_equal(data[:, :2], saved[:, :2])
                or not np.array_equal(data[:, 1], data[:, 0] * steps_per_update)
                or run["steps_per_update"] != steps_per_update
                or int(data[-1, 1]) != run["effective_steps"]
                or np.isinf(data[:, 2]).any()):
            raise ValueError(f"Invalid training history: {key}")
        values = trailing_mean(data[:, 2], window)
        if not np.allclose(values, saved[:, 2], rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f"Smoothing differs from the original figure: {key}")
        curves[key] = np.column_stack((data[:, :2], values))
        version, condition, seed = key
        for idx, (update, steps, value) in enumerate(data):
            rows.append(dict(version=version, condition=condition, seed=seed,
                update=int(update), env_steps=int(steps), episode_return=value,
                smoothed_episode_return=values[idx]))
        validation.append(dict(version=version, condition=condition, seed=seed,
            updates=n, missing_episode_updates=int(np.isnan(data[:, 2]).sum()),
            final_env_steps=int(data[-1, 1]), log=run["log"],
            log_sha256=run["log_sha256"]))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_seed_returns.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42}):
        fig = make_figure(curves, window, steps_per_update)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"training_episode_return_per_seed.{ext}",
                        dpi=200, facecolor="white")
        plt.close(fig)
    report = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        original_figure=str(source / "training_episode_return.png"),
        original_plot_script=str(ROOT / "train/plot_protocol_training_curves.py"),
        original_chat=dict(title="Explain fixed_allocation variance difference",
            thread_id="01a13268-60f2-7342-88d3-16b19a24058c",
            turn_id="01a1327c-30f3-75a2-84a9-d4b9ef986ed3"),
        source_files={str(source / name): sha256(source / name)
            for name in ("updates.csv", "smoothed_per_seed.csv", "summary.json")},
        script_sha256=sha256(Path(__file__)),
        versions=list(VERSIONS), conditions=[c for c, _ in CONDITIONS],
        excluded_conditions=["rnn_diverse_noinfluence"], seeds=list(SEEDS),
        panels=9, curves_per_panel=5, policies=len(curves), records=len(rows),
        window_updates=window, window_steps=window * steps_per_update,
        smoothing=metadata["smoothing"],
        validation="All 45 recomputed curves match the original saved smoothed "
                   "returns to absolute tolerance 1e-12, including missing values.",
        runs=validation)
    (out / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(out), "panels": 9, "curves_per_panel": 5,
                      "policies": len(curves), "records": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
