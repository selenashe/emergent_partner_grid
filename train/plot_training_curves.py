"""Reconstruct v1/v2 learning curves from the logs of the evaluated runs.

Run with Python + NumPy + Matplotlib; no JAX/GPU or training changes needed.
The stdout episode metrics average completed episodes within each PPO update;
episode counts were not logged, so their smoothing weights updates equally.
"""

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_record_paths
BATCH = "balanced1096_20261002_022924"
AUDIT = ROOT / "eval/protocol_comparison" / BATCH / "final_training_grid_statistics.json"
OUT = ROOT / "train/training_curves" / BATCH
CONDITIONS = ["rnn_diverse_influence", "mlp_diverse_influence",
              "rnn_single_influence", "rnn_diverse_noinfluence"]
LABELS = ["Diverse RNN + influence", "Diverse MLP + influence",
          "Single RNN + influence", "Diverse RNN, no influence"]
COLORS = ["#7868d8", "#199c82", "#dc8b25", "#5988bf"]
WINDOW = 46  # 3,014,656 collected environment steps, trailing only.
PATTERN = re.compile(
    r"\[u\s+(\d+)\]\s+env=\s*(\d+)\s+rsucc=([\d.]+)"
    r"\s+n_rounds=\s*(\d+)\s+ret=([+\-\d.]+)\s+len=\s*([\d.]+)"
    r"\s+ent\(t0,t>=1\)=\(([\d.]+),([\d.]+)\)"
    r"\s+aL=([+\-\d.]+)\s+vL=([\d.]+).*?(?:switch=([\d.]+))?$"
)
FIELDS = ["update", "env_steps", "round_success", "rounds_completed",
          "episode_return", "episode_steps", "entropy_t0", "entropy_movement",
          "actor_loss", "value_loss", "assignment_switch_rate"]


def smooth(values, weights=None):
    """Full-width trailing means, preserving early missing episode metrics."""
    valid = np.isfinite(values)
    weights = np.ones_like(values) if weights is None else weights.copy()
    weights = np.where(valid, weights, 0.0)
    numerator = np.convolve(np.where(valid, values, 0.0) * weights,
                            np.ones(WINDOW), mode="valid")
    denominator = np.convolve(weights, np.ones(WINDOW), mode="valid")
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 0)


def savefig(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    audit = resolve_record_paths(json.loads(AUDIT.read_text()))
    curves, provenance, raw_rows = {}, [], []
    for run in audit["per_seed_training_rounds"]:
        content = Path(run["log"]).read_text()
        records = []
        for line in content.splitlines():
            if not line.startswith("[u "):
                continue
            match = PATTERN.fullmatch(line)
            if match is None:
                raise ValueError("Unrecognized update line: " + line)
            record = dict(zip(FIELDS, [float(x) if x is not None else float("nan")
                                      for x in match.groups()]))
            records.append(record)
        config = resolve_record_paths(json.loads(Path(run["config"]).read_text()))
        n = config["NUM_UPDATES"]
        steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
        assert len(records) == n == 915
        assert [int(r["update"]) for r in records] == list(range(1, n + 1))
        assert all(r["env_steps"] == r["update"] * steps_per_update for r in records)
        assert sum(r["rounds_completed"] for r in records) == run["completed_rounds"]
        key = (run["version"], run["condition"], run["seed"])
        data = {field: np.array([r[field] for r in records]) for field in FIELDS}
        # Zeros before the first completed episode are placeholders, not outcomes.
        absent = data["episode_steps"] <= 0
        data["episode_return"][absent] = np.nan
        data["episode_steps"][absent] = np.nan
        series = {field: smooth(data[field], data["rounds_completed"]
                               if field == "round_success" else None)
                  for field in FIELDS[2:] if field != "rounds_completed"}
        series["steps_millions"] = data["env_steps"][WINDOW - 1:] / 1e6
        curves[key] = series
        for record in records:
            raw_rows.append({"version": run["version"], "condition": run["condition"],
                             "seed": run["seed"], **record})
        provenance.append({**run, "log_sha256": hashlib.sha256(content.encode()).hexdigest(),
                           "updates": n, "saved_checkpoint": config["SAVE_PARAMS_PATH"],
                           "lr": config["LR"], "lr_warmup": config["LR_WARMUP"]})
    assert len(curves) == 40
    with (OUT / "updates.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    aggregates, snapshots, bands = {}, [], []
    for version in ["v1", "v2"]:
        for condition in CONDITIONS:
            seeds = [curves[(version, condition, s)] for s in range(1, 6)]
            x = seeds[0]["steps_millions"]
            group = {metric: np.stack([s[metric] for s in seeds])
                     for metric in seeds[0] if metric != "steps_millions"}
            means = {metric: values.mean(axis=0) for metric, values in group.items()}
            aggregates[(version, condition)] = (x, group, means)
            for budget in [5, 10, 15, 20, 30, 40, 45, 50, 55, 60]:
                idx = max(0, min(np.searchsorted(x, budget, side="right") - 1, len(x) - 1))
                row = {"version": version, "condition": condition,
                       "requested_budget_millions": budget, "actual_steps_millions": float(x[idx])}
                for metric in ["round_success", "episode_return", "episode_steps"]:
                    row[metric + "_mean"] = float(means[metric][idx])
                    row[metric + "_seed_sd"] = float(group[metric][:, idx].std(ddof=1))
                snapshots.append(row)

            def band_start(success, returns, lengths):
                within = ((success >= success[-1] - 0.02)
                          & (returns >= returns[-1] - 0.5)
                          & (lengths <= lengths[-1] * 1.05))
                sustained = np.logical_and.accumulate(within[::-1])[::-1]
                indices = np.flatnonzero(sustained)
                return float(x[indices[0]]) if indices.size else None

            entry = {"version": version, "condition": condition,
                     "group_mean_enters_and_stays_in_final_band_millions": band_start(
                         means["round_success"], means["episode_return"], means["episode_steps"]),
                     "per_seed_enters_and_stays_in_final_band_millions": {
                         str(seed): band_start(group["round_success"][seed - 1],
                                               group["episode_return"][seed - 1],
                                               group["episode_steps"][seed - 1])
                         for seed in range(1, 6)},
                     "late_return_change_45_to_final": float(means["episode_return"][-1]
                         - means["episode_return"][np.searchsorted(x, 45, side="right") - 1])}
            bands.append(entry)

    with (OUT / "budget_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(snapshots[0]))
        writer.writeheader()
        writer.writerows(snapshots)

    metrics = ["round_success", "episode_return", "episode_steps"]
    ylabels = ["Training round success (%)", "Training return per 20-round episode",
               "Training steps per 20-round episode"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey="col")
    for row, version in enumerate(["v1", "v2"]):
        for condition, label, color in zip(CONDITIONS, LABELS, COLORS):
            x, group, means = aggregates[(version, condition)]
            for col, metric in enumerate(metrics):
                scale = 100 if metric == "round_success" else 1
                mean = means[metric] * scale
                sd = group[metric].std(axis=0, ddof=1) * scale
                axes[row, col].plot(x, mean, color=color, label=label, lw=2)
                axes[row, col].fill_between(x, mean - sd, mean + sd, color=color, alpha=0.12)
        for col, ax in enumerate(axes[row]):
            ax.set_title(version + (": commit at round start" if version == "v1"
                                    else ": random start + switching"))
            ax.set_ylabel(ylabels[col])
            ax.grid(alpha=0.18)
            ax.set_xlim(0, 60)
            if row == 1:
                ax.set_xlabel("Collected environment steps (millions)")
    fig.suptitle("Learning curves across all five training seeds", fontsize=16)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False)
    fig.text(0.5, 0.015, "Trailing 3.01M-step averages; bands: sample SD across five seeds. "
             "Training outcomes, not periodic held-out evaluations.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.045, 1, 0.9))
    savefig(fig, "learning_curves")

    fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharex=True, sharey=True)
    for row, version in enumerate(["v1", "v2"]):
        for col, (condition, label) in enumerate(zip(CONDITIONS, LABELS)):
            ax = axes[row, col]
            for seed in range(1, 6):
                series = curves[(version, condition, seed)]
                ax.plot(series["steps_millions"], series["episode_return"], label=f"Seed {seed}", lw=1.6)
            ax.set_title(version + " · " + label, fontsize=10)
            ax.grid(alpha=0.18)
            ax.set_xlim(0, 60)
            if row == 1:
                ax.set_xlabel("Environment steps (millions)")
            if col == 0:
                ax.set_ylabel("Training episode return")
    fig.suptitle("Some seeds learn quickly; others keep improving late", fontsize=16)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.945), ncol=5, frameon=False)
    fig.text(0.5, 0.012, "Same trailing 3.01M-step smoothing. Single-partner training is evaluated "
             "here on its one training profile.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.045, 1, 0.9))
    savefig(fig, "episode_return_by_seed")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for condition, label, color in zip(CONDITIONS, LABELS, COLORS):
        for col, version in enumerate(["v1", "v2"]):
            x, group, means = aggregates[(version, condition)]
            axes[col].plot(x, means["entropy_movement"], color=color, label=label, lw=2)
        x, group, means = aggregates[("v2", condition)]
        axes[2].plot(x, means["assignment_switch_rate"] * 100, color=color, label=label, lw=2)
    for ax, title, ylabel in zip(axes, ["v1: movement entropy", "v2: movement + allocation entropy",
                                      "v2: actual partner assignment switches"],
                                 ["Policy entropy at t ≥ 1 (nats)", "Policy entropy at t ≥ 1 (nats)",
                                  "Switches per eligible step (%)"]):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Environment steps (millions)")
        ax.set_xlim(0, 60)
        ax.grid(alpha=0.18)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    savefig(fig, "policy_diagnostics")

    summary = {
        "runs": provenance, "window_updates": WINDOW,
        "window_steps": WINDOW * 65536,
        "smoothing": {"round_success": "Round-count-weighted trailing mean per seed",
                      "episode_return_and_length": "Equal-update-weighted trailing mean of completed-episode means; zero placeholders excluded",
                      "seed_aggregation": "Equal weight for each of five seeds; sample SD"},
        "final_band_definition": "Earliest smoothed point such that it and ALL subsequent points have success >= final - 2 percentage points, return >= final - 0.5, and episode steps <= final * 1.05. Final is last trailing-window value. A descriptive retrospective threshold, not a convergence test.",
        "final_band_results": bands, "budget_snapshots": snapshots,
        "limitations": ["Only final model checkpoints were saved; earlier held-out performance cannot be recovered.",
                        "Curves are stochastic-policy training outcomes on familiar partner profiles and training layouts.",
                        "Fewer sampled timesteps does not imply fewer distinct training layouts are sufficient.",
                        "Learning-rate warmup and cosine decay depend on the total update budget. A shorter fresh run has a different schedule from truncating this 60M run.",
                        "Plateau of task performance would not establish convergence of partner representations or probe accuracy.",
                        "Episode counts per update were not persisted, so exact pooled episode-weighted return/length curves cannot be reconstructed.",
                        "Printed success is rounded to 0.001, return to 0.001, length to 0.1."]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "bands": bands,
                      "snapshots": [s for s in snapshots if s["requested_budget_millions"] in [30, 45, 50, 60]]}, indent=2))


if __name__ == "__main__":
    main()
