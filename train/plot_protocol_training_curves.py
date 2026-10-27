"""Extract losses and training returns for the evaluated V1_matched/V2/V3 runs.

Use immutable submission manifests to identify all five seeds per condition.
Reconstruct the total PPO objective from rounded actor/value/entropy summaries;
never treat optimizer loss as a behavioral score or a held-out evaluation.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_path, resolve_record_paths
from eval.plot_episode_return_dots import CONDITION_STYLES
from train.training_log_format import FIELDS, PATTERN

VERSIONS = ("v1_matched", "v2", "v3")
BATCHES = ('v1_matched_v2_v3_20261022_080155_t0fix_direct_input',)
METRICS = ("total_loss_reconstructed", "actor_loss", "value_loss", "entropy",
           "episode_return", "round_success", "episode_steps")
LABELS = {
    "total_loss_reconstructed": "Total PPO objective (reconstructed)",
    "actor_loss": "Actor loss (clipped surrogate)",
    "value_loss": "Value loss (critic)",
    "episode_return": "Training return per 20-round episode",
}


def trailing_mean(values, window):
    """Use only current/past finite values; early windows use available updates."""
    valid = np.isfinite(values)
    sums = np.r_[0.0, np.cumsum(np.where(valid, values, 0.0))]
    counts = np.r_[0, np.cumsum(valid)]
    stop = np.arange(1, len(values) + 1)
    start = np.maximum(0, stop - window)
    denominator = counts[stop] - counts[start]
    return np.divide(sums[stop] - sums[start], denominator,
                     out=np.full(len(values), np.nan), where=denominator > 0)


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract_runs():
    reference_path = ROOT / "train/manifests/sbatch_v1_matched_v2_v3_20261022_080155_t0fix_direct_input.json"
    reference = resolve_record_paths(json.loads(reference_path.read_text()))
    expected = {(v, c, s) for v in VERSIONS for c, _, _ in CONDITION_STYLES
                for s in reference["seeds"]}
    runs, provenance = {}, []
    for batch in BATCHES:
        manifest_path = ROOT / "train/manifests" / f"sbatch_{batch}.json"
        manifest = resolve_record_paths(json.loads(manifest_path.read_text()))
        for job in manifest["training_jobs"]:
            version, condition, seed = job["version"], job["condition"], job["seed"]
            if version not in VERSIONS:
                continue
            key = (version, condition, seed)
            if key in runs:
                raise ValueError(f"Duplicate policy: {key}")
            log_name = (f"train_{job['job_id']}.out" if "log_dir" in manifest
                        else f"cg_cap_{job['job_id']}.out")
            log_path = resolve_path(job.get("log_path", Path(manifest.get(
                "log_dir", ROOT / "train/slurm_logs")) / log_name))
            checkpoint_root = Path(manifest["versions"][version]["checkpoint_root"])
            eval_root = Path(reference["versions"][version]["evaluation_root"])
            if checkpoint_root.name != eval_root.name:
                raise ValueError(f"Training/evaluation batch mismatch: {key}")
            config_path = checkpoint_root / f"{job['tag']}_config.json"
            config = resolve_record_paths(json.loads(config_path.read_text()))
            audit_path = checkpoint_root / f"{job['tag']}_sampling_audit.json"
            audit = json.loads(audit_path.read_text())
            content = log_path.read_text()
            records = []
            for line in content.splitlines():
                if not line.startswith("[u "):
                    continue
                match = PATTERN.fullmatch(line)
                if match is None:
                    raise ValueError(f"Unrecognized update line in {log_path}: {line}")
                records.append(dict(zip(FIELDS, [float(x) if x is not None else np.nan
                                                 for x in match.groups()])))
            n_updates = config["NUM_UPDATES"]
            steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
            if len(records) != n_updates or n_updates != 915:
                raise ValueError(f"Incomplete training log: {log_path}")
            if [int(r["update"]) for r in records] != list(range(1, n_updates + 1)):
                raise ValueError(f"Unordered updates: {log_path}")
            if any(r["env_steps"] != r["update"] * steps_per_update for r in records):
                raise ValueError(f"Unexpected step counts: {log_path}")
            if sum(r["rounds_completed"] for r in records) != audit["rounds_completed"]:
                raise ValueError(f"Sampling audit mismatch: {log_path}")
            data = {field: np.array([r[field] for r in records]) for field in FIELDS}
            for metric in ("actor_loss", "value_loss", "entropy"):
                matches = re.findall(r"^\s+" + metric + r": \[(.*)\]$", content, re.M)
                if len(matches) != 1:
                    raise ValueError(f"Missing/duplicate {metric} summary: {log_path}")
                summary = np.array([float(x) for x in matches[0].split(",")])
                if len(summary) != n_updates or not np.isfinite(summary).all():
                    raise ValueError(f"Invalid {metric} summary: {log_path}")
                if metric in data and not np.allclose(summary, data[metric], rtol=0, atol=0.000101):
                    raise ValueError(f"Summary/callback mismatch for {metric}: {log_path}")
                data[metric] = summary
            data["total_loss_reconstructed"] = (data["actor_loss"]
                + config["VF_COEF"] * data["value_loss"]
                - config["ENT_COEF"] * data["entropy"])
            # Zero episode length denotes a placeholder before any completed episode.
            absent = data["episode_steps"] <= 0
            data["episode_return"][absent] = np.nan
            data["episode_steps"][absent] = np.nan
            runs[key] = data
            provenance.append({
                "version": version, "condition": condition, "seed": seed,
                "manifest": str(manifest_path), "job_id": job["job_id"],
                "log": str(log_path), "log_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "config": str(config_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "sampling_audit": str(audit_path), "evaluation_root": str(eval_root),
                "updates": n_updates, "steps_per_update": steps_per_update,
                "effective_steps": int(data["env_steps"][-1]),
                "vf_coef": config["VF_COEF"], "ent_coef": config["ENT_COEF"],
                "lr": config["LR"], "lr_warmup": config["LR_WARMUP"],
                "total_loss_rounding_bound": 0.00005 * (1 + abs(config["VF_COEF"]) + abs(config["ENT_COEF"])),
            })
    if set(runs) != expected:
        raise ValueError(f"Unexpected policy coverage; missing={expected-set(runs)}, extra={set(runs)-expected}")
    return runs, provenance


def plot_metrics(curves, out_dir, metrics, name, window, raw=False):
    fig, axes = plt.subplots(len(metrics), 3, figsize=(15.2, 3.6 * len(metrics) + 1),
                             sharex=True, sharey="row", squeeze=False)
    for col, version in enumerate(VERSIONS):
        for condition, label, color in CONDITION_STYLES:
            policies = [curves[(version, condition, s)] for s in range(1, 6)]
            x = policies[0]["env_steps"] / 1e6
            for row, metric in enumerate(metrics):
                values = np.stack([p[metric] for p in policies])
                valid_count = np.isfinite(values).sum(axis=0)
                mean = np.divide(np.nansum(values, axis=0), valid_count,
                                 out=np.full(len(x), np.nan), where=valid_count > 0)
                variance = np.divide(np.nansum((values - mean) ** 2, axis=0), valid_count - 1,
                                     out=np.full(len(x), np.nan), where=valid_count > 1)
                # Only compare complete five-policy aggregates (early returns may be missing).
                mean[valid_count != 5] = np.nan
                sd = np.sqrt(variance)
                axes[row, col].plot(x, mean, color=color, lw=1.8, label=label)
                axes[row, col].fill_between(x, mean - sd, mean + sd, color=color, alpha=.12)
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            ax.grid(alpha=.18)
            ax.set_xlim(0, 60)
            if row == 0:
                ax.set_title(version, weight="bold", fontsize=15)
            if col == 0:
                ax.set_ylabel(LABELS[metric])
            if row == len(metrics) - 1:
                ax.set_xlabel("Collected environment steps (millions)")
    title = "Training episode return" if metrics == ("episode_return",) else "PPO training losses"
    fig.suptitle(title + " · all five training seeds", fontsize=18, weight="bold", y=.985)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4,
               bbox_to_anchor=(.5, .93), frameon=False)
    smoothing = ("No smoothing" if raw else
                 f"Trailing {window}-update mean ({window * 65536 / 1e6:.2f}M steps); early windows use available updates")
    fig.text(.5, .025, smoothing + ". Bands: sample SD across five seeds.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .075, 1, .85 if len(metrics) == 1 else .89))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{name}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "train/training_curves/v1_matched_v2_v3_20261022_080155_t0fix_direct_input")
    parser.add_argument("--window-updates", type=int, default=46)
    args = parser.parse_args()
    if args.window_updates < 1:
        parser.error("--window-updates must be positive")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, provenance = extract_runs()
    raw_rows, smoothed_rows, summary_rows = [], [], []
    curves = {}
    for (version, condition, seed), data in runs.items():
        curves[(version, condition, seed)] = {
            "env_steps": data["env_steps"],
            **{metric: trailing_mean(data[metric], args.window_updates) for metric in METRICS},
        }
        for idx in range(len(data["update"])):
            identity = {"version": version, "condition": condition, "seed": seed,
                        "update": int(data["update"][idx]), "env_steps": int(data["env_steps"][idx])}
            raw_rows.append({**identity, **{m: float(data[m][idx]) for m in data if m not in identity}})
            smoothed_rows.append({**identity, **{m: float(curves[(version, condition, seed)][m][idx]) for m in METRICS}})
    for version in VERSIONS:
        for condition, _, _ in CONDITION_STYLES:
            for idx in range(915):
                entry = {"version": version, "condition": condition, "update": idx + 1,
                         "env_steps": int(curves[(version, condition, 1)]["env_steps"][idx])}
                for metric in METRICS:
                    values = np.array([curves[(version, condition, s)][metric][idx] for s in range(1, 6)])
                    finite = values[np.isfinite(values)]
                    entry[metric + "_n_seeds"] = len(finite)
                    entry[metric + "_mean"] = float(finite.mean()) if len(finite) == 5 else np.nan
                    entry[metric + "_sd"] = float(finite.std(ddof=1)) if len(finite) == 5 else np.nan
                summary_rows.append(entry)
    write_csv(out_dir / "updates.csv", raw_rows)
    write_csv(out_dir / "smoothed_per_seed.csv", smoothed_rows)
    write_csv(out_dir / "smoothed_summary.csv", summary_rows)
    write_csv(out_dir / "final_window_summary.csv", [r for r in summary_rows if r["update"] == 915])
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        plot_metrics(curves, out_dir, ("total_loss_reconstructed",), "total_ppo_loss", args.window_updates)
        plot_metrics(curves, out_dir, ("actor_loss", "value_loss"), "loss_components", args.window_updates)
        plot_metrics(runs, out_dir, ("actor_loss", "value_loss"), "loss_components_unsmoothed", args.window_updates, raw=True)
        plot_metrics(curves, out_dir, ("episode_return",), "training_episode_return", args.window_updates)
    summary = {
        "versions": list(VERSIONS), "seeds": list(range(1, 6)), "policies": len(runs),
        "raw_update_records": len(raw_rows), "window_updates": args.window_updates,
        "window_steps": args.window_updates * 65536,
        "smoothing": "Trailing equal-update mean per policy; early windows use available updates; missing episode placeholders excluded.",
        "aggregation": "Equal-policy mean and sample SD after smoothing; means require all five seeds.",
        "total_loss": "actor_loss + VF_COEF * value_loss - ENT_COEF * entropy, reconstructed from four-decimal summaries; exact total loss was not saved.",
        "limitations": [
            "PPO losses are optimization diagnostics, not test losses or behavioral performance scores.",
            "Actor loss uses normalized advantages; values near zero do not establish task convergence.",
            "Value targets change during training, so critic loss need not decline monotonically.",
            "Training episode returns average completed episodes per update; episode counts were not saved, so updates are weighted equally.",
            "Single-partner training uses only (1,4); other conditions use the 24-profile training population.",
            "Only final policy checkpoints were saved; these logs do not recover testing curves during training.",
            "Printed loss/entropy summaries have four decimals; return has three decimals, length one decimal, and success three decimals.",
        ],
        "runs": provenance,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(out_dir), "policies": len(runs), "updates": len(raw_rows)}, indent=2))


if __name__ == "__main__":
    main()
