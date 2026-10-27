"""Plot held-out sliding results after training-only seed selection at each t.

Combine both original sliding batches, reusing their shared historical t=0
baselines once. Match plot_overcooked_paper_match's episode-return estimator
and ordinary episode-bootstrap confidence intervals.
"""
import argparse
import json
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import artifact_root, resolve_path
from train.training_log_format import LOG_LINE
from eval.plot_overcooked_paper_match import (
    STYLES, bootstrap_mean_ci, episode_returns, sha256, write_csv,
)

MODES = ("irreversible", "reversible")
SEEDS = tuple(range(1, 6))
DEFAULT_MANIFESTS = tuple(ROOT / "train/manifests" / f"sbatch_{batch}.json" for batch in (
    "sliding_observation_20261008_224954", "sliding_observation_20261009_194137"))
DEFAULT_OUT = ROOT / "eval/sliding_observation/best_training_seed_t0_t10"


def read_manifest(path, provenance):
    path = resolve_path(path).resolve()
    raw = path.read_bytes()
    provenance[str(path)] = dict(path=str(path), sha256=sha256(raw))
    return json.loads(raw)


def merge_records(manifests):
    """Require all cells and deduplicate only identical shared t=0 policies."""
    merged = {}
    for manifest in manifests:
        if manifest["evaluation_seeds"] != list(SEEDS):
            raise ValueError("Sliding seed selection requires exactly seeds 1-5")
        for original in manifest["evaluation_records"]:
            record = dict(original, batch=manifest["batch"])
            record["prefix"] = str(resolve_path(record["prefix"]))
            key = (record["variant"], record["condition"], record["seed"])
            mode = "reversible" if record["reversible"] else "irreversible"
            if (record["variant"] != f"t{record['t']}_{mode}"
                    or bool(record["baseline"]) != (record["t"] == 0)):
                raise ValueError(f"Inconsistent variant metadata: {key}")
            if key in merged:
                old = {k: v for k, v in merged[key].items() if k != "batch"}
                new = {k: v for k, v in record.items() if k != "batch"}
                if record["t"] != 0 or old != new:
                    raise ValueError(f"Conflicting or duplicate policy: {key}")
                continue
            merged[key] = record
    expected = {(f"t{t}_{mode}", c, s) for t in range(11) for mode in MODES
                for c, _, _ in STYLES for s in SEEDS}
    if set(merged) != expected:
        raise ValueError(f"Require all t=0-10 cells; missing={expected-set(merged)}, extra={set(merged)-expected}")
    return merged


def log_path(job):
    outputs = [x.split("=", 1)[1] for x in job["command"] if x.startswith("--output=")]
    if len(outputs) > 1:
        raise ValueError(f"Ambiguous training log: {job['tag']}")
    return resolve_path(outputs[0].replace("%j", str(job["job_id"])) if outputs
                        else ROOT / "train/slurm_logs" / f"cg_cap_{job['job_id']}.out")


def score_training(job, config_path, budget):
    """Use the final logged rollout, and reject partial or mismatched runs."""
    log = log_path(job)
    raw = log.read_bytes()
    updates = []
    for line in raw.decode().splitlines():
        if line.startswith("[u "):
            match = LOG_LINE.match(line)
            if match is None:
                raise ValueError(f"Unrecognized training metric: {log}: {line}")
            updates.append(tuple(float(value) for value in match.groups()))
    config_path = resolve_path(config_path)
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    n = int(config["NUM_UPDATES"])
    if len(updates) != n or [int(r[0]) for r in updates] != list(range(1, n + 1)):
        raise ValueError(f"Incomplete training log: {log}")
    steps = int(config["NUM_ENVS"]) * int(config["NUM_STEPS"])
    if any(int(r[1]) != int(r[0]) * steps for r in updates):
        raise ValueError(f"Inconsistent training budget: {log}")
    if config["SEED"] != job["seed"] or config["NUM_SEEDS"] != 1:
        raise ValueError(f"Configuration seed mismatch: {config_path}")
    final = updates[-1]
    if not np.isfinite(final).all() or final[5] <= 0 or int(final[1]) != budget:
        raise ValueError(f"Invalid final completed-episode return/budget: {log}")
    if "t" in job:
        env = config["ENV_KWARGS"]
        if (env["precommit_observation_steps"] != job["t"]
                or env["allocation_reversible"] != job["reversible"]):
            raise ValueError(f"Configuration variant mismatch: {config_path}")
    return dict(training_return=final[4], final_update=int(final[0]),
                environment_steps=int(final[1]), training_job_id=str(job["job_id"]),
                training_log=str(log), training_log_sha256=sha256(raw),
                config=str(config_path), config_sha256=sha256(config_raw))


def training_candidates(manifests, records, recovery_paths, provenance):
    retries = {}
    batches = {m["batch"] for m in manifests}
    for path in recovery_paths:
        plan = read_manifest(path, provenance)
        original = resolve_path(plan["original_manifest"])
        if original.stem.removeprefix("sbatch_") not in batches:
            raise ValueError(f"Recovery plan belongs to another batch: {path}")
        if sha256(original.read_bytes()) != plan["original_manifest_sha256"]:
            raise ValueError(f"Recovery manifest hash mismatch: {path}")
        for task in plan["tasks"]:
            if task["stage"] == "training":
                if task["target"] in retries:
                    raise ValueError(f"Multiple recovery training runs: {task['target']}")
                retries[task["target"]] = task

    jobs, configs, sources, budgets = {}, {}, {}, {}
    for manifest in manifests:
        for mode, link in zip(MODES, ("base_manifest", "v3_manifest")):
            base = read_manifest(manifest[link], provenance)
            variant = f"t0_{mode}"
            version = manifest["variants"][variant]["baseline_version"]
            root = Path(manifest["variants"][variant]["checkpoint_root"])
            for job in base["training_jobs"]:
                key = (variant, job["condition"], job["seed"])
                if job["version"] != version or key not in records:
                    continue
                if key in jobs and jobs[key] != job:
                    raise ValueError(f"Conflicting baseline training job: {key}")
                jobs[key], configs[key] = job, root / f"{job['tag']}_config.json"
                sources[key] = str(resolve_path(manifest[link]))
                budgets[key] = int(base["effective_timesteps_per_policy"])
        for job in manifest["training_jobs"]:
            key = (job["variant"], job["condition"], job["seed"])
            if key in jobs:
                raise ValueError(f"Duplicate training job: {key}")
            actual = dict(job)
            if job["tag"] in retries:
                actual.update({k: retries[job["tag"]][k] for k in ("command", "job_id")})
            jobs[key] = actual
            root = Path(manifest["variants"][job["variant"]]["checkpoint_root"])
            configs[key] = root / f"{job['tag']}_config.json"
            sources[key] = manifest["batch"]
            budgets[key] = int(manifest["effective_timesteps_per_policy"])
    if set(jobs) != set(records):
        raise ValueError("Training and evaluation policy sets disagree")
    rows = []
    for key in sorted(records, key=lambda k: (records[k]["t"], records[k]["reversible"], k[1], k[2])):
        record = records[key]
        rows.append(dict(variant=record["variant"], t=record["t"],
            reversibility="reversible" if record["reversible"] else "irreversible",
            condition=record["condition"], seed=record["seed"], baseline=record["baseline"],
            training_source=sources[key], **score_training(jobs[key], configs[key], budgets[key])))
    return rows


def select_candidates(candidates):
    selected = []
    for mode in MODES:
        for t in range(11):
            for condition, _, _ in STYLES:
                group = [r for r in candidates if r["t"] == t
                         and r["reversibility"] == mode and r["condition"] == condition]
                if sorted(r["seed"] for r in group) != list(SEEDS):
                    raise ValueError(f"Missing or duplicate training candidates: t{t}/{mode}/{condition}")
                if not all(np.isfinite(r["training_return"]) for r in group):
                    raise ValueError("Nonfinite selection score")
                selected.append(max(group, key=lambda r: (r["training_return"], -r["seed"])))
    return selected


def read_evaluation(record):
    path = resolve_path(record["prefix"] + "_test.h5")
    summary_path = resolve_path(record["prefix"] + "_summary.json")
    summary_raw = summary_path.read_bytes()
    summary = json.loads(summary_raw)["test"]
    keys = ("rewards", "dones", "round_done", "round_idx", "capability_pool", "capability_index_per_ep")
    with h5py.File(path) as handle:
        if handle.attrs.get("allocation_protocol", "fixed_allocation") != (
                "online_v3" if record["reversible"] else "fixed_allocation"):
            raise ValueError(f"Allocation protocol mismatch: {path}")
        if not record["baseline"]:
            for key in ("t", "reversible", "condition", "seed"):
                if handle.attrs[key] != record[key]:
                    raise ValueError(f"Rollout metadata mismatch: {path}: {key}")
        data = {key: np.asarray(handle[key]) for key in keys}
    returns = episode_returns(data["rewards"], data["dones"].astype(bool),
                              data["round_done"].astype(bool), data["round_idx"])
    profiles, pool = data["capability_index_per_ep"], data["capability_pool"]
    if pool.shape != (22, 2) or len(np.unique(pool, axis=0)) != 22:
        raise ValueError(f"Expected 22 distinct held-out profiles: {path}")
    if (profiles.shape != (440,) or not np.issubdtype(profiles.dtype, np.integer)
            or not np.array_equal(np.sort(profiles), np.repeat(np.arange(22), 20))):
        raise ValueError(f"Expected twenty episodes per held-out profile: {path}")
    if len(returns) != summary["n_episodes"] or summary["n_rounds"] != len(returns) * 20:
        raise ValueError(f"Evaluation counts disagree: {summary_path}")
    if not np.isclose(returns.mean(), summary["mean_ep_return"], atol=3e-5, rtol=0):
        raise ValueError(f"Returns disagree with saved summary: {summary_path}")
    source = dict(evaluation_rollouts=str(path), evaluation_summary=str(summary_path),
                  evaluation_summary_sha256=sha256(summary_raw),
                  scalar_datasets_sha256={k: sha256(data[k].tobytes()) for k in keys},
                  n_episodes=len(returns), n_held_out_profiles=22, episodes_per_profile=20)
    return returns, profiles, pool, source


def plot(results, out, overlay_training=False):
    lookup = {(r["reversibility"], r["t"], r["condition"]): r for r in results}
    bounds = [value for r in results for value in r["ci95"]]
    if overlay_training:
        bounds.extend(r["training_return"] for r in results)
    pad = max(.5, np.ptp(bounds) * .12)
    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(2, 1, figsize=(13.2, 8.8), sharex=True, sharey=True)
        fig.subplots_adjust(left=.085, right=.98, bottom=.16, top=.82, hspace=.32)
        for ax, mode in zip(axes, MODES):
            ax.axvspan(-.5, .5, color="#f3f3f3", zorder=0)
            for offset, (condition, label, color) in zip((-.27, -.09, .09, .27), STYLES):
                rows = [lookup[mode, t, condition] for t in range(11)]
                x = np.arange(11) + offset
                ax.vlines(x, [r["ci95"][0] for r in rows], [r["ci95"][1] for r in rows],
                          color=color, linewidth=2, zorder=2)
                ax.plot(x, [r["mean"] for r in rows], "o", color=color, markersize=6,
                        markeredgewidth=0, zorder=3)
                if overlay_training:
                    ax.plot(x, [r["training_return"] for r in rows], "D", color=color,
                            markerfacecolor="none", markersize=6, markeredgewidth=1.2, zorder=4)
                for position, row in zip(x, rows):
                    top = max(row["ci95"][1], row["training_return"]) if overlay_training else row["ci95"][1]
                    ax.annotate(f"s{row['seed']}", (position, top), xytext=(0, 5),
                                textcoords="offset points", ha="center", fontsize=6.5, color="#555555")
            ax.set_title(mode.capitalize() + " allocation", loc="left", fontsize=13, fontweight="bold", pad=12)
            ax.set_ylabel("Held-out episode return")
            ax.set_xlim(-.6, 10.6)
            ax.set_ylim(min(bounds) - pad * .5, max(bounds) + pad)
            ax.set_xticks(range(11))
            ax.grid(axis="y", color="#e6e6e6", linewidth=.7, zorder=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors="#444444")
        axes[0].tick_params(labelbottom=True)
        axes[-1].set_xlabel("Observation transitions before initial allocation (t)", labelpad=9)
        labels = {"RNN": "RNN diverse-partner"}
        handles = [Line2D([], [], marker="o", linestyle="none", color=color, markersize=7,
                          label=labels.get(label, label)) for _, label, color in STYLES]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .91),
                   ncol=4, frameon=False, fontsize=11)
        fig.text(.085, .975, "Sliding observation: held-out returns of the best training seeds",
                 fontsize=17, fontweight="bold", va="top")
        fig.text(.085, .935, "Independent selection at every t × allocation setting × model condition; seeds 1–5",
                 fontsize=11, color="#555555", va="top")
        note = ("Dots: mean of 440 held-out episodes; bars: 95% episode-bootstrap CI (1,000 resamples). Labels: selected seed.\n"
                "Selection uses only the final logged training return. Shared t=0 baselines appear once; shaded at left.")
        if overlay_training:
            note += "\nHollow diamonds: final logged training returns of the selected seeds."
        fig.text(.085, .035, note, fontsize=9.5, color="#555555", va="bottom")
        name = "heldout_episode_return" + ("_with_training_returns" if overlay_training else "")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"{name}.{ext}", dpi=220, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", default=DEFAULT_MANIFESTS)
    parser.add_argument("--recovery-plans", type=Path, nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    provenance = {}
    manifests = [read_manifest(path, provenance) for path in args.manifests]
    records = merge_records(manifests)
    recovery_paths = args.recovery_plans
    if recovery_paths is None:
        recovery_paths = [p for m in manifests for p in sorted(
            (artifact_root() / "recovery").glob(m["batch"] + "*/recovery.json"))]
    candidates = training_candidates(manifests, records, recovery_paths, provenance)
    # Freeze every selection before opening any held-out evaluation file.
    selected = select_candidates(candidates)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "training_candidates.csv", candidates)
    write_csv(out / "selected_training_seeds.csv", selected)
    print(f"Selected {len(selected)} policies from {len(candidates)} candidates using training returns only.", flush=True)
    results, episodes = [], []
    for index, row in enumerate(selected):
        record = records[row["variant"], row["condition"], row["seed"]]
        returns, profiles, pool, source = read_evaluation(record)
        rng_seed = args.bootstrap_seed + index
        mean, ci = bootstrap_mean_ci(returns, n_boot=1000, seed=rng_seed)
        results.append(dict(**row, **source, mean=mean, ci95=ci, bootstrap_seed=rng_seed))
        for ep, value in enumerate(returns):
            red, blue = pool[profiles[ep]]
            episodes.append(dict(variant=row["variant"], t=row["t"], reversibility=row["reversibility"],
                condition=row["condition"], seed=row["seed"], episode=ep,
                delay_red=int(red), delay_blue=int(blue), episode_return=float(value)))
        print(f"{row['variant']} {row['condition']} s{row['seed']}: train={row['training_return']:.3f}, held-out={mean:.3f}", flush=True)
    write_csv(out / "selected_seed_metrics.csv", [dict(variant=r["variant"], t=r["t"],
        reversibility=r["reversibility"], condition=r["condition"], seed=r["seed"],
        training_return=r["training_return"], mean=r["mean"], ci95_low=r["ci95"][0], ci95_high=r["ci95"][1]) for r in results])
    write_csv(out / "episode_returns.csv", episodes)
    report = dict(selection=dict(candidate_seeds=list(SEEDS), per="t, reversibility, condition",
        metric="Highest final logged training mean completed-episode return", held_out_results_used=False,
        tie_break="Lowest seed number", logged_return_decimal_places=3),
        uncertainty=dict(method="Ordinary unstratified episode-row percentile bootstrap of the mean",
            confidence=95, repetitions=1000, bootstrap_seed=args.bootstrap_seed,
            includes_training_seed_variability=False),
        shared_t0_baselines="Deduplicated historical irreversible baseline/V3 policies; no baseline replacement",
        manifests=list(provenance.values()), source_sha256=sha256(Path(__file__).read_bytes()),
        reference_script=str(ROOT / "eval/plot_overcooked_paper_match.py"), results=results,
        limitations=[
            "Intervals condition on the selected policy and omit training-seed and seed-selection uncertainty.",
            "Final logged training rollouts precede the last gradient update; held-out evaluation uses the final checkpoint.",
            "Training returns are rounded to three decimals and use condition-specific training partner populations.",
            "Episode rows are pooled without profile stratification or shared-rollout-seed clustering, matching the reference.",
            "Observation transitions consume the existing round budget and incur step costs.",
            "Historical irreversible and reversible policies retain their different PPO replay/reset alignment.",
        ])
    (out / "selection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot(results, out)
    plot(results, out, overlay_training=True)
    print(out, flush=True)


if __name__ == "__main__":
    main()
