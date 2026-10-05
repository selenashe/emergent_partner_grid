"""Plot held-out evaluations after selecting each condition's best training seed.

Default selection: highest mean completed-episode return in the final logged
PPO rollout (before its gradient update). Never select on held-out results.
An explicit trailing window can be requested; logged updates have equal weight
because counts of completed episodes were not logged.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_path, resolve_record_paths
from eval.compare_allocation_protocols import CONDITIONS

LOG_LINE = re.compile(
    r"^\[u\s+(\d+)\]\s+env=\s*(\d+)\s+rsucc=([\d.]+)"
    r"\s+n_rounds=\s*(\d+)\s+ret=([+\-\d.]+)\s+len=\s*([\d.]+)\s"
)
COLORS = {"v1": "#5979ad", "v2": "#de8845"}
METRICS = ("success", "episode_return", "episode_steps")


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_seeds(manifest, window):
    candidates = []
    for job in manifest["training_jobs"]:
        version, condition, seed = job["version"], job["condition"], job["seed"]
        log = (Path(job["log"]) if "log" in job else
               ROOT / "train/slurm_logs" / f"cg_cap_{job['job_id']}.out")
        log = resolve_path(log)
        content = log.read_bytes()
        records = []
        for line in content.decode().splitlines():
            if line.startswith("[u "):
                match = LOG_LINE.match(line)
                if not match:
                    raise ValueError(f"Unrecognized rollout metric in {log}: {line}")
                records.append(tuple(float(v) for v in match.groups()))
        config_path = (Path(job["config"]) if "config" in job else
                       Path(manifest["versions"][version]["checkpoint_root"])
                       / f"{job['tag']}_config.json")
        config = json.loads(config_path.read_text())
        n_updates = int(config["NUM_UPDATES"])
        assert len(records) == n_updates
        assert [int(row[0]) for row in records] == list(range(1, n_updates + 1))
        assert int(records[-1][1]) == manifest["effective_timesteps_per_policy"]
        assert config["SEED"] == seed and config["NUM_SEEDS"] == 1
        valid_window = [row for row in records[-window:] if row[5] > 0]
        if window > n_updates or not valid_window:
            raise ValueError("Selection window exceeds available completed-episode metrics")
        candidates.append(dict(version=version, condition=condition, seed=seed,
            training_return=float(np.mean([row[4] for row in valid_window])),
            final_logged_return=records[-1][4],
            final_46_update_return=float(np.mean([row[4] for row in records[-46:] if row[5] > 0])),
            selection_window_updates=window, selection_valid_updates=len(valid_window), final_update=n_updates,
            environment_steps=int(records[-1][1]), training_log=str(log),
            training_log_sha256=hashlib.sha256(content).hexdigest(),
            config=str(config_path)))
    selected = []
    for version in ("v1", "v2"):
        for condition in CONDITIONS:
            group = [r for r in candidates if r["version"] == version
                     and r["condition"] == condition]
            assert sorted(r["seed"] for r in group) == list(range(1, 6))
            # Returns in stdout have three decimals; break any logged tie by seed.
            selected.append(max(group, key=lambda r: (r["training_return"], -r["seed"])))
    return candidates, selected


def held_out_metrics(directory, condition, seed, n_bootstrap, bootstrap_seed):
    summary_path = directory / f"{condition}_seed{seed}_summary.json"
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)["test"]
    rollout_path = directory / f"{condition}_seed{seed}_test.h5"
    with h5py.File(rollout_path) as file:
        arrays = {key: np.asarray(file[key]) for key in (
            "dones", "round_done", "success", "rewards", "round_idx",
            "capability_index_per_ep", "capability_pool")}
    dones = arrays["dones"]
    assert dones.shape[0] == summary["n_episodes"] == 440
    assert dones.any(axis=1).all()
    steps = dones.argmax(axis=1) + 1
    alive = np.arange(dones.shape[1])[None, :] < steps[:, None]
    round_done = arrays["round_done"] & alive
    assert np.all(round_done.sum(axis=1) == 20)
    success = arrays["success"] & round_done
    by_round = np.stack([
        (success & (arrays["round_idx"] == r)).sum(axis=1) for r in range(20)
    ], axis=1)
    assert np.all((by_round == 0) | (by_round == 1))
    episode_return = (arrays["rewards"] * alive).sum(axis=1)
    data = np.column_stack([success.sum(axis=1) / 20, episode_return, steps, by_round])
    means = data.mean(axis=0)
    assert np.isclose(means[0], summary["round_success_rate"])
    assert np.isclose(means[1], summary["mean_ep_return"], atol=3e-5)
    assert np.allclose(means[3:], summary["per_round_success_rate"], atol=1e-7)
    assert abs(means[2] - (1.01 * 20 * means[0] - means[1]) / .01) < .02
    profiles = arrays["capability_index_per_ep"]
    assert len(np.unique(profiles)) == len(arrays["capability_pool"]) == 22
    rng = np.random.default_rng(bootstrap_seed)
    boot = np.zeros((n_bootstrap, data.shape[1]))
    # Resample whole episodes within each fixed profile. Keep all 20 rounds
    # together and preserve equal weights for the 22 held-out profiles.
    for profile in np.unique(profiles):
        group = data[profiles == profile]
        assert len(group) == 20
        indices = rng.integers(len(group), size=(n_bootstrap, len(group)))
        boot += group[indices].mean(axis=1) / 22
    intervals = np.percentile(boot, [2.5, 97.5], axis=0)
    result = {metric: dict(mean=float(means[i]), ci95=intervals[:, i].tolist())
              for i, metric in enumerate(METRICS)}
    result.update(round_success=means[3:].tolist(),
        round_success_ci95=intervals[:, 3:].tolist(),
        evaluation_summary=str(summary_path),
        evaluation_summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
        evaluation_rollouts=str(rollout_path),
        metric_datasets_sha256=hashlib.sha256(b"".join(
            arrays[key].tobytes() for key in sorted(arrays))).hexdigest(),
        n_episodes=440, n_rounds=8800, n_held_out_profiles=22)
    return result


def save_figure(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=180)
    plt.close(fig)


def make_plots(results, out, window, counterbalanced=True, versions=("v1", "v2"),
               performance_only=False, common_seed=None, axis_limits=None,
               selection_scope="both experiment sets", action_selection=None):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    legend = [Patch(facecolor=COLORS[v], alpha=.85, label=label) for v, label in (
        ("v1", "v1: commit at round start"), ("v2", "v2: random start + switching"))
        if v in versions]

    def metric_plot(ax, metric, scale, ylabel):
        for vi, version in enumerate(versions):
            selected = [r for r in results if r["version"] == version]
            xs = np.arange(4) + (vi - (len(versions) - 1) / 2) * .34
            means = np.array([r[metric]["mean"] for r in selected]) * scale
            ci = np.array([r[metric]["ci95"] for r in selected]).T * scale
            ax.bar(xs, means, width=.32 if len(versions) == 2 else .55,
                   color=COLORS[version], alpha=.85,
                   yerr=np.maximum(0, np.stack([means - ci[0], ci[1] - means])), capsize=4)
            for x, upper, r in zip(xs, ci[1], selected):
                ax.annotate(f"s{r['seed']}", (x, upper), xytext=(0, 5),
                            textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(np.arange(4), list(CONDITIONS.values()), fontsize=10)
        ax.set_ylabel(ylabel)
        # Same scale on separate v1/v2 panels, so heights remain comparable.
        upper = max(0, max(r[metric]["ci95"][1] for r in results) * scale)
        lower = min(0, min(r[metric]["ci95"][0] for r in results) * scale)
        span = max(upper - lower, 1)
        limit = axis_limits[metric] if axis_limits else upper + span * .15
        ax.set_ylim(lower - span * .08 if lower < 0 else 0, limit)
        if lower < 0:
            ax.axhline(0, color="black", linewidth=.7)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)

    rule = ("highest final logged training return" if window == 1 else
            f"highest trailing {window}-update training return")
    corpus_note = ("Same 1096 layouts." if counterbalanced else
                   "Original random sampling; v1: 1000 layouts; v2: 1096 layouts.")
    selection_note = ("One policy per condition, selected by " + rule + "." if common_seed is None
                     else f"Seed {common_seed} for every condition/version; selected by mean " +
                          ("final logged training return" if window == 1 else f"trailing {window}-update training return") +
                          f" across {selection_scope}.")
    footnote = (selection_note + "\n"
                "Error bars: 95% episode-bootstrap CIs within held-out profiles. "
                "Labels: selected seed.\n" + corpus_note)
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.1))
    for ax, metric, scale, label in zip(axes, METRICS, (100, 1, 1), (
            "Held-out partner round success (%)", "Held-out partner episode return",
            "Mean steps per 20-round episode")):
        metric_plot(ax, metric, scale, label)
    axes[0].legend(handles=legend,
                   loc="upper left" if action_selection == "greedy_random_ties" else "lower left", fontsize=9)
    title = ("Counterbalanced" if counterbalanced else "Original random-sampling")
    if action_selection == "greedy_random_ties":
        title += " greedy-action"
    title += " " + ("v1 versus v2" if len(versions) == 2 else versions[0])
    seed_label = "best training seed" if common_seed is None else f"common training seed {common_seed}"
    fig.suptitle(title + ": " + seed_label, fontsize=16)
    fig.text(.5, .02, footnote, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .95))
    suffix = "" if len(versions) == 2 else "_" + versions[0]
    save_figure(fig, out, "performance_comparison" + suffix)
    if performance_only:
        return
    fig, ax = plt.subplots(figsize=(9, 5.8))
    metric_plot(ax, "episode_steps", 1, "Mean steps per 20-round episode")
    ax.legend(handles=legend, loc="upper left", fontsize=9)
    ax.set_title("Held-out episode steps: " + seed_label)
    fig.text(.5, .02, "Includes timeouts and all 20 rounds.\n" + footnote, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .13, 1, 1))
    save_figure(fig, out, "episode_steps")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, condition in zip(axes.flat, CONDITIONS):
        for version in ("v1", "v2"):
            r = next(r for r in results if r["version"] == version and r["condition"] == condition)
            ci = np.array(r["round_success_ci95"]) * 100
            ax.plot(np.arange(1, 21), np.array(r["round_success"]) * 100,
                    color=COLORS[version], label=f"{version}: seed {r['seed']}")
            ax.fill_between(np.arange(1, 21), ci[0], ci[1], color=COLORS[version], alpha=.15)
        ax.set_title(CONDITIONS[condition].replace("\n", " "))
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=.2)
        ax.set_xticks([1, 5, 10, 15, 20])
    fig.supxlabel("Round within a held-out partner episode (1–20)")
    fig.supylabel("Round success (%)")
    fig.suptitle(seed_label.capitalize() + ": held-out success across rounds")
    fig.text(.5, .045, "Shading: pointwise 95% episode-bootstrap CIs within partner profiles.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(.015, .09, 1, .96))
    save_figure(fig, out, "round_comparison")


def load_source(args):
    if not args.original_versions:
        path = resolve_path(args.manifest)
        content = path.read_bytes()
        return path, content, resolve_record_paths(json.loads(content))
    # The original v1 runs predate the v2 submission manifest; this audit
    # records the precise logs/configs of both sets of evaluated checkpoints.
    batch = "balanced1096_20261002_022924"
    path = ROOT / "eval/protocol_comparison" / batch / "final_training_grid_statistics.json"
    content = path.read_bytes()
    audit = resolve_record_paths(json.loads(content))
    runs = audit["per_seed_training_rounds"]
    assert len(runs) == 40 and len({r["steps"] for r in runs}) == 1
    source = dict(batch=batch, training_jobs=runs,
        effective_timesteps_per_policy=runs[0]["steps"],
        sampling_protocol="uniform_with_replacement",
        corpus_note="Original v1: 1000 layouts; original v2: 1096 layouts. Protocol and other implementation details also differ.",
        corpora=audit["corpora"],
        versions={"v1": dict(evaluation_root=str(ROOT / "eval/eval_out")),
                  "v2": dict(evaluation_root=str(ROOT / "eval/eval_out/online_v2" / batch))})
    return path, content, source


def main(args):
    manifest_path, manifest_bytes, manifest = load_source(args)
    candidates, selected = select_seeds(manifest, args.selection_window_updates)
    # All eight selections are finalized before opening held-out evaluations.
    results = []
    for index, row in enumerate(selected):
        directory = Path(manifest["versions"][row["version"]]["evaluation_root"])
        results.append({**row, **held_out_metrics(directory, row["condition"], row["seed"],
                       args.bootstrap_repetitions, args.bootstrap_seed + index)})
    out = args.output_dir or resolve_path(ROOT / "eval/protocol_comparison" / manifest["batch"]) / "best_training_seed"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "training_seed_scores.csv", candidates)
    write_csv(out / "selected_seed_metrics.csv", [dict(
        version=r["version"], condition=r["condition"], seed=r["seed"],
        training_return=r["training_return"], **{
            key: r[metric][field] if field == "mean" else r[metric]["ci95"][idx]
            for metric in METRICS for key, field, idx in (
                (metric, "mean", 0), (metric + "_ci95_low", "ci95", 0),
                (metric + "_ci95_high", "ci95", 1))}) for r in results])
    report = dict(batch=manifest["batch"], source_record=str(manifest_path),
        source_record_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        sampling_protocol=manifest["sampling_protocol"],
        corpus_note=manifest["corpus_note"],
        selection=dict(metric="training completed-episode return", window_updates=args.selection_window_updates,
            weights="Equal weight per logged PPO rollout", held_out_results_used=False,
            tie_break="Lowest seed number for equal logged scores", logged_return_decimal_places=3,
            note="Final logged rollout precedes the last PPO gradient update; evaluation uses the final saved checkpoint."),
        uncertainty=dict(method="Percentile bootstrap of whole episodes independently within each of 22 fixed partner profiles",
            repetitions=args.bootstrap_repetitions, seed=args.bootstrap_seed,
            confidence=.95, includes_training_seed_variability=False,
            scope="Conditional on the selected policy and fixed partner profiles; does not include seed-selection uncertainty."),
        selected=results, candidates=candidates)
    (out / "selection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    make_plots(results, out, args.selection_window_updates,
               counterbalanced=not args.original_versions)
    for version in ("v1", "v2"):
        make_plots(results, out, args.selection_window_updates,
                   counterbalanced=not args.original_versions, versions=(version,),
                   performance_only=True)
    for r in results:
        print(f"{r['version']} {r['condition']}: seed {r['seed']}; "
              f"train return {r['training_return']:.3f}; held-out success "
              f"{100*r['success']['mean']:.2f}%; return {r['episode_return']['mean']:.3f}; "
              f"steps {r['episode_steps']['mean']:.1f}")
    print(f"Saved plots and selection records to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--manifest", type=Path, default=ROOT / "train/manifests/sbatch_counterbalanced1096_20261002_235609.json")
    inputs.add_argument("--original-versions", action="store_true",
                        help="Use the original random-sampling v1/v2 runs instead of the counterbalanced batch")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selection-window-updates", type=int, default=1)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    if args.selection_window_updates < 1 or args.bootstrap_repetitions < 100:
        parser.error("Selection window must be positive; use at least 100 bootstrap repetitions")
    main(args)
