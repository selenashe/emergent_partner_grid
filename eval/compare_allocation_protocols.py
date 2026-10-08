"""Compare saved v1/v2 evaluations, including counterbalanced training batches."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import h5py


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_path, resolve_record_paths
BATCH = "counterbalanced1096_20261002_235609"
INPUTS = {v: ROOT / "eval/eval_out" / f"{v}_balanced_training" / BATCH for v in ("v1", "v2")}
OUT = ROOT / "eval/protocol_comparison" / BATCH
COUNTERBALANCED = True
SEEDS = tuple(range(1, 6))
POPULATION = "test"
CONDITIONS = {
    "rnn_diverse_influence": "Diverse RNN\n+ influence",
    "mlp_diverse_influence": "Diverse MLP\n+ influence",
    "rnn_single_influence": "Single RNN\n+ influence",
    "rnn_diverse_noinfluence": "Diverse RNN\nno influence",
}


def main():
    # Audit guide:
    # Gather all requested policies per condition from the appropriate manifests and
    # summarize the requested population performance separately for fixed v1 and online v2. Plot
    # means and sample standard deviations over learner seeds. Original v1/v2
    # comparisons also change corpus size; counterbalanced comparisons share a corpus
    # and schedule design.
    #
    OUT.mkdir(parents=True, exist_ok=True)
    rows, trajectories, sources = [], {}, []
    for version, directory in INPUTS.items():
        trajectories[version] = {}
        for condition in CONDITIONS:
            data = []
            for seed in SEEDS:
                # An extension combines old and new runs without copying or overwriting
                # their results. Map each seed explicitly to its evaluation directory.
                seed_directory = Path(directory[str(seed)]) if isinstance(directory, dict) else directory
                path = seed_directory / f"{condition}_seed{seed}_summary.json"
                sources.append(str(path))
                d = json.loads(path.read_text())[POPULATION]
                expected_episodes = (24 if POPULATION == "train" else 22) * 20
                assert d["n_episodes"] == expected_episodes and d["n_rounds"] == expected_episodes * 20
                if version == "v2":
                    assert d["allocation_protocol"] == "online_v2"
                rollout_path = seed_directory / f"{condition}_seed{seed}_{POPULATION}.h5"
                sources.append(str(rollout_path))
                with h5py.File(rollout_path) as f:
                    dones = np.asarray(f["dones"], dtype=bool)
                    round_done = np.asarray(f["round_done"], dtype=bool)
                assert dones.shape[0] == expected_episodes and dones.any(axis=1).all()
                # The scan continues after termination; count only through
                # the first final done, including its terminal transition.
                episode_steps = dones.argmax(axis=1) + 1
                alive = np.arange(dones.shape[1])[None, :] < episode_steps[:, None]
                assert np.all((round_done & alive).sum(axis=1) == 20)
                mean_episode_steps = float(episode_steps.mean())
                # Every non-success transition costs .01, while success
                # transitions pay 1. Check against the saved return summary.
                inferred_steps = (1.01 * 20 * d["round_success_rate"] - d["mean_ep_return"]) / .01
                assert abs(mean_episode_steps - inferred_steps) < .02
                data.append(d)
                rows.append(dict(version=version, condition=condition, seed=seed,
                    success=d["round_success_rate"], episode_return=d["mean_ep_return"],
                    episode_steps=mean_episode_steps,
                    successful_completion_steps=d["mean_successful_completion_time"],
                    switches_per_round=d.get("assignment_switches_per_round", 0.0)))
            trajectories[version][condition] = np.array([d["per_round_success_rate"] for d in data]).tolist()
    with (OUT / "per_seed.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for version in INPUTS:
        for condition in CONDITIONS:
            rs = [r for r in rows if r["version"] == version and r["condition"] == condition]
            a = dict(version=version, condition=condition, n_seeds=len(rs))
            for metric in ("success", "episode_return", "episode_steps", "successful_completion_steps", "switches_per_round"):
                x = np.array([r[metric] for r in rs])
                a[metric + "_mean"] = float(x.mean())
                a[metric + "_sd"] = float(x.std(ddof=1))
            aggregates.append(a)
    report = dict(sources=sources, seeds=list(SEEDS), population=POPULATION, evaluation=f"{24 if POPULATION == 'train' else 22} capability profiles x 20 episodes x 20 rounds per trained seed; {len(SEEDS)} trained seeds per condition",
        caveats=["Layouts are familiar within each experiment; no layout holdout.",
                 "Both versions use the same frozen 1096 layouts and counterbalanced schedule.",
                 "v2 changes allocation masks, switch cooldowns, no-influence assignment and PPO recurrent replay resets.",
                 f"Standard deviations describe {len(SEEDS)} trained seeds, not confidence intervals or a causal comparison.",
                 "Successful completion time excludes failures; return includes their penalties.",
                 "Episode steps count environment transitions across all 20 rounds, including initialization and timeouts, through the first final done."],
        aggregates=aggregates, trajectories=trajectories)
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 4.7))
    colors = {"v1": "#5979ad", "v2": "#de8845"}
    def plot_metric(ax, metric, scale, label):
        for vi, version in enumerate(INPUTS):
            xs = np.arange(4) + (vi - 0.5) * 0.34
            aa = [a for a in aggregates if a["version"] == version]
            ax.bar(xs, [a[metric + "_mean"] * scale for a in aa], width=0.32,
                   color=colors[version], alpha=.8, label=version,
                   yerr=[a[metric + "_sd"] * scale for a in aa], capsize=4)
            for i, condition in enumerate(CONDITIONS):
                rs = [r for r in rows if r["version"] == version and r["condition"] == condition]
                ax.scatter(xs[i] + np.linspace(-.07, .07, len(rs)), [r[metric] * scale for r in rs],
                           s=17, color="black", alpha=.6, zorder=3)
        ax.set_xticks(np.arange(4), list(CONDITIONS.values()), fontsize=10)
        ax.set_ylabel(label)
        if metric != "episode_return" or min(r[metric] for r in rows) >= 0:
            ax.set_ylim(bottom=0)
        else:
            ax.axhline(0, color="black", linewidth=.7)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    for ax, metric, scale, label in zip(axes, ["success", "episode_return", "episode_steps"], [100, 1, 1],
            [f"{'Familiar' if POPULATION == 'train' else 'Novel'}-partner round success (%)", f"{'Familiar' if POPULATION == 'train' else 'Novel'}-partner episode return", "Mean steps per 20-round episode"]):
        plot_metric(ax, metric, scale, label)
    axes[0].legend(handles=[Patch(facecolor=colors["v1"], alpha=.8, label="v1: commit at round start"),
                           Patch(facecolor=colors["v2"], alpha=.8, label="v2: random start + switching")],
                   loc="lower left", fontsize=9)
    fig.suptitle(("Familiar partner performance: " if POPULATION == "train" else "Novel partner performance: ") + "counterbalanced v1 versus v2", fontsize=16)
    footnote = "Same 1096 layouts and counterbalanced schedule; v1/v2 protocols retained."
    fig.text(.5, .02, f"Dots: {len(SEEDS)} trained seeds. Error bars: sample SD. " + footnote, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"performance_comparison.{ext}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    plot_metric(ax, "episode_steps", 1, "Mean steps per 20-round episode")
    ax.legend(handles=[Patch(facecolor=colors["v1"], alpha=.8, label="v1: commit at round start"),
                       Patch(facecolor=colors["v2"], alpha=.8, label="v2: random start + switching")],
              loc="upper left", fontsize=9)
    ax.set_title(f"Steps taken with {'familiar' if POPULATION == 'train' else 'novel'} partners: v1 versus v2", fontsize=14)
    fig.text(.5, .02, f"Includes all 20 rounds and timeouts. Dots: {len(SEEDS)} trained seeds. Error bars: sample SD.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"episode_steps.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, condition in zip(axes.flat, CONDITIONS):
        for version in INPUTS:
            a = np.array(trajectories[version][condition]) * 100
            ax.plot(np.arange(1, 21), a.mean(axis=0), color=colors[version], label=version)
            ax.fill_between(np.arange(1, 21), a.mean(axis=0) - a.std(axis=0, ddof=1),
                            a.mean(axis=0) + a.std(axis=0, ddof=1), color=colors[version], alpha=.12)
        ax.set_title(CONDITIONS[condition].replace("\n", " "))
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.supxlabel(f"Round within a {'familiar' if POPULATION == 'train' else 'novel'}-partner episode (1–20)")
    fig.supylabel("Round success (%)")
    fig.suptitle("Success across rounds; shaded bands show sample SD across seeds")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"round_comparison.{ext}", dpi=180)
    print(json.dumps(aggregates, indent=2))
    print(f"Saved comparison to {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterbalanced-batch", help="Batch under eval/eval_out/{v1,v2}_balanced_training")
    parser.add_argument("--extension-manifest", type=Path, help="Explicit original/new evaluation directories for all ten seeds")
    parser.add_argument("--population", choices=("train", "test"), default="test")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    POPULATION = args.population
    if args.extension_manifest:
        if args.counterbalanced_batch:
            parser.error("Choose either a batch or an extension manifest")
        manifest = resolve_record_paths(json.loads(args.extension_manifest.read_text()))
        if manifest.get("action_selection", "categorical") != "categorical":
            parser.error("Seed extensions must retain original categorical sampling")
        SEEDS = tuple(manifest["evaluation_seeds"])
        if len(SEEDS) < 2 or len(SEEDS) != len(set(SEEDS)):
            parser.error("At least two distinct learner seeds are required")
        INPUTS = manifest["evaluation_inputs"]
        COUNTERBALANCED = True
        OUT = Path(manifest["aggregate_root"]) / ("familiar" if POPULATION == "train" else "novel")
    if args.counterbalanced_batch:
        BATCH = args.counterbalanced_batch
        COUNTERBALANCED = True
        INPUTS = {v: resolve_path(ROOT / "eval/eval_out" / f"{v}_balanced_training" / BATCH) for v in ("v1", "v2")}
        OUT = resolve_path(ROOT / "eval/protocol_comparison" / BATCH)
        manifest = json.loads(resolve_path(ROOT / "train/manifests" / f"sbatch_{BATCH}.json").read_text())
        if manifest.get("sampling_protocol") != "paired_counterbalanced_v1" or manifest.get("action_selection", "categorical") != "categorical":
            parser.error("Expected a categorical counterbalanced batch")
    if args.out_dir:
        OUT = args.out_dir
    main()
