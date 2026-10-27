"""Plot completed-round return prefixes from saved post-training evaluations.

Training-profile plots use each condition's policy-training population, so
single-partner RNN episodes are restricted to (1,4). A supplementary shared
24-profile view preserves the population used by the familiar evaluation.
Error bars describe pooled episode SD, not uncertainty in the plotted mean.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.plot_episode_return_dots import CONDITION_STYLES
from repo_paths import resolve_record_paths

VERSIONS = ("v1_matched", "v2", "v3")


def extract_round_returns(rewards, dones, round_done, round_index, round_time, success):
    """Return full-round and cumulative returns, masking after the first done."""
    if not dones.any(axis=1).all():
        raise ValueError("Every saved episode must have a final done")
    lengths = dones.argmax(axis=1) + 1
    alive = np.arange(dones.shape[1])[None, :] < lengths[:, None]
    terminal = round_done & alive
    if not np.all(terminal.sum(axis=1) == 20):
        raise ValueError("Every episode must contain exactly 20 completed rounds")
    ends = np.nonzero(terminal)[1].reshape(len(lengths), 20)
    episodes = np.arange(len(lengths))[:, None]
    if not np.array_equal(round_index[episodes, ends], np.tile(np.arange(20), (len(lengths), 1))):
        raise ValueError("Round ends must be ordered from round 0 through 19")
    if not np.array_equal(ends[:, -1], lengths - 1):
        raise ValueError("The twentieth round must end at the first final done")
    steps = np.diff(ends + 1, prepend=np.zeros((len(lengths), 1), dtype=int), axis=1)
    if not np.array_equal(steps, round_time[episodes, ends]) or not np.all((steps >= 1) & (steps <= 100)):
        raise ValueError("Round durations do not match saved round-local time")
    if np.any(success & alive & ~terminal):
        raise ValueError("Success must terminate its round")
    if not np.isfinite(rewards[alive]).all():
        raise ValueError("Valid rewards must be finite")
    np.testing.assert_allclose(rewards[alive], np.where(success[alive], 1., -.03), atol=1e-6, rtol=0)
    reward_prefix = np.cumsum(np.where(alive, rewards, 0), axis=1, dtype=np.float64)
    cumulative = reward_prefix[episodes, ends]
    incremental = np.diff(cumulative, prepend=np.zeros((len(lengths), 1)), axis=1)
    expected = 1.03 * success[episodes, ends] - .03 * steps
    np.testing.assert_allclose(incremental, expected, atol=2e-6, rtol=0)
    return cumulative, incremental, steps, success[episodes, ends]


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_evaluations(manifest):
    groups, sources = {}, []
    completed = 0
    for population in ("train", "test"):
        expected_episodes = 480 if population == "train" else 440
        expected_profiles = 24 if population == "train" else 22
        for version in VERSIONS:
            for condition, _, _ in CONDITION_STYLES:
                batches = []
                for seed in manifest["evaluation_seeds"]:
                    directory = Path(manifest["evaluation_inputs"][version][str(seed)])
                    prefix = directory / f"{condition}_seed{seed}"
                    path = Path(str(prefix) + f"_{population}.h5")
                    summary_path = Path(str(prefix) + "_summary.json")
                    summary = json.loads(summary_path.read_text())[population]
                    with h5py.File(path) as file:
                        arrays = {name: file[name][:] for name in (
                            "rewards", "dones", "round_done", "round_idx", "round_time", "success",
                            "capability_pool", "capability_index_per_ep")}
                    if arrays["dones"].shape != (expected_episodes, 2000):
                        raise ValueError(f"Unexpected episode count/scan shape: {path}")
                    if arrays["capability_pool"].shape != (expected_profiles, 2):
                        raise ValueError(f"Unexpected profile population: {path}")
                    counts = np.bincount(arrays["capability_index_per_ep"], minlength=expected_profiles)
                    np.testing.assert_array_equal(counts, np.full(expected_profiles, 20))
                    cumulative, incremental, steps, success = extract_round_returns(
                        arrays["rewards"], arrays["dones"].astype(bool), arrays["round_done"].astype(bool),
                        arrays["round_idx"], arrays["round_time"], arrays["success"].astype(bool))
                    np.testing.assert_allclose(cumulative[:, -1].mean(), summary["mean_ep_return"], atol=3e-3, rtol=0)
                    np.testing.assert_allclose(success.mean(axis=0), summary["per_round_success_rate"], atol=1e-7, rtol=0)
                    if summary["n_episodes"] != expected_episodes or summary["n_rounds"] != 20 * expected_episodes:
                        raise ValueError(f"Summary episode counts differ: {summary_path}")
                    batches.append({
                        "cumulative_return": cumulative, "round_return": incremental,
                        "round_steps": steps, "round_success": success,
                        "training_seed": np.full(expected_episodes, seed, dtype=np.int16),
                        "source_episode": np.arange(expected_episodes, dtype=np.int16),
                        "capability": arrays["capability_pool"][arrays["capability_index_per_ep"]],
                    })
                    digest = hashlib.sha256()
                    for name, array in arrays.items():
                        digest.update(name.encode())
                        digest.update(np.ascontiguousarray(array).tobytes())
                    sources.append({"population": population, "version": version, "condition": condition,
                                    "seed": seed, "rollout": str(path), "bytes": path.stat().st_size,
                                    "mtime_ns": path.stat().st_mtime_ns,
                                    "used_datasets_sha256": digest.hexdigest(), "summary": str(summary_path),
                                    "saved_mean_return": summary["mean_ep_return"],
                                    "reconstructed_mean_return": float(cumulative[:, -1].mean())})
                    completed += 1
                    if completed % 10 == 0:
                        print(f"Validated {completed}/120 rollout files", flush=True)
                groups[(population, version, condition)] = {
                    field: np.concatenate([b[field] for b in batches], axis=0) for field in batches[0]
                }
    return groups, sources


def make_views(groups):
    views = {"training_profiles": {}, "heldout_profiles": {}, "shared_training_profiles": {}}
    for (population, version, condition), data in groups.items():
        key = (version, condition)
        if population == "test":
            views["heldout_profiles"][key] = data
        else:
            views["shared_training_profiles"][key] = data
            if condition == "rnn_single_influence":
                selected = np.all(data["capability"] == [1, 4], axis=1)
                if selected.sum() != 200:
                    raise ValueError("Single-partner view must have 20 (1,4) episodes per seed")
                views["training_profiles"][key] = {name: value[selected] for name, value in data.items()}
            else:
                views["training_profiles"][key] = data
    return views


def statistics(views):
    rows, per_seed = [], []
    lookup = {}
    for view, groups in views.items():
        for (version, condition), data in groups.items():
            seed_values = {}
            for seed in range(1, 6):
                mask = data["training_seed"] == seed
                if not mask.any():
                    raise ValueError("Every plotted condition must retain all five policy seeds")
                seed_values[seed] = {metric: data[metric][mask] for metric in ("cumulative_return", "round_return")}
                for r in range(20):
                    row = {"view": view, "version": version, "condition": condition, "seed": seed,
                           "round": r+1, "n_episodes": int(mask.sum())}
                    for metric, values in seed_values[seed].items():
                        row[metric + "_mean"] = float(values[:, r].mean())
                        row[metric + "_episode_sd"] = float(values[:, r].std(ddof=1))
                    per_seed.append(row)
            for r in range(20):
                row = {"view": view, "version": version, "condition": condition, "round": r+1,
                       "n_episodes": len(data["training_seed"]), "n_policy_seeds": 10}
                for metric in ("cumulative_return", "round_return"):
                    values = data[metric][:, r]
                    row[metric + "_mean"] = float(values.mean())
                    row[metric + "_episode_variance"] = float(values.var(ddof=1))
                    row[metric + "_episode_sd"] = float(values.std(ddof=1))
                    seed_means = [seed_values[s][metric][:, r].mean() for s in range(1, 6)]
                    row[metric + "_seed_mean_sd"] = float(np.std(seed_means, ddof=1))
                    numerator = sum((len(seed_values[s][metric])-1) * seed_values[s][metric][:, r].var(ddof=1)
                                    for s in range(1, 6))
                    row[metric + "_within_seed_episode_sd"] = float(np.sqrt(numerator/(len(values)-10)))
                rows.append(row)
                lookup[(view, version, condition, r+1)] = row
    return rows, per_seed, lookup


def plot(view, metric, lookup, out):
    titles = {"training_profiles": "Each condition's training partner profiles",
              "heldout_profiles": "Held-out partner profiles",
              "shared_training_profiles": "Shared 24-profile familiar evaluation"}
    subtitles = {
        "training_profiles": "Single-partner RNN: (1,4), 200 episodes. Other models: 24 profiles, 2,400 episodes each per protocol.",
        "heldout_profiles": "All models: 22 held-out profiles, 2,200 episodes each per protocol. Grid layouts are familiar.",
        "shared_training_profiles": "All models: 24 training profiles, 2,400 episodes each per protocol, including the single-partner-trained RNN.",
    }
    label = "Cumulative episode return" if metric == "cumulative_return" else "Return earned in one round"
    x = np.arange(1, 21)
    lower = min(p[metric + "_mean"]-p[metric + "_episode_sd"] for p in lookup.values())
    upper = max(p[metric + "_mean"]+p[metric + "_episode_sd"] for p in lookup.values())
    lower = min(0., lower)
    padding = .04 * (upper-lower)
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharex=True, sharey=True)
        fig.subplots_adjust(left=.065, right=.985, bottom=.17, top=.70, wspace=.13)
        for ax, version in zip(axes, VERSIONS):
            for condition, legend_label, color in CONDITION_STYLES:
                points = [lookup[(view, version, condition, r)] for r in x]
                mean = np.array([p[metric + "_mean"] for p in points])
                sd = np.array([p[metric + "_episode_sd"] for p in points])
                ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor=color, alpha=.25,
                            elinewidth=.7, capsize=1.8, capthick=.7, zorder=1)
                ax.plot(x, mean, color=color, lw=2.1, marker="o", markersize=2.5,
                        label=legend_label, zorder=3)
            ax.set_title(version, fontsize=16, fontweight="bold", pad=12)
            ax.set_xlim(.5, 20.5)
            ax.set_ylim(lower-padding, upper+padding)
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.set_xlabel("Round within partner episode")
            ax.grid(alpha=.16)
            ax.axhline(0, color="#777777", lw=.6, zorder=0)
        axes[0].set_ylabel(label)
        title = "Cumulative episode return across rounds" if metric == "cumulative_return" else "Return earned per round"
        fig.text(.065, .95, f"{title} · {titles[view]}", va="top", fontsize=18, weight="bold")
        fig.text(.065, .89, subtitles[view], va="top", fontsize=11, color="#555555")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4,
                   bbox_to_anchor=(.52, .83), frameon=False)
        fig.text(.065, .055, "Means weight episodes equally across 10 learner seeds. Error bars: ±1 pooled episode sample SD, including differences between seeds.",
                 fontsize=10, color="#555555")
        detail = ("Cumulative SD is computed from episode prefixes, preserving correlations between rounds."
                  if metric == "cumulative_return" else
                  "Round returns include initialization ticks, waiting, successful terminal rewards, and timeouts.")
        fig.text(.065, .025, "Saved post-training evaluations; each episode has 20 rounds. " + detail,
                 fontsize=9.5, color="#555555")
        stem = f"{view}_{metric}"
        for extension in ("png", "pdf", "svg"):
            fig.savefig(out / f"{stem}.{extension}", dpi=180, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "train/manifests/sbatch_v1_matched_20261009_182845.json")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--replot", action="store_true", help="Render verified saved summaries without rereading rollout files")
    args = parser.parse_args()
    manifest = resolve_record_paths(json.loads(args.manifest.read_text()))
    if manifest["evaluation_seeds"] != list(range(1, 6)):
        raise ValueError("Expected the same ten evaluated learner seeds")
    out = (args.out_dir or Path(manifest["aggregate_root"]) / "round_returns").resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        metadata_path = out / "analysis_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if not metadata["passed"]:
            raise ValueError("Cached rollout analysis must have passed its checks")
        with (out / "round_summary.csv").open() as file:
            rows = list(csv.DictReader(file))
        lookup = {}
        for row in rows:
            for key in row:
                if key not in ("view", "version", "condition"):
                    row[key] = float(row[key])
            lookup[(row["view"], row["version"], row["condition"], int(row["round"]))] = row
        for view in dict.fromkeys(row["view"] for row in rows):
            for metric in ("cumulative_return", "round_return"):
                plot(view, metric, lookup, out)
        metadata["plot_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        metadata["plot_axes"] = "Each metric uses the same y-axis range across all population views and protocols."
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Replotted verified summaries: {out}")
        return
    groups, sources = read_evaluations(manifest)
    views = make_views(groups)
    rows, per_seed, lookup = statistics(views)
    write_csv(out / "round_summary.csv", rows)
    write_csv(out / "round_per_seed.csv", per_seed)
    # Persist the reconstructed episode prefixes once; filtered views can be regenerated.
    arrays_dir = out / "per_episode"
    arrays_dir.mkdir(exist_ok=True)
    for (population, version, condition), data in groups.items():
        np.savez_compressed(arrays_dir / f"{population}_{version}_{condition}.npz", **data)
    for view in views:
        for metric in ("cumulative_return", "round_return"):
            plot(view, metric, lookup, out)
    result = {
        "passed": True, "manifest": str(args.manifest.resolve()), "versions": list(VERSIONS),
        "learner_seeds": list(range(1, 6)), "rollout_files": len(sources),
        "unique_saved_episodes": sum(len(g["training_seed"]) for g in groups.values()),
        "cumulative_return": "Sum of actual rewards from episode start through the terminal transition of round r.",
        "round_return": "Reward earned within round r; cumulative prefix differences.",
        "episode_variability": "Pooled sample variance/SD across episodes and ten policies; includes within-policy episode variability and between-policy differences. This is not a confidence interval or SD of policy means.",
        "weighting": "Every episode has equal weight within its view; equal episode counts per policy give equal learner-seed weight. Each profile has 20 episodes per policy.",
        "training_profiles": "Policy-training population: (1,4) for single-partner RNN; all 24 profiles for the other conditions. Saved post-training evaluations, not trajectory logs from PPO optimization.",
        "shared_training_profiles": "Supplementary familiar view: all 24 training profiles for every condition, matching the existing familiar comparison.",
        "heldout_profiles": "All 22 held-out profiles for every condition; same familiar 1,096-layout grid corpus.",
        "checks": "First-final-done mask; exactly 20 ordered round ends; duration/reward formula agreement; 20 episodes per profile; all saved mean returns and per-round success rates reconstructed.",
        "sources": sources,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "analysis_metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(out), "saved_episodes": result["unique_saved_episodes"], "rollout_files": len(sources)}, indent=2))


if __name__ == "__main__":
    main()
