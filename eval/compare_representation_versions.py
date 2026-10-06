"""Plot counterbalanced v1/v2 decoding for every training seed and their means.

Consumes completed representation_analysis.py outputs; does not refit probes.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.representation_analysis import (
    CONDITIONS, CONDITION_LABELS, TARGETS, plotting_style, require, save_figure,
)

COLORS = ("#0072B2", "#D55E00", "#009E73")
VERSIONS = ("v1_balanced_training", "v2_balanced_training")
VERSION_LABELS = ("v1: commit at round start", "v2: random start, can switch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="eval/representation_results/counterbalanced1096_20261002_235609")
    args = parser.parse_args()
    root = Path(args.results_root)
    out_dir = root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = plotting_style()
    inputs, splits = {}, []
    for version in VERSIONS:
        folder = root / version
        metadata = json.loads((folder / "analysis_metadata.json").read_text())
        require(metadata["n_checkpoints"] == 15 and metadata["training_seeds"] == list(range(1, 6)),
                "Each version must include all fifteen RNN checkpoints")
        expected_protocol = "fixed_v1" if version.startswith("v1") else "online_v2"
        require(metadata["arguments"]["allocation_protocol"] == expected_protocol,
                f"Unexpected protocol for {version}")
        splits.append(json.loads((folder / "probe_split.json").read_text()))
        inputs[version] = {
            "time": pd.read_csv(folder / "probe_timestep_per_seed.csv"),
            "round": pd.read_csv(folder / "probe_by_round_per_seed.csv"),
            "time_summary": pd.read_csv(folder / "probe_timestep_summary.csv"),
            "round_summary": pd.read_csv(folder / "probe_by_round_summary.csv"),
            "random": pd.read_csv(folder / "random_baseline.csv"),
        }
    require(splits[0] == splits[1], "Versions must use the same probe train/test split")
    endpoints = []
    for axis, x, start, end, xlabel in (
        ("time", "reference_t", 1, 400, "Episode timestep"),
        ("round", "round_idx", 0, 19, "Completed round (1–20)"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True, layout="constrained")
        for row, (version, version_label) in enumerate(zip(VERSIONS, VERSION_LABELS)):
            tables = inputs[version]
            summary = tables[f"{axis}_summary"]
            raw = tables[axis]
            for col, target in enumerate(TARGETS):
                ax = axes[row, col]
                for condition, color in zip(CONDITIONS, COLORS):
                    group = summary[(summary.condition == condition) & (summary.target == target)
                                    & (summary.capability_subset == "all")].sort_values(x)
                    require((group.n_training_seeds == 5).all(), "A plotted mean is missing policy seeds")
                    xx = group[x].to_numpy() + (1 if axis == "round" else 0)
                    ax.plot(xx, group.distance_accuracy_mean, label=CONDITION_LABELS[condition],
                            color=color, marker="o", ms=3, lw=2)
                    ax.fill_between(xx, group.distance_accuracy_ci_low, group.distance_accuracy_ci_high,
                                    color=color, alpha=.16, lw=0)
                baseline = tables["random"]
                baseline = baseline[(baseline.target == target) & (baseline.capability_subset == "all")]
                ax.axhline(baseline.distance_accuracy.mean(), ls="--", color="#555555", lw=1.5,
                           label="Random Normal baseline")
                ax.set(title=f"{version_label}\n{'Red' if target == 'd_R' else 'Blue'} delay ({target})",
                       xlabel=xlabel, ylim=(.45, 1.01))
                ax.grid(axis="y", alpha=.2)
                if col == 0:
                    ax.set_ylabel("Distance-aware test accuracy")
                if axis == "round":
                    ax.set_xticks([1, 5, 10, 15, 20])
            # Individual policies remain visible: five rows, two decoded delays.
            seed_fig, seed_axes = plt.subplots(5, 2, figsize=(11, 14), sharey=True, sharex=True,
                                              layout="constrained")
            for seed in range(1, 6):
                for col, target in enumerate(TARGETS):
                    ax = seed_axes[seed - 1, col]
                    for condition, color in zip(CONDITIONS, COLORS):
                        group = raw[(raw.condition == condition) & (raw.training_seed == seed)
                                    & (raw.target == target) & (raw.capability_subset == "all")].sort_values(x)
                        require(len(group) == (9 if axis == "time" else 20), "Incomplete per-seed curve")
                        xx = group[x].to_numpy() + (1 if axis == "round" else 0)
                        ax.plot(xx, group.distance_accuracy, label=CONDITION_LABELS[condition],
                                color=color, lw=1.7, marker="o", ms=2.5)
                    baseline = tables["random"]
                    baseline = baseline[(baseline.target == target) & (baseline.capability_subset == "all")]
                    ax.axhline(baseline.distance_accuracy.mean(), ls="--", color="#555555", lw=1.3,
                               label="Random Normal baseline")
                    ax.set(title=f"Seed {seed}: {'red' if target == 'd_R' else 'blue'} delay", ylim=(.45, 1.01))
                    ax.grid(axis="y", alpha=.2)
                    if col == 0:
                        ax.set_ylabel("Distance-aware test accuracy")
                    if seed == 5:
                        ax.set_xlabel(xlabel)
            seed_axes[0, 1].legend(fontsize=8, loc="lower right", frameon=False)
            seed_fig.suptitle(version_label + ": every trained RNN", fontsize=13)
            save_figure(seed_fig, out_dir, f"{version}_{axis}_every_seed")
            plt.close(seed_fig)
            endpoint = raw[(raw.capability_subset == "all") & raw[x].isin([start, end])].copy()
            endpoint.insert(0, "version", version)
            endpoint.insert(1, "axis", axis)
            endpoint.rename(columns={x: "step"}, inplace=True)
            endpoints.append(endpoint)
        axes[0, 1].legend(fontsize=8, loc="lower right", frameon=False)
        fig.suptitle("All five training seeds: mean and 95% bootstrap CI", fontsize=13)
        save_figure(fig, out_dir, f"v1_v2_{axis}_all_seeds")
        plt.close(fig)
    endpoint_table = pd.concat(endpoints, ignore_index=True)
    endpoint_table.to_csv(out_dir / "endpoints_per_seed.csv", index=False)
    differences = []
    indices = np.random.default_rng(30_000).integers(0, 5, size=(10_000, 5))
    for (axis, step, condition, target), group in endpoint_table.groupby(["axis", "step", "condition", "target"]):
        pivot = group.pivot(index="training_seed", columns="version", values="distance_accuracy").sort_index()
        require(pivot.index.tolist() == list(range(1, 6)) and not pivot.isna().any().any(),
                "Version contrasts require all five nominally paired seeds")
        delta = (pivot[VERSIONS[1]] - pivot[VERSIONS[0]]).to_numpy()
        low, high = np.quantile(delta[indices].mean(axis=1), [.025, .975])
        differences.append({"axis": axis, "step": step, "condition": condition, "target": target,
                            "v1_mean": pivot[VERSIONS[0]].mean(), "v2_mean": pivot[VERSIONS[1]].mean(),
                            "v2_minus_v1_mean": delta.mean(), "difference_ci_low": low,
                            "difference_ci_high": high, "n_training_seeds": 5})
    pd.DataFrame(differences).to_csv(out_dir / "version_endpoint_differences.csv", index=False)
    lines = ["# Counterbalanced v1/v2 representation comparison", "",
             "All three RNN conditions and all five trained seeds are included for each version. "
             "Probes, preprocessing and the shared 16/4-per-profile split match the earlier analysis. "
             "Only linear readouts are trained; policy weights and evaluation rollouts are unchanged.", "",
             "The score is `1 − mean(abs(predicted_delay − actual_delay))/9`, rather than exact-class accuracy. "
             "The dashed baseline uses five independent random Normal representations. Shading resamples "
             "the five policy seeds, using 10,000 percentile bootstrap draws. Version differences pair "
             "nominal seed numbers; the episode trajectories are not paired. These are descriptive comparisons.", "",
             "Probe training includes all 46 capability profiles. Probe testing holds out episodes within "
             "each profile; the familiar/novel breakdown refers to whether a profile appeared in policy training. "
             "It does not test a probe's transfer to capability classes absent from probe fitting.", "",
             "| Condition | Target | v1, round 20 | v2, round 20 | v2 − v1 [95% CI] |",
             "| --- | --- | --- | --- | --- |"]
    for row in differences:
        if row["axis"] == "round" and row["step"] == 19:
            lines.append(f"| {CONDITION_LABELS[row['condition']]} | {row['target']} | "
                         f"{row['v1_mean']:.3f} | {row['v2_mean']:.3f} | "
                         f"{row['v2_minus_v1_mean']:+.3f} [{row['difference_ci_low']:+.3f}, "
                         f"{row['difference_ci_high']:+.3f}] |")
    lines.extend(["", "`v1_v2_time_all_seeds` and `v1_v2_round_all_seeds` compare version means. "
                  "The four `*_every_seed` figures expose all individual policy curves. "
                  "PNG and PDF versions are saved. Every network's UMAP figures and fitted probes "
                  "are in the adjacent version directories. Decoding establishes recoverable information, "
                  "not causal use of that information by the actor.", "", "Reproduce after completing both version analyses:",
                  "", "```bash", f"python eval/compare_representation_versions.py --results-root {root}", "```", ""])
    (out_dir / "README.md").write_text("\n".join(lines))
    print(f"Saved comparison, all-seed figures and endpoint tables to {out_dir}")


if __name__ == "__main__":
    main()
