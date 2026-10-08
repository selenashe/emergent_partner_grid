"""Compare the existing written-methods probes with released-code probes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.representation_analysis import CONDITIONS, CONDITION_LABELS, TARGETS, plotting_style, require, save_figure


def main():
    # Audit guide:
    # Compare the written-method and released-code probe outputs for each allocation
    # version without conflating their optimizers, splits, warm starts, or test
    # selection. Keep random-feature baseline labels explicit because the two analysis
    # methods differ in their targets.
    #
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-root", default="eval/representation_results/counterbalanced1096_20261002_235609")
    parser.add_argument("--strict-root", default="eval/representation_results/counterbalanced1096_20261002_235609_strict_replication_of_covercooked")
    args = parser.parse_args()
    roots = {"previous": Path(args.previous_root), "strict": Path(args.strict_root)}
    out = roots["strict"] / "protocol_comparison"
    out.mkdir(parents=True, exist_ok=True)
    tables = {}
    for version in ("v1", "v2"):
        sources = []
        for protocol, root in roots.items():
            folder = root / f"{version}_balanced_training"
            metadata = json.loads((folder / "analysis_metadata.json").read_text())
            require(metadata["n_checkpoints"] == 15, "Missing RNN checkpoints")
            if protocol == "strict":
                require(metadata["arguments"]["probe_protocol"] == "strict_released_code", "Not strict outputs")
            sources.append({(r["condition"], r["training_seed"]): r["sources"] for r in metadata["validation"]})
            for axis, stem in (("time", "probe_timestep"), ("round", "probe_by_round")):
                tables[version, protocol, axis] = pd.read_csv(folder / f"{stem}_per_seed.csv")
                tables[version, protocol, axis + "_summary"] = pd.read_csv(folder / f"{stem}_summary.csv")
        require(sources[0] == sources[1], "Protocols used different source evaluation files")
    plt = plotting_style()
    endpoints = []
    rng_indices = np.random.default_rng(30_000).integers(0, 5, size=(10_000, 5))
    for axis, step_column, final_step, xlabel in (("time", "reference_t", 400, "Episode timestep"),
                                                ("round", "round_idx", 19, "Completed round (1–20)")):
        fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharey=True, layout="constrained")
        for row, condition in enumerate(CONDITIONS):
            for col, target in enumerate(TARGETS):
                ax = axes[row, col]
                for version, color in (("v1", "#0072B2"), ("v2", "#D55E00")):
                    for protocol, style in (("previous", "-"), ("strict", "--")):
                        source = tables[version, protocol, axis + "_summary"]
                        group = source[(source.condition == condition) & (source.target == target)
                                       & (source.capability_subset == "all")].sort_values(step_column)
                        require((group.n_training_seeds == 5).all(), "A summary omits policy seeds")
                        x = group[step_column].to_numpy() + (1 if axis == "round" else 0)
                        ax.plot(x, group.distance_accuracy_mean, label=f"{version}: {protocol}",
                                color=color, ls=style, lw=2, marker="o", ms=2)
                        ax.fill_between(x, group.distance_accuracy_ci_low, group.distance_accuracy_ci_high,
                                        color=color, alpha=.08, lw=0)
                    raw_groups = []
                    for protocol in ("previous", "strict"):
                        raw = tables[version, protocol, axis]
                        group = raw[(raw.condition == condition) & (raw.target == target)
                                    & (raw.capability_subset == "all") & (raw[step_column] == final_step)]
                        group = group.set_index("training_seed").sort_index()
                        require(group.index.tolist() == list(range(1, 6)), "Endpoint omits policy seeds")
                        raw_groups.append(group)
                    for metric in ("distance_accuracy", "exact_accuracy", "mae"):
                        delta = (raw_groups[1][metric] - raw_groups[0][metric]).to_numpy()
                        low, high = np.quantile(delta[rng_indices].mean(axis=1), [.025, .975])
                        endpoints.append({"axis": axis, "step": final_step, "version": version,
                                          "condition": condition, "target": target, "metric": metric,
                                          "previous_mean": raw_groups[0][metric].mean(),
                                          "strict_mean": raw_groups[1][metric].mean(), "strict_minus_previous": delta.mean(),
                                          "difference_ci_low": low, "difference_ci_high": high, "n_training_seeds": 5})
                ax.set(title=f"{CONDITION_LABELS[condition]} — {'red' if target == 'd_R' else 'blue'} delay",
                       xlabel=xlabel, ylim=(.45, 1.01))
                ax.grid(axis="y", alpha=.2)
                if col == 0:
                    ax.set_ylabel("Distance-aware test accuracy")
                if axis == "round":
                    ax.set_xticks([1, 5, 10, 15, 20])
        axes[0, 1].legend(fontsize=8, loc="lower right", frameon=False)
        fig.suptitle("Same frozen RNNs: previous vs strict released-code probes\nFive-seed means and 95% bootstrap intervals", fontsize=13)
        save_figure(fig, out, f"strict_vs_previous_{axis}")
        plt.close(fig)
    table = pd.DataFrame(endpoints)
    table.to_csv(out / "endpoint_protocol_differences.csv", index=False)
    lines = ["# Probe-protocol comparison on the same frozen RNNs", "",
             "Source HDF5 file provenance is identical between protocols for all 30 RNN checkpoints. "
             "Solid curves use the preceding written-methods implementation; dashed curves use the "
             "pinned released-code routine. Colors identify v1/v2, and shading bootstraps all five policy seeds.", "",
             "The scores use the same normalized absolute-error metric, but test partitions and fitting "
             "procedures differ. Score changes are changes to measurement on fixed models, not changes "
             "to policy training or behavior. Strict probes select by test score and can carry training "
             "information into later resampled test partitions, as the per-version reports document.", "",
             "| Version | Condition | Target | Previous, round 20 | Strict, round 20 | Change |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in table[(table.axis == "round") & (table.metric == "distance_accuracy")].itertuples():
        lines.append(f"| {r.version} | {CONDITION_LABELS[r.condition]} | {r.target} | "
                     f"{r.previous_mean:.3f} | {r.strict_mean:.3f} | {r.strict_minus_previous:+.3f} |")
    lines.extend(["", "`endpoint_protocol_differences.csv` also reports exact-class accuracy and MAE, "
                  "with descriptive paired-policy-seed intervals. The baseline differs between "
                  "implementations and is omitted from this sensitivity figure; the primary per-protocol "
                  "figures display their respective baselines.", "", "```bash",
                  f"python eval/compare_representation_protocols.py --previous-root {args.previous_root} \\",
                  f"  --strict-root {args.strict_root}", "```", ""])
    (out / "README.md").write_text("\n".join(lines))
    print(f"Saved strict-versus-previous comparison to {out}")


if __name__ == "__main__":
    main()
