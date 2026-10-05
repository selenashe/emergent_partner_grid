"""Use one common seed for every condition across original/counterbalanced v1/v2.

Select by the equal-weight mean final logged training episode return across
16 policies (four conditions x two versions x two experiment sets). Held-out
results are read only after this single seed has been selected.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.plot_best_training_seed import (
    METRICS, load_source, select_seeds, held_out_metrics,
    make_plots, write_csv,
)
from repo_paths import resolve_path


def main(args):
    sources, candidates = {}, []
    experiments = (("counterbalanced", False),) if args.batch_only else (
        ("original", True), ("counterbalanced", False))
    for experiment, original in experiments:
        path, raw, source = load_source(SimpleNamespace(
            original_versions=original, manifest=args.counterbalanced_manifest))
        runs, _ = select_seeds(source, window=args.selection_window_updates)
        candidates.extend({"experiment": experiment, **row} for row in runs)
        sources[experiment] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                   source=source)
    n_policies = 8 * len(sources)
    assert len(candidates) == 5 * n_policies
    scores = []
    for seed in range(1, 6):
        group = [r for r in candidates if r["seed"] == seed]
        assert len(group) == n_policies
        row = dict(seed=seed, mean_training_return=float(np.mean([
            r["training_return"] for r in group])), n_policies=n_policies)
        for experiment in sources:
            row[experiment + "_mean_training_return"] = float(np.mean([
                r["training_return"] for r in group if r["experiment"] == experiment]))
        scores.append(row)
    chosen = max(scores, key=lambda r: (r["mean_training_return"], -r["seed"]))["seed"]
    print(f"Common seed: {chosen}", flush=True)
    selected = [r for r in candidates if r["seed"] == chosen]
    # Selection is finalized before reading any held-out metrics.
    results = []
    for index, row in enumerate(selected):
        source = sources[row["experiment"]]["source"]
        directory = Path(source["versions"][row["version"]]["evaluation_root"])
        results.append({**row, **held_out_metrics(directory, row["condition"], chosen,
            args.bootstrap_repetitions, args.bootstrap_seed + index)})
    out = ROOT / "eval/protocol_comparison"
    if args.batch_only:
        out = resolve_path(out / sources["counterbalanced"]["source"]["batch"])
    out /= "common_training_seed"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "seed_selection_scores.csv", scores)
    write_csv(out / "training_candidates.csv", candidates)
    write_csv(out / "selected_seed_metrics.csv", [dict(
        experiment=r["experiment"], version=r["version"], condition=r["condition"],
        seed=chosen, training_return=r["training_return"],
        **{metric: r[metric]["mean"] for metric in METRICS}) for r in results])
    outputs = {}
    axis_limits = {}
    for metric in METRICS:
        scale = 100 if metric == "success" else 1
        upper = max(0, max(r[metric]["ci95"][1] for r in results) * scale)
        lower = min(0, min(r[metric]["ci95"][0] for r in results) * scale)
        axis_limits[metric] = upper + max(upper - lower, 1) * .15
    selection_scope = "all eight policies in this batch" if args.batch_only else "both experiment sets"
    for experiment, record in sources.items():
        source = record["source"]
        target = resolve_path(ROOT / "eval/protocol_comparison" / source["batch"]) / f"common_training_seed_{chosen}"
        target.mkdir(parents=True, exist_ok=True)
        group = [r for r in results if r["experiment"] == experiment]
        assert len(group) == 8 and {r["seed"] for r in group} == {chosen}
        make_plots(group, target, args.selection_window_updates, counterbalanced=experiment == "counterbalanced",
                   common_seed=chosen, axis_limits=axis_limits,
                   selection_scope=selection_scope, action_selection=source.get("action_selection"))
        for version in ("v1", "v2"):
            make_plots(group, target, args.selection_window_updates, counterbalanced=experiment == "counterbalanced",
                       versions=(version,), performance_only=True, common_seed=chosen,
                       axis_limits=axis_limits, selection_scope=selection_scope,
                       action_selection=source.get("action_selection"))
        outputs[experiment] = str(target)
    report = dict(common_seed=chosen,
        selection=dict(metric="Mean logged training episode return",
            window_updates=args.selection_window_updates,
            missing_episode_updates="Exclude updates with no completed episodes; their logged zeros are placeholders",
            scope=
            ("One seed shared across v1/v2, all four conditions in this batch" if args.batch_only else
             "One seed shared across original and counterbalanced v1/v2, all four conditions"),
            n_policies_per_seed=n_policies, weighting="Equal weight for each condition/version/experiment",
            held_out_results_used=False, tie_break="Lowest seed number for equal mean scores",
            logged_return_decimal_places=3,
            note="Final logged training rollout precedes the last gradient update; evaluation uses the final checkpoint."),
        uncertainty=dict(method="95% percentile bootstrap of whole episodes within each fixed held-out partner profile",
            repetitions=args.bootstrap_repetitions, seed=args.bootstrap_seed,
            includes_training_seed_variability=False),
        scores=scores, candidates=candidates, selected=results, outputs=outputs,
        shared_axis_limits=axis_limits,
        sources={experiment:dict(path=r["path"], sha256=r["sha256"],
            batch=r["source"]["batch"], corpus_note=r["source"]["corpus_note"])
            for experiment,r in sources.items()})
    (out / "selection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for target in outputs.values():
        (Path(target) / "selection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for r in results:
        print(f"{r['experiment']} {r['version']} {r['condition']}: seed {chosen}; "
            f"success {100*r['success']['mean']:.2f}%; return {r['episode_return']['mean']:.3f}; "
            f"steps {r['episode_steps']['mean']:.1f}")
    print(f"Selection records: {out}")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterbalanced-manifest", type=Path,
        default=ROOT / "train/manifests/sbatch_counterbalanced1096_20261002_235609.json")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--batch-only", action="store_true",
        help="Select one seed across only the supplied batch; preserve earlier comparison outputs")
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--selection-window-updates", type=int, default=1)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 100 or args.selection_window_updates < 1:
        parser.error("Use at least 100 bootstrap repetitions and a positive selection window")
    main(args)
