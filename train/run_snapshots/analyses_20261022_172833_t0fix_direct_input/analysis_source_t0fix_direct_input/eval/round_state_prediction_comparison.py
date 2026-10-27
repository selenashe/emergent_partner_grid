"""Compare class prediction fractions from cumulative means and round-end GRU states.

The displayed values are mean(predicted_class == k) within a ground-truth
subset. Incorrect predictions are counted directly; correct rejection is never
substituted for their frequency. Both solid lines sum to one.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
import pandas as pd

CLASS_NAMES = {0: "Red faster", 1: "Blue faster"}
CLASS_DELAYS = {0: "d_red < d_blue", 1: "d_blue < d_red"}
COLORS = {0: "#d94b41", 1: "#3779bf"}
PANEL_ORDER = (0, 1)
VERSIONS = ("v1_matched", "v2", "v3")
CONDITIONS = {
    "rnn_diverse_influence": "Diverse-partner RNN",
    "rnn_single_influence": "Single-partner RNN",
    "rnn_diverse_noinfluence": "No-influence RNN",
}
FEATURE_NAMES = {
    "cumulative_mean": "Cumulative mean of GRU states",
    "round_end_state": "Single GRU state at round end",
}


def write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def round_end_indices(dones, round_done, round_idx, n_rounds=20):
    """Last valid transition of each round, including the terminal transition."""
    dones, round_done, round_idx = map(np.asarray, (dones, round_done, round_idx))
    if dones.ndim != 1 or dones.shape != round_done.shape or dones.shape != round_idx.shape:
        raise ValueError("Invalid transition metadata")
    terminal = np.flatnonzero(dones)
    if not len(terminal):
        raise ValueError("No episode terminal flag")
    length = int(terminal[0]) + 1
    ends = np.flatnonzero(round_done[:length])
    if len(ends) != n_rounds or not np.array_equal(round_idx[ends], np.arange(n_rounds)):
        raise ValueError("Require all ordered valid round endings")
    if ends[-1] != length - 1:
        raise ValueError("The final round does not end at the first episode done")
    expected = np.repeat(np.arange(n_rounds), np.diff(np.r_[0, ends + 1]))
    if not np.array_equal(round_idx[:length], expected):
        raise ValueError("Round labels disagree with round boundaries")
    return ends, length


def prediction_fractions(truth, predictions, ground_truth_class):
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    if truth.ndim != 1 or truth.shape != predictions.shape:
        raise ValueError("Truth and predictions must align")
    if not np.isin(truth, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Invalid class labels")
    subset = truth == ground_truth_class
    n = int(subset.sum())
    if not n:
        raise ValueError("Empty ground-truth subset")
    counts = np.bincount(predictions[subset], minlength=2)
    fractions = counts / n
    np.testing.assert_allclose(fractions.sum(), 1, atol=1e-12)
    return fractions, counts, n


def extract_states(record):
    """Read actual saved h_t vectors at the same boundaries used by mean features."""
    prior = Path(record["analysis_dir"])
    episodes = pd.read_csv(prior / "episode_manifest.csv")
    if len(episodes) != 920:
        raise ValueError("Expected 920 canonical episodes")
    features = np.empty((20, 920, 128), dtype=np.float32)
    endpoint_steps = np.empty((20, 920), dtype=np.int32)
    prior_sources = {p["path"]: p for p in json.loads((prior / "completion.json").read_text())["sources"]}
    audited = []
    for path, group in episodes.groupby("source_file", sort=True):
        stat = Path(path).stat()
        expected = prior_sources[path]
        if stat.st_size != expected["size_bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Previously validated rollout changed: {path}")
        mapping = {int(row.source_episode): row for row in group.itertuples()}
        with h5py.File(path) as handle:
            if handle.attrs.get("allocation_protocol", "fixed_allocation") != record["allocation_protocol"]:
                raise ValueError("Wrong saved allocation protocol")
            E, T, H = handle["hidden_state"].shape
            if H != 128 or T != 2000 or len(mapping) != E or set(mapping) != set(range(E)):
                raise ValueError("Missing source episodes or malformed state array")
            dones, ends_flags, round_ids = (handle[k][:] for k in ("dones", "round_done", "round_idx"))
            # Chunked rollout files are read in blocks; uncompressed specialist
            # files support direct ordered point selections.
            compressed = handle["hidden_state"].chunks is not None
            for start in range(0, E, 16):
                stop = min(start + 16, E)
                block = handle["hidden_state"][start:stop] if compressed else None
                for source_episode in range(start, stop):
                    row = mapping[source_episode]
                    ends, length = round_end_indices(dones[source_episode], ends_flags[source_episode], round_ids[source_episode])
                    if length != row.valid_length:
                        raise ValueError("Valid trajectory length changed")
                    cap = handle["capability"][source_episode, 0]
                    if not np.array_equal(cap, [row.d_R, row.d_B]):
                        raise ValueError("Episode label does not match saved rollout")
                    selected = (block[source_episode - start, ends] if block is not None
                                else handle["hidden_state"][source_episode, ends, :])
                    if selected.shape != (20, 128) or not np.isfinite(selected).all():
                        raise ValueError("Invalid selected GRU state")
                    features[:, row.Index] = selected
                    endpoint_steps[:, row.Index] = ends
        after = Path(path).stat()
        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError("Rollout changed during extraction")
        audited.append(dict(path=path, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, episodes=E))
    return features, endpoint_steps, episodes, audited


def reference_module(root):
    path = root / "reference_source/eval/relative_speed_representation.py"
    spec = importlib.util.spec_from_file_location("round_state_reference_fitter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy_folder(root, record):
    return root / "policies" / record["version"] / f"{record['condition']}_seed{record['seed']}"


def fit_policy(root, manifest, record, reference):
    import torch
    torch.set_num_threads(1)
    out = policy_folder(root, record)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "completion.json").exists():
        print(f"[reuse] {record['version']} {record['condition']} seed{record['seed']}", flush=True)
        return
    print(f"[extract] {record['version']} {record['condition']} seed{record['seed']}", flush=True)
    features, endpoint_steps, episodes, sources = extract_states(record)
    prior_path = Path(record["analysis_dir"]) / "probe_models.npz"
    with np.load(prior_path) as prior:
        labels, train, test = prior["truth"][0].copy(), prior["train"].copy(), prior["test"].copy()
        pairs = prior["pairs"].copy()
        cumulative_predictions = prior["predictions"][9:29].copy()
    np.testing.assert_array_equal(labels, episodes["class"])
    np.testing.assert_array_equal(pairs, episodes[["d_R", "d_B"]])
    np.testing.assert_array_equal(train, episodes.probe_partition == "train")
    np.testing.assert_array_equal(test, episodes.probe_partition == "test")
    if (train.sum(), test.sum()) != (736, 184):
        raise ValueError("Expected the same within-profile episode split")
    np.savez_compressed(out / "round_end_features.npz", features=features, endpoint_steps=endpoint_steps,
                        labels=labels, pairs=pairs, train=train, test=test)
    print(f"[fit] {record['version']} {record['condition']} seed{record['seed']}", flush=True)
    targets = np.broadcast_to(labels, (20, 920)).copy()
    predictions, probabilities, models = reference.train_probes(features, targets, train, [10000] * 20)
    np.savez_compressed(out / "round_end_probes.npz", **models, predictions=predictions,
                        probabilities=probabilities, labels=labels, train=train, test=test)
    episodes.to_csv(out / "episode_manifest.csv", index=False)
    rows = []
    for feature_type, predicted in (("cumulative_mean", cumulative_predictions), ("round_end_state", predictions)):
        for r in range(20):
            for actual in PANEL_ORDER:
                fractions, counts, n = prediction_fractions(labels[test], predicted[r, test], actual)
                for decoded in PANEL_ORDER:
                    rows.append(dict(version=record["version"], condition=record["condition"],
                        training_seed=record["seed"], feature_type=feature_type, round=r + 1,
                        ground_truth_class=actual, predicted_class=decoded,
                        prediction_fraction=float(fractions[decoded]), prediction_count=int(counts[decoded]), n_episodes=n))
    pd.DataFrame(rows).to_csv(out / "prediction_fractions.csv", index=False)
    write(out / "completion.json", dict(status="completed", version=record["version"], condition=record["condition"],
        seed=record["seed"], n_episodes=920, n_train=736, n_test=184, discrete_probes=20,
        sources=sources, cumulative_probe_file=str(prior_path), cumulative_probe_sha256=sha(prior_path),
        feature_protocol="exact h_t at each round's final valid transition, post-observation before action/outcome",
        completed_utc=datetime.now(timezone.utc).isoformat()))
    print(f"[done] {record['version']} {record['condition']} seed{record['seed']}", flush=True)


def worker(root, version):
    root = Path(root).resolve()
    manifest = json.loads((root / "analysis_run.json").read_text())
    reference = reference_module(root)
    for record in manifest["policies"]:
        if record["version"] == version:
            fit_policy(root, manifest, record, reference)


def summarize(table):
    keys = ["version", "condition", "feature_type", "round", "ground_truth_class", "predicted_class"]
    resamples = np.random.default_rng(30000).integers(0, 5, size=(10000, 5))
    rows = []
    for group_keys, group in table.groupby(keys, sort=True):
        if len(group) != 5 or set(group.training_seed) != set(range(1, 6)):
            raise ValueError("Every summary must include all five seeds")
        values = group.sort_values("training_seed").prediction_fraction.to_numpy()
        boot = values[resamples].mean(axis=1)
        low, high = np.quantile(boot, [.025, .975])
        rows.append(dict(zip(keys, group_keys), mean=float(values.mean()), sd=float(values.std(ddof=1)),
                         ci_low=float(low), ci_high=float(high), n_seeds=5))
    return pd.DataFrame(rows)


def panel(ax, data, actual, aggregated):
    for predicted in (0, 1):
        group = data[(data.ground_truth_class == actual) & (data.predicted_class == predicted)].sort_values("round")
        ax.plot(group["round"], group["mean" if aggregated else "prediction_fraction"],
                color=COLORS[predicted], lw=2, label=f"Predicted: {CLASS_NAMES[predicted]}")
        if aggregated:
            ax.fill_between(group["round"], group.ci_low, group.ci_high, color=COLORS[predicted], alpha=.15)
    ax.axhline(1 / 2, color="#b0b0b0", ls="--", lw=.8)
    ax.set_ylim(0, 1.02); ax.set_xlim(1, 20); ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Round")
    ax.set_title(f"Ground truth: {CLASS_NAMES[actual]}\n({CLASS_DELAYS[actual]})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def comparison_figure(table, version, condition, seed=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.2), sharey=True)
    selected = table[(table.version == version) & (table.condition == condition)]
    if seed is not None:
        selected = selected[selected.training_seed == seed]
    for row, feature in enumerate(FEATURE_NAMES):
        for col, actual in enumerate(PANEL_ORDER):
            panel(axes[row, col], selected[selected.feature_type == feature], actual, aggregated=seed is None)
        axes[row, 0].set_ylabel(FEATURE_NAMES[feature] + "\nFraction predicted as each class")
    title = f"{version} · {CONDITIONS[condition]}"
    title += " · all 5 seeds (95% CI across seeds)" if seed is None else f" · seed {seed}"
    fig.suptitle(title, y=.98, fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .035), ncol=2, frameon=False)
    fig.text(.5, .012, "Two solid lines sum to 1 within each ground-truth subset. "
             "Matching line = recall; other lines = misclassification fractions. Dashed guide = 1/2.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, .96))
    return fig


def aggregate(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    root = Path(root).resolve()
    manifest = json.loads((root / "analysis_run.json").read_text())
    tables = []
    for record in manifest["policies"]:
        out = policy_folder(root, record)
        done = json.loads((out / "completion.json").read_text())
        if done["status"] != "completed" or done["discrete_probes"] != 20:
            raise ValueError("Incomplete policy comparison")
        tables.append(pd.read_csv(out / "prediction_fractions.csv"))
    table = pd.concat(tables, ignore_index=True)
    if len(table) != 7200:
        raise ValueError("Missing policy/cutoff/subset/class results")
    sum_keys = ["version", "condition", "training_seed", "feature_type", "round", "ground_truth_class"]
    np.testing.assert_allclose(table.groupby(sum_keys).prediction_fraction.sum(), 1, atol=1e-12)
    summary = summarize(table)
    summary_keys = [k for k in sum_keys if k != "training_seed"]
    np.testing.assert_allclose(summary.groupby(summary_keys)["mean"].sum(), 1, atol=1e-12)
    table.to_csv(root / "prediction_fractions_per_seed.csv", index=False)
    summary.to_csv(root / "prediction_fractions_summary.csv", index=False)
    summary[summary["round"] == 20].to_csv(root / "round20_summary.csv", index=False)
    figures = dict(aggregate_png=0, aggregate_pdf=0, individual_seed_png=0, individual_seed_pages=0)
    for condition in CONDITIONS:
        folder = root / condition
        folder.mkdir(exist_ok=True)
        with PdfPages(folder / "comparison_all_versions.pdf") as overview:
            for version in VERSIONS:
                fig = comparison_figure(summary, version, condition)
                for extension in ("png", "pdf"):
                    fig.savefig(folder / f"comparison_{version}.{extension}", dpi=170)
                    figures[f"aggregate_{extension}"] += 1
                overview.savefig(fig)
                plt.close(fig)
        with PdfPages(folder / "comparison_individual_seeds.pdf") as book:
            for version in VERSIONS:
                for seed in range(1, 6):
                    fig = comparison_figure(table, version, condition, seed=seed)
                    book.savefig(fig)
                    fig.savefig(folder / f"comparison_{version}_seed{seed}.png", dpi=110)
                    figures["individual_seed_png"] += 1; figures["individual_seed_pages"] += 1
                    plt.close(fig)
    write(root / "completion_audit.json", dict(status="completed", n_policy_entries=45,
        n_new_discrete_state_probes=900, n_cumulative_round_probes_reused=900, n_per_seed_rows=7200,
        n_summary_rows=len(summary), per_subset_solid_line_sums="verified equal to 1", figures=figures,
        completed_utc=datetime.now(timezone.utc).isoformat()))
    (root / "README.md").write_text("# Specialist-only prediction fractions: mean versus discrete GRU states\n\nThis is the reference visualization for v1_matched, v2, and v3, three RNN conditions and all five learner seeds (45 policies).\nOnly the original 46 specialist profiles are included: 920 episodes per policy, 736 fitting and 184 held out. Each class has 368 fitting and 92 test episodes.\n\nClasses: red faster (d_red<d_blue, label 0) and blue faster (d_blue<d_red, label 1).\nEach figure has two ground-truth panels. Lines show P(predicted class=k | true class=g) using argmax predictions, conditional on that panel's 92 held-out episodes. Both lines sum to one. Matching line=recall; other line=misclassification frequency. Dashed chance guide=1/2.\n\nCumulative means average post-observation GRU states from episode start through the final valid transition of each round. Discrete features use the single state at that same endpoint, before its action/outcome. Include terminal transitions; exclude padding. Each policy/round has an independent affine 128→2 softmax probe, full-batch Adam lr=.01, 1000 updates, initialization seed 10000. Primary class weights are exactly 1. No scaling, regularization, early stopping, or test-based selection.\n\nAggregate means and 95% percentile-bootstrap intervals use five policy seeds, 10000 shared resamples, RNG 30000. Individual-seed plots have no between-seed interval. Each condition contains comparison_all_versions.png (six rows × two columns), comparison_<version>.png/pdf, and all fifteen individual-seed PNGs plus a PDF book. Models/features/episode splits are retained under policies/; frozen sources under source/ and reference_source/.\n")
    manifest["status"], manifest["completed_utc"] = "completed", datetime.now(timezone.utc).isoformat()
    write(root / "analysis_run.json", manifest)
    print(f"[completed] {root}", flush=True)


def prepare(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    prior = json.loads((source / "analysis_run.json").read_text())
    if (prior["status"] != "completed" or len(prior["policies"]) != 45
            or prior["episodes_per_policy"] != 920 or len(prior["classes"]) != 2):
        raise ValueError("Require completed source analysis")
    if out.exists():
        raise ValueError("Prepare a fresh output directory")
    out.mkdir(parents=True)
    (out / "source").mkdir(); (out / "logs").mkdir()
    shutil.copyfile(__file__, out / "source" / Path(__file__).name)
    hashes = {}
    for relative in ("eval/relative_speed_representation.py", "eval/representation_analysis.py", "repo_paths.py",
                     "jaxmarl/environments/coordination_grid/capability_populations.py"):
        target = out / "reference_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "source" / relative, target)
        hashes[relative] = sha(target)
    manifest = dict(status="prepared", source_results=str(source), root=str(out),
        policies=prior["policies"], source_manifest_sha256=sha(source / "analysis_run.json"),
        script_sha256=sha(out / "source" / Path(__file__).name), reference_sha256=hashes,
        python=prior["analysis_python"], prepared_utc=datetime.now(timezone.utc).isoformat(),
        metric="P(predicted class=k | ground-truth class=g)",
        discrete_state="post-observation h_t at the final valid transition of each round, before action/outcome",
        n_policy_entries=45, n_rounds=20, processes=[])
    write(out / "analysis_run.json", manifest)
    return out


def run(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "analysis_run.json").read_text())
    if manifest["status"] == "completed":
        return
    script = root / "source" / Path(__file__).name
    if sha(script) != manifest["script_sha256"]:
        raise ValueError("Frozen comparison source changed")
    for relative, digest in manifest["reference_sha256"].items():
        if sha(root / "reference_source" / relative) != digest:
            raise ValueError("Frozen reference fitting recipe changed")
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1", PYTHONUNBUFFERED="1", MPLCONFIGDIR=str(root / "mpl_cache"))
    processes, handles = [], []
    manifest["status"] = "running"
    write(root / "analysis_run.json", manifest)
    try:
        for version in VERSIONS:
            log = root / "logs" / f"{version}.log"
            handle = log.open("a"); handles.append(handle)
            p = subprocess.Popen([manifest["python"], str(script), "worker", str(root), "--version", version],
                env=environment, stdout=handle, stderr=subprocess.STDOUT)
            processes.append(p)
            manifest["processes"].append(dict(version=version, pid=p.pid, log=str(log)))
            write(root / "analysis_run.json", manifest)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("A comparison worker failed; inspect logs")
            done = sum((policy_folder(root, r) / "completion.json").exists() for r in manifest["policies"])
            print(f"[progress] discrete-state fits {done}/45", flush=True)
            time.sleep(30)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("A comparison worker failed")
        aggregate(root)
    except BaseException:
        for p in processes:
            if p.poll() is None: p.terminate()
        manifest["status"] = "failed"
        write(root / "analysis_run.json", manifest)
        raise
    finally:
        for handle in handles: handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "worker", "aggregate"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--source-results", type=Path)
    parser.add_argument("--version", choices=VERSIONS)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.source_results is None: parser.error("prepare requires --source-results")
        print(prepare(args.source_results, args.root))
    elif args.mode == "run": run(args.root)
    elif args.mode == "aggregate": aggregate(args.root)
    elif args.mode == "worker":
        if args.version is None: parser.error("worker requires --version")
        worker(args.root, args.version)


if __name__ == "__main__":
    main()
