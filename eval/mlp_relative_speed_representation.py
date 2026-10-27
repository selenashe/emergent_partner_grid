"""Probe saved MLP policies using final actor-layer activations.

Replay each frozen evaluator with its original seed and capture Dense_1 after
ReLU, immediately before the action logits. Require every saved trajectory
field to match before accepting features. The MLP's returned dummy carry is
never used as a representation. Original evaluations and RNN results are read
only; this extension has its own manifest, features, probes, and plots.
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

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("v1_matched", "v2", "v3")
CONDITION = "mlp_diverse_influence"
FEATURE_NAMES = {
    "cumulative_mean": "Cumulative mean of final MLP hidden layer",
    "round_end_state": "Final MLP hidden layer at round end",
}
ANALYSIS_PYTHON = "/juice6/u/nlp/climblab/tmp/epg_repr_all10_20261008/bin/python"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference(root):
    return load_module("mlp_reference_probe", Path(root) / "reference_source/eval/relative_speed_representation.py")


def plotting(root):
    module = load_module("mlp_reference_plotting", Path(root) / "reference_source/round_state_prediction_comparison.py")
    module.CONDITIONS = {CONDITION: "Diverse-partner MLP"}
    module.FEATURE_NAMES = FEATURE_NAMES
    return module


def policy_dir(root, record):
    return Path(root) / "policies" / record["version"] / f"{CONDITION}_seed{record['seed']}"


def verify_inputs(root, manifest, record=None):
    for relative, digest in manifest["recipe_sha256"].items():
        if sha(Path(root) / relative) != digest:
            raise ValueError("Analysis recipe changed: " + relative)
    for item in manifest["policies"] if record is None else [record]:
        for field in ("checkpoint", "config"):
            if sha(item[field]) != item[field + "_sha256"]:
                raise ValueError("Trained input changed: " + item[field])
        for relative, digest in item["source_sha256"].items():
            if sha(Path(item["source_root"]) / relative) != digest:
                raise ValueError("Frozen policy source changed: " + relative)
        for path, expected in item["original_stats"].items():
            stat = Path(path).stat()
            if (stat.st_size, stat.st_mtime_ns) != (expected["size"], expected["mtime_ns"]):
                raise ValueError("Original evaluation changed: " + path)


def prepare(training_manifest, rnn_results, out):
    training_manifest, rnn_results, out = map(lambda p: Path(p).resolve(), (training_manifest, rnn_results, out))
    training, prior = read(training_manifest), read(rnn_results / "analysis_run.json")
    if training["status"] != "completed" or prior["status"] != "completed":
        raise ValueError("Require completed training and RNN reference analyses")
    if training["seeds"] != [1, 2, 3, 4, 5] or len(prior["policies"]) != 45:
        raise ValueError("Expected the matched five-seed cohort")
    out.mkdir(parents=True, exist_ok=False)
    (out / "source").mkdir(); (out / "logs").mkdir()
    shutil.copyfile(__file__, out / "source" / Path(__file__).name)
    for relative in ("eval/relative_speed_representation.py", "eval/representation_analysis.py", "repo_paths.py",
                     "jaxmarl/environments/coordination_grid/capability_populations.py"):
        destination = out / "reference_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rnn_results / "source" / relative, destination)
    shutil.copyfile(rnn_results / "prediction_fractions_state_comparison/source/round_state_prediction_comparison.py",
                    out / "reference_source/round_state_prediction_comparison.py")
    policies = []
    for job in training["training_jobs"]:
        if job["condition"] != CONDITION:
            continue
        version = job["version"]
        data = training["versions"][version]
        source = Path(training["source_variants"][job["source_variant"]]["frozen_source_root"])
        prefix = Path(data["checkpoint_root"]) / job["tag"]
        checkpoint, config = Path(str(prefix) + ".safetensors"), Path(str(prefix) + "_config.json")
        cfg = read(config)
        if (cfg["MODEL_TYPE"], cfg["GRID_ENCODER"], cfg["INCLUDE_IS_T0"], cfg["FC_DIM_SIZE"], cfg["SEED"]) != (
                "mlp", "flat", True, 128, job["seed"]):
            raise ValueError("Unexpected MLP architecture/cohort")
        rollouts = {p: str((Path(data["evaluation_root"]) / (job["tag"] + "_" + p + ".h5")).resolve())
                    for p in ("train", "test")}
        policies.append(dict(version=version, condition=CONDITION, seed=job["seed"],
            checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), config=str(config), config_sha256=sha(config),
            source_root=str(source), source_sha256={str(p.relative_to(source)): sha(p)
                for p in sorted(source.rglob("*.py")) if "outputs" not in p.parts},
            allocation_protocol=data["allocation_protocol"], original_rollouts=rollouts,
            original_stats={p: dict(size=Path(p).stat().st_size, mtime_ns=Path(p).stat().st_mtime_ns)
                            for p in rollouts.values()}))
    if {(p["version"], p["seed"]) for p in policies} != {(v, s) for v in VERSIONS for s in range(1, 6)}:
        raise ValueError("Require all 15 MLP policies")
    manifest = dict(status="prepared", prepared_utc=now(), root=str(out), rnn_results=str(rnn_results),
        training_manifest=str(training_manifest), training_manifest_sha256=sha(training_manifest),
        rnn_manifest_sha256=sha(rnn_results / "analysis_run.json"), policies=policies,
        layouts_dir=training["frozen_layouts_dir"], layout_files_sha256=training["layout_files_sha256"],
        evaluation_seed=training["evaluation"]["seed"], episodes_per_profile=20, episodes_per_policy=920,
        seeds=list(range(1, 6)), versions=list(VERSIONS), analysis_python=ANALYSIS_PYTHON,
        layer="relu(Dense_1), final 128-D actor hidden layer immediately before Dense_2 action logits",
        feature_types=FEATURE_NAMES, probe=dict(updates=1000, learning_rate=.01, init_seed=10000,
            split_seed=0, bootstrap_seed=30000, bootstrap_resamples=10000),
        recipe_sha256={str(p.relative_to(out)): sha(p) for directory in ("source", "reference_source")
                      for p in sorted((out / directory).rglob("*.py"))})
    write(out / "analysis_run.json", manifest)
    verify_inputs(out, manifest)
    print(out / "analysis_run.json", flush=True)


def capture_network(network, audit):
    """Expose the actual actor activation while preserving policy probabilities."""
    import jax.numpy as jnp
    from flax import linen as nn

    class CapturingNetwork:
        def apply(self, params, hidden, inputs):
            (unused, pi, value), captured = network.apply(params, hidden, inputs,
                capture_intermediates=lambda module, method: isinstance(module, nn.Dense) and module.name == "Dense_1",
                mutable=["intermediates"])
            preactivation = captured["intermediates"]["Dense_1"]["__call__"][0]
            activation = nn.relu(preactivation)
            if activation.shape[0] != 1 or activation.shape[-1] != 128:
                raise ValueError("Unexpected final actor activation shape")
            if not audit:
                # Separate eager forward verifies that this activation is precisely
                # the input to the action-logit projection, with the original mask.
                logits = activation @ params["params"]["Dense_2"]["kernel"] + params["params"]["Dense_2"]["bias"]
                reconstructed = jnp.where(jnp.isfinite(pi.logits), logits, -jnp.inf)
                import jax
                reconstructed = jax.nn.log_softmax(reconstructed, axis=-1)
                np.testing.assert_allclose(np.asarray(reconstructed), np.asarray(pi.logits), rtol=2e-5, atol=2e-5)
                (_, plain_pi, plain_value) = network.apply(params, hidden, inputs)
                np.testing.assert_array_equal(np.asarray(pi.logits), np.asarray(plain_pi.logits))
                np.testing.assert_array_equal(np.asarray(value), np.asarray(plain_value))
                audit.append(dict(logits_reconstructed_from_final_layer=True,
                                  capture_preserves_policy_logits_and_value_exactly=True))
            return activation[0], pi, value

    return CapturingNetwork()


def extract_policy(root, index):
    """Replay exactly the original full batch; reduce only after trajectory checks."""
    root = Path(root).resolve()
    manifest = read(root / "analysis_run.json")
    record = manifest["policies"][index]
    verify_inputs(root, manifest, record)
    out = policy_dir(root, record); out.mkdir(parents=True, exist_ok=True)
    if (out / "extraction.json").exists():
        prior = read(out / "extraction.json")
        if prior["features_sha256"] != sha(out / "features.npz"):
            raise ValueError("Saved MLP features changed")
        print("[reuse extraction]", record["version"], record["seed"], flush=True)
        return
    source = Path(record["source_root"])
    sys.path.insert(0, str(source))
    evaluator = load_module("frozen_mlp_evaluator", source / "analysis/evaluate_partner_modelling.py")
    import jax
    import jax.numpy as jnp
    from jaxmarl.wrappers.baselines import load_params
    cfg = read(record["config"])
    params = load_params(record["checkpoint"])
    original_build = evaluator.trainer_mod.build_network
    layer_audit = []

    def build_with_capture(config, action_dim):
        network, initialize, model_type = original_build(config, action_dim)
        if model_type != "mlp":
            raise ValueError("Only MLP policies can use this collector")
        return capture_network(network, layer_audit), initialize, model_type

    evaluator.trainer_mod.build_network = build_with_capture
    ref, plots = reference(root), plotting(root)
    rows, cumulative, discrete, timestep, timestep_discrete, endpoints, checks = [], [], [], [], [], [], []
    for population, pool in (("train", evaluator.TRAIN_CAPABILITY_PAIRS), ("test", evaluator.TEST_CAPABILITY_PAIRS)):
        print("[replay]", record["version"], record["seed"], population, str(jax.devices()), flush=True)
        # Audit the capture eagerly on real-shaped observations before lax.scan.
        if not layer_audit:
            kwargs = dict(cfg["ENV_KWARGS"], layouts_dir=manifest["layouts_dir"], augment_symmetries=False,
                          partner_capability_pairs=[list(p) for p in pool], influence=bool(cfg["INFLUENCE"]))
            kwargs.pop("layout_path", None); kwargs.pop("layout_paths", None)
            env = evaluator.CoordinationGrid(**kwargs)
            observation, _ = env.reset(jax.random.PRNGKey(1))
            obs = jax.tree_util.tree_map(lambda x: x[None, None], observation["agent_0"])
            net, init, _ = build_with_capture(cfg, env.n_ego_actions)
            net.apply(params, init(1), (obs, jnp.zeros((1, 1), dtype=bool)))
        trajectory, _ = evaluator.rollout_condition(params, cfg, pool,
            n_episodes_per_capability=manifest["episodes_per_profile"], layouts_dir=manifest["layouts_dir"],
            seed=manifest["evaluation_seed"], save_hidden=True)
        path = record["original_rollouts"][population]
        with h5py.File(path, "r") as saved:
            compared = []
            for field, value in trajectory.items():
                if field == "hidden_state":
                    continue
                np.testing.assert_array_equal(value, saved[field][:], err_msg=f"Replay differs: {path}: {field}")
                compared.append(field)
        checks.append(dict(population=population, path=path, exact_trajectory_fields=compared,
                           episodes=int(trajectory["dones"].shape[0])))
        hidden = trajectory["hidden_state"]
        if hidden.shape != (len(pool) * 20, 2000, 128) or not np.isfinite(hidden).all() or hidden.min() < 0:
            raise ValueError("Expected finite post-ReLU final-layer activations")
        counts = {}
        for e in range(len(hidden)):
            ends, length = plots.round_end_indices(trajectory["dones"][e], trajectory["round_done"][e],
                                                   trajectory["round_idx"][e])
            pair = tuple(map(int, trajectory["capability"][e, 0]))
            if not np.all(trajectory["capability"][e, :length] == pair):
                raise ValueError("Capability changes within a valid episode")
            rep = counts.get(pair, 0); counts[pair] = rep + 1
            valid = hidden[e, :length]
            mean = ref.original.prefix_mean_hidden(valid, ends + 1)
            # Independent slice means catch endpoint and denominator mistakes.
            for r in (0, 19):
                np.testing.assert_allclose(mean[r], valid[:ends[r]+1].mean(axis=0, dtype=np.float64), atol=2e-6)
            cumulative.append(mean); discrete.append(valid[ends].copy())
            timestep.append(ref.original.prefix_mean_hidden(valid, ref.original.REFERENCE_TIMES))
            timestep_discrete.append(valid[np.minimum(ref.original.REFERENCE_TIMES, length)-1].copy())
            endpoints.append(ends)
            rows.append(dict(source_file=path, source_episode=e, capability_slice=population,
                capability_index=int(trajectory["capability_index_per_ep"][e]), d_R=pair[0], d_B=pair[1],
                rollout_rep=rep, valid_length=length, scan_length=2000))
        if set(counts) != set(map(tuple, pool)) or set(counts.values()) != {20}:
            raise ValueError("Require 20 episodes per original specialist profile")
        del trajectory, hidden
        jax.clear_caches()
    episodes = pd.DataFrame(rows).sort_values(["d_R", "d_B", "rollout_rep"])
    order = episodes.index.to_numpy(); episodes = episodes.reset_index(drop=True)
    pairs = episodes[["d_R", "d_B"]].to_numpy()
    labels = ref.relative_classes(pairs)
    train, test = ref.masks(episodes.rollout_rep.to_numpy(), pairs, ref.original.make_probe_split(0))
    if len(episodes) != 920 or (train.sum(), test.sum()) != (736, 184):
        raise ValueError("Incorrect episode or split count")
    np.testing.assert_array_equal(np.bincount(labels[train]), [368, 368])
    np.testing.assert_array_equal(np.bincount(labels[test]), [92, 92])
    episodes["class"] = labels; episodes["probe_partition"] = np.where(train, "train", "test")
    arrays = {key: np.asarray(values, dtype=np.float32)[order].transpose(1, 0, 2)
              for key, values in (("cumulative_mean", cumulative), ("round_end_state", discrete),
                                  ("timestep_cumulative_mean", timestep), ("timestep_state", timestep_discrete))}
    for key, value in arrays.items():
        if not np.isfinite(value).all() or np.std(value) == 0:
            raise ValueError("Invalid/constant features: " + key)
    np.savez_compressed(out / "features.npz", **arrays, labels=labels, pairs=pairs, train=train, test=test,
        endpoint_steps=np.asarray(endpoints)[order].T, time_cutoffs=ref.original.REFERENCE_TIMES)
    episodes.to_csv(out / "episode_manifest.csv", index=False)
    verify_inputs(root, manifest, record)
    write(out / "extraction.json", dict(status="completed", version=record["version"], seed=record["seed"],
        layer=manifest["layer"], layer_audit=layer_audit, trajectories=checks, episodes=920, train=736, test=184,
        features_sha256=sha(out / "features.npz"), episodes_sha256=sha(out / "episode_manifest.csv"),
        feature_shapes={k:list(v.shape) for k,v in arrays.items()}, backend=str(jax.devices()),
        terminal_included=True, padding_excluded=True, independent_prefix_mean_checks=True, completed_utc=now()))
    evaluator.trainer_mod.build_network = original_build
    print("[extracted]", record["version"], record["seed"], flush=True)


def fit_policy(root, index):
    import torch
    torch.set_num_threads(1)
    root = Path(root).resolve(); manifest = read(root / "analysis_run.json")
    record = manifest["policies"][index]; out = policy_dir(root, record)
    verify_inputs(root, manifest, record)
    if (out / "completion.json").exists():
        print("[reuse probes]", record["version"], record["seed"], flush=True)
        return
    if read(out / "extraction.json")["features_sha256"] != sha(out / "features.npz"):
        raise ValueError("Extracted features changed")
    ref, plots = reference(root), plotting(root)
    with np.load(out / "features.npz") as data:
        labels, train, test, pairs = (data[k].copy() for k in ("labels", "train", "test", "pairs"))
        features = np.concatenate([data[k] for k in FEATURE_NAMES] +
                                  [data["timestep_cumulative_mean"], data["timestep_state"]])
    specs = [(feature, "round", r) for feature in FEATURE_NAMES for r in range(1, 21)]
    specs += [(feature, "timestep", int(t)) for feature in FEATURE_NAMES for t in ref.original.REFERENCE_TIMES]
    truth = np.broadcast_to(labels, features.shape[:2]).copy()
    print("[fit]", record["version"], record["seed"], len(features), "primary probes", flush=True)
    predictions, probabilities, models = ref.train_probes(features, truth, train, [10000] * len(features))
    np.savez_compressed(out / "probe_models.npz", **models, predictions=predictions,
        probabilities=probabilities, truth=truth, train=train, test=test, pairs=pairs,
        feature_type=np.array([s[0] for s in specs]), axis=np.array([s[1] for s in specs]),
        cutoff=np.array([s[2] for s in specs]))
    fractions, metrics, confusions = [], [], []
    for i, (feature, axis, cutoff) in enumerate(specs):
        common = dict(version=record["version"], condition=CONDITION, training_seed=record["seed"],
                      feature_type=feature, axis=axis, cutoff=cutoff)
        metric, confusion = ref.class_metrics(labels[test], predictions[i, test])
        metrics.append(dict(**common, **metric))
        for actual in (0, 1):
            frac, counts, n = plots.prediction_fractions(labels[test], predictions[i, test], actual)
            for decoded in (0, 1):
                confusions.append(dict(**common, true_class=actual, predicted_class=decoded,
                                       count=int(confusion[actual, decoded])))
                if axis == "round":
                    fractions.append(dict(version=record["version"], condition=CONDITION, training_seed=record["seed"],
                        feature_type=feature, round=cutoff, ground_truth_class=actual, predicted_class=decoded,
                        prediction_fraction=float(frac[decoded]), prediction_count=int(counts[decoded]), n_episodes=n))
    pd.DataFrame(fractions).to_csv(out / "prediction_fractions.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "probe_per_seed.csv", index=False)
    pd.DataFrame(confusions).to_csv(out / "confusion_counts.csv", index=False)
    # Same Gaussian feature controls as the RNN analysis, plus shuffled labels
    # for each feature view at the final round. No test data choose settings.
    random_features = np.stack([np.random.default_rng(s).standard_normal((920, 128)).astype(np.float32) for s in range(5)])
    shuffled = np.random.default_rng(20000).permutation(labels)
    control_features = np.concatenate([random_features, features[[19, 39]]])
    control_truth = np.concatenate([np.broadcast_to(labels, (5, 920)), np.broadcast_to(shuffled, (2, 920))])
    control_predictions, control_probabilities, control_models = ref.train_probes(
        control_features, control_truth, train, [10000] * 5 + [20000] * 2)
    np.savez_compressed(out / "control_models.npz", **control_models, predictions=control_predictions,
        probabilities=control_probabilities, truth=control_truth, train=train, test=test)
    controls = []
    for i, name in enumerate([f"gaussian_features_{s}" for s in range(5)] +
                             ["shuffled_cumulative_mean", "shuffled_round_end_state"]):
        metric, _ = ref.class_metrics(control_truth[i, test], control_predictions[i, test])
        controls.append(dict(version=record["version"], condition=CONDITION, training_seed=record["seed"],
                             control=name, **metric))
    pd.DataFrame(controls).to_csv(out / "controls.csv", index=False)
    write(out / "completion.json", dict(status="completed", version=record["version"], seed=record["seed"],
        primary_probes=len(features), control_probes=7, n_train=736, n_test=184,
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}, completed_utc=now()))
    print("[done probes]", record["version"], record["seed"], flush=True)


def aggregate(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    root = Path(root).resolve(); manifest = read(root / "analysis_run.json")
    verify_inputs(root, manifest)
    plots = plotting(root)
    tables = {name:pd.concat([pd.read_csv(policy_dir(root, r) / (name + ".csv"))
                             for r in manifest["policies"]], ignore_index=True)
              for name in ("prediction_fractions", "probe_per_seed", "controls", "confusion_counts")}
    for r in manifest["policies"]:
        done = read(policy_dir(root, r) / "completion.json")
        if done["primary_probes"] != 58 or done["status"] != "completed":
            raise ValueError("Incomplete policy")
    if len(tables["prediction_fractions"]) != 2400:
        raise ValueError("Missing MLP prediction curves")
    table = tables["prediction_fractions"]
    summary = plots.summarize(table)
    table.to_csv(root / "prediction_fractions_per_seed.csv", index=False)
    summary.to_csv(root / "prediction_fractions_summary.csv", index=False)
    for name in ("probe_per_seed", "controls", "confusion_counts"):
        tables[name].to_csv(root / (name + ".csv"), index=False)
    resamples = np.random.default_rng(30000).integers(0, 5, size=(10000, 5))
    metric_summary = []
    for key, group in tables["probe_per_seed"].groupby(["version", "feature_type", "axis", "cutoff"], sort=True):
        if set(group.training_seed) != set(range(1, 6)) or len(group) != 5:
            raise ValueError("Require all five policy seeds")
        row = dict(zip(("version", "feature_type", "axis", "cutoff"), key))
        for metric in ("accuracy", "balanced_accuracy", "recall_0", "recall_1"):
            values = group.sort_values("training_seed")[metric].to_numpy()
            low, high = np.quantile(values[resamples].mean(axis=1), [.025, .975])
            row.update({metric:float(values.mean()), metric+"_ci_low":float(low), metric+"_ci_high":float(high)})
        metric_summary.append(dict(**row, n_seeds=5))
    metrics = pd.DataFrame(metric_summary)
    metrics.to_csv(root / "probe_summary.csv", index=False)
    metrics[(metrics.axis == "round") & (metrics.cutoff == 20)].to_csv(root / "round20_summary.csv", index=False)
    for version in VERSIONS:
        fig = plots.comparison_figure(summary, version, CONDITION)
        for extension in ("png", "pdf"):
            fig.savefig(root / f"comparison_{version}.{extension}", dpi=170)
        plt.close(fig)
    with PdfPages(root / "comparison_individual_seeds.pdf") as book:
        for version in VERSIONS:
            for seed in range(1, 6):
                fig = plots.comparison_figure(table, version, CONDITION, seed)
                fig.savefig(root / f"comparison_{version}_seed{seed}.png", dpi=110)
                book.savefig(fig); plt.close(fig)
    fig, axes = plt.subplots(6, 2, figsize=(10, 22), sharey=True, sharex=True)
    fig.suptitle("Diverse-partner MLP · final actor hidden layer\nAll 5 training seeds · shading: 95% bootstrap CI across seeds", y=.99)
    for v, version in enumerate(VERSIONS):
        for f, feature in enumerate(FEATURE_NAMES):
            selected = summary[(summary.version == version) & (summary.feature_type == feature)]
            for col, actual in enumerate((0, 1)):
                ax = axes[2*v+f, col]
                plots.panel(ax, selected, actual, aggregated=True)
                ax.set_title(f"{version} · Ground truth: {plots.CLASS_NAMES[actual]}\n({plots.CLASS_DELAYS[actual]})", fontsize=11)
            axes[2*v+f, 0].set_ylabel(FEATURE_NAMES[feature] + "\nFraction predicted as each class", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .015), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .04, 1, .96), h_pad=2)
    for extension in ("png", "pdf"):
        fig.savefig(root / f"comparison_all_versions.{extension}", dpi=180)
    plt.close(fig)
    for axis, xlabel in (("round", "Round"), ("timestep", "Timestep cutoff")):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for ax, version in zip(axes, VERSIONS):
            for feature, color in zip(FEATURE_NAMES, ("#6653b5", "#dd8800")):
                group = metrics[(metrics.version == version) & (metrics.feature_type == feature) & (metrics.axis == axis)].sort_values("cutoff")
                label = "Cumulative mean" if feature == "cumulative_mean" else "Single activation"
                ax.plot(group.cutoff, group.balanced_accuracy, color=color, label=label)
                ax.fill_between(group.cutoff, group.balanced_accuracy_ci_low, group.balanced_accuracy_ci_high, color=color, alpha=.15)
            ax.axhline(.5, color="gray", ls="--", lw=1)
            ax.set_title(version); ax.set_xlabel(xlabel); ax.set_ylim(0, 1.02)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Held-out balanced accuracy"); axes[0].legend(frameon=False)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(root / f"relative_speed_{axis}_all_seeds.{extension}", dpi=170)
        plt.close(fig)
    report = ["# MLP relative-speed decoding", "", "15 fixed MLP policies: v1_matched, v2, v3; seeds 1–5.", "",
        "Representation: the 128-dimensional final actor hidden layer, relu(Dense_1), immediately before action logits. "
        "The MLP dummy carry is excluded. Both views use this same layer: the mean from episode start through each "
        "round's final valid transition, and the single activation at that transition before its action/outcome. "
        "Terminal transitions are included; padding is excluded. Timestep curves additionally use cutoffs 1,50,...,400.", "",
        "Original frozen model/evaluator sources, checkpoints, layouts and evaluation seed 12345 are reused. "
        "Every recorded trajectory field must match the existing evaluation exactly. No policy is retrained.", "",
        "The original 46 specialist profiles supply 20 episodes each: 920 per policy. The original PCG64 seed-0 "
        "16/4 repetition split gives 736 fitting and 184 held-out episodes, with 368/368 fitting and 92/92 testing "
        "examples per class. The target is red faster versus blue faster, not a numeric delay estimate.", "",
        "Independent affine 128→2 softmax probes reuse the frozen RNN fitter: full-batch Adam, learning rate .01, "
        "1000 updates, initialization seed 10000, no scaling/regularization/early stopping/test-based selection. "
        "Five policy seeds determine means and 95% percentile-bootstrap intervals (10000 resamples, RNG 30000). "
        "Gaussian-feature and shuffled-label controls are saved separately.", "",
        "[All-version red/blue prediction curves](comparison_all_versions.png); "
        "[round accuracy](relative_speed_round_all_seeds.png); [timestep accuracy](relative_speed_timestep_all_seeds.png). "
        "Per-version and individual-seed figures have matching comparison_*.png names. "
        "Features, exact replay audits, models, predictions, and episode manifests are under policies/.", "",
        "## Round-20 held-out balanced accuracy", "",
        "| Version | Feature | Mean | 95% CI across seeds |", "|---|---|---:|---:|"]
    for row in metrics[(metrics.axis == "round") & (metrics.cutoff == 20)].itertuples():
        report.append(f"| {row.version} | {FEATURE_NAMES[row.feature_type]} | {row.balanced_accuracy:.3f} | "
                      f"[{row.balanced_accuracy_ci_low:.3f}, {row.balanced_accuracy_ci_high:.3f}] |")
    (root / "README.md").write_text("\n".join(report) + "\n")
    manifest.update(status="completed", completed_utc=now())
    write(root / "analysis_run.json", manifest)
    print("[aggregated]", root, flush=True)


def verify(root):
    """Independently recompute probabilities, metrics, and all displayed fractions."""
    root = Path(root).resolve(); manifest = read(root / "analysis_run.json")
    verify_inputs(root, manifest)
    for record in manifest["policies"]:
        out = policy_dir(root, record)
        extraction, done = read(out / "extraction.json"), read(out / "completion.json")
        for name, digest in done["output_sha256"].items():
            if sha(out / name) != digest:
                raise ValueError("Saved result changed: " + str(out / name))
        if any(len(t["exact_trajectory_fields"]) < 10 for t in extraction["trajectories"]):
            raise ValueError("Missing exact replay checks")
        with np.load(out / "features.npz") as data, np.load(out / "probe_models.npz") as probes:
            features = np.concatenate([data[k] for k in FEATURE_NAMES] +
                                      [data["timestep_cumulative_mean"], data["timestep_state"]])
            labels, train, test = (data[k] for k in ("labels", "train", "test"))
            for i in range(58):
                logits = features[i] @ probes["weight"][i] + probes["bias"][i]
                logits -= logits.max(axis=1, keepdims=True)
                probabilities = np.exp(logits); probabilities /= probabilities.sum(axis=1, keepdims=True)
                np.testing.assert_allclose(probabilities, probes["probabilities"][i], atol=2e-5, rtol=2e-5)
                np.testing.assert_array_equal(probabilities.argmax(axis=1), probes["predictions"][i])
            episodes = pd.read_csv(out / "episode_manifest.csv")
            np.testing.assert_array_equal(episodes["class"], labels)
            np.testing.assert_array_equal(episodes.probe_partition == "train", train)
            for pair in np.unique(data["pairs"], axis=0):
                selected = (data["pairs"] == pair).all(axis=1)
                if (sum(selected & train), sum(selected & test)) != (16, 4):
                    raise ValueError("Wrong per-profile probe split")
            metrics = pd.read_csv(out / "probe_per_seed.csv")
            fractions = pd.read_csv(out / "prediction_fractions.csv")
            for i in range(58):
                y, prediction = labels[test], probes["predictions"][i, test]
                recall = [np.mean(prediction[y == k] == k) for k in (0, 1)]
                np.testing.assert_allclose(metrics.iloc[i].balanced_accuracy, np.mean(recall), atol=1e-12)
                if i < 40:
                    subset = fractions[(fractions.feature_type == probes["feature_type"][i]) &
                                       (fractions["round"] == probes["cutoff"][i])]
                    for row in subset.itertuples():
                        selected = y == row.ground_truth_class
                        count = int(sum(prediction[selected] == row.predicted_class))
                        if count != row.prediction_count or row.n_episodes != 92:
                            raise ValueError("Wrong prediction count")
                        np.testing.assert_allclose(row.prediction_fraction, count/92, atol=1e-12)
    combined = pd.read_csv(root / "prediction_fractions_per_seed.csv")
    expected = plotting(root).summarize(combined)
    saved = pd.read_csv(root / "prediction_fractions_summary.csv")
    pd.testing.assert_frame_equal(expected, saved, check_exact=False, atol=1e-12, rtol=1e-12)
    groups = ["version", "feature_type", "round", "ground_truth_class"]
    np.testing.assert_allclose(saved.groupby(groups)["mean"].sum(), 1, atol=1e-12)
    from PIL import Image
    figures = list(root.glob("*.png"))
    if len(figures) != 21:
        raise ValueError(f"Expected 21 PNG figures, found {len(figures)}")
    for path in figures:
        with Image.open(path) as image:
            image.verify()
    write(root / "independent_validation.json", dict(status="passed", policies=15,
        primary_probes=870, control_probes=105, episodes_per_policy=920, png_figures=len(figures),
        checks=["frozen_input_and_output_hashes", "original_trajectory_replay_exact",
                "real_final_actor_layer_not_dummy_carry", "terminal_included_padding_excluded",
                "original_16_4_split_each_profile", "numpy_probabilities_match_saved_probes",
                "all_held_out_metrics_recomputed", "all_prediction_counts_recomputed",
                "five_seed_bootstrap_recomputed", "two_curves_sum_to_one", "all_pngs_decode"], validated_utc=now()))
    print("[verified] 15 MLP policies, 870 primary probes, 21 PNG figures", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "extract", "fit", "worker", "aggregate", "verify"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--rnn-results", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--version", choices=VERSIONS)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.training_manifest, args.rnn_results, args.root)
    elif args.mode == "extract":
        extract_policy(args.root, args.index)
    elif args.mode == "fit":
        fit_policy(args.root, args.index)
    elif args.mode == "worker":
        manifest = read(args.root / "analysis_run.json")
        for i, record in enumerate(manifest["policies"]):
            if record["version"] == args.version:
                extract_policy(args.root, i)
                subprocess.run([manifest["analysis_python"], str(Path(__file__).resolve()), "fit", str(args.root), "--index", str(i)], check=True)
    elif args.mode == "aggregate":
        aggregate(args.root)
    else:
        verify(args.root)


if __name__ == "__main__":
    main()
