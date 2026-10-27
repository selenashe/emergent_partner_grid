"""Recompute MLP negative controls and verify aggregated metrics independently."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "analysis_run.json").read_text())
    if sha(manifest["training_manifest"]) != manifest["training_manifest_sha256"]:
        raise ValueError("Original training manifest changed")
    if sha(Path(manifest["rnn_results"]) / "analysis_run.json") != manifest["rnn_manifest_sha256"]:
        raise ValueError("Original RNN manifest changed")
    metrics, fractions, controls = [], [], []
    for record in manifest["policies"]:
        out = root / "policies" / record["version"] / f"mlp_diverse_influence_seed{record['seed']}"
        with np.load(out / "features.npz") as data, np.load(out / "control_models.npz") as model:
            features = np.concatenate([
                np.stack([np.random.default_rng(seed).standard_normal((920, 128)).astype(np.float32)
                          for seed in range(5)]),
                data["cumulative_mean"][-1:], data["round_end_state"][-1:]])
            labels, train, test = (data[k] for k in ("labels", "train", "test"))
            truth = np.broadcast_to(labels, (7, 920)).copy()
            truth[5:] = np.random.default_rng(20000).permutation(labels)
            np.testing.assert_array_equal(truth, model["truth"])
            for k in ("train", "test"):
                np.testing.assert_array_equal(data[k], model[k])
            logits = features @ model["weight"] + model["bias"][:, None, :]
            logits -= logits.max(axis=-1, keepdims=True)
            probability = np.exp(logits); probability /= probability.sum(axis=-1, keepdims=True)
            np.testing.assert_allclose(probability, model["probabilities"], atol=2e-5, rtol=2e-5)
            np.testing.assert_array_equal(probability.argmax(axis=-1), model["predictions"])
            control = pd.read_csv(out / "controls.csv")
            for i in range(7):
                y, predicted = truth[i, test], model["predictions"][i, test]
                recalls = [np.mean(predicted[y == k] == k) for k in (0, 1)]
                np.testing.assert_allclose(control.iloc[i].balanced_accuracy, np.mean(recalls), atol=1e-12)
                counts = np.bincount(truth[i, train], minlength=2)
                np.testing.assert_allclose(model["class_weights"][i], train.sum() / (2 * counts))
        metrics.append(pd.read_csv(out / "probe_per_seed.csv"))
        fractions.append(pd.read_csv(out / "prediction_fractions.csv"))
        controls.append(control)
    for tables, filename in ((metrics, "probe_per_seed.csv"), (fractions, "prediction_fractions_per_seed.csv"),
                             (controls, "controls.csv")):
        expected = pd.concat(tables, ignore_index=True)
        pd.testing.assert_frame_equal(expected, pd.read_csv(root / filename), check_exact=False, atol=1e-12, rtol=1e-12)
    per_seed = pd.concat(metrics, ignore_index=True)
    summary = pd.read_csv(root / "probe_summary.csv")
    resamples = np.random.default_rng(30000).integers(0, 5, size=(10000, 5))
    for row in summary.itertuples():
        selected = per_seed[(per_seed.version == row.version) & (per_seed.feature_type == row.feature_type)
                            & (per_seed.axis == row.axis) & (per_seed.cutoff == row.cutoff)].sort_values("training_seed")
        np.testing.assert_array_equal(selected.training_seed, np.arange(1, 6))
        for metric in ("accuracy", "balanced_accuracy", "recall_0", "recall_1"):
            values = selected[metric].to_numpy()
            low, high = np.quantile(values[resamples].mean(axis=1), [.025, .975])
            np.testing.assert_allclose([getattr(row, metric), getattr(row, metric+"_ci_low"), getattr(row, metric+"_ci_high")],
                                       [values.mean(), low, high], atol=1e-12)
    audit = dict(status="passed", policies=len(manifest["policies"]), control_probes=105,
        checks=["original_training_and_rnn_manifests_unchanged", "gaussian_features_regenerated",
                "shuffled_targets_regenerated", "control_probabilities_predictions_metrics_and_class_weights",
                "aggregate_tables_match_all_policy_outputs", "all_metric_means_and_bootstrap_intervals_recomputed"])
    (root / "control_validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("[verified controls and metric summaries]", root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    verify(parser.parse_args().root)
