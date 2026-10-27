"""Audit exact round-end states, matched splits, and displayed prediction fractions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def verify(root, partial=False):
    root = Path(root).resolve()
    manifest = json.loads((root / "analysis_run.json").read_text())
    assert partial or manifest["status"] == "completed"
    checks = []
    for record in manifest["policies"]:
        out = root / "policies" / record["version"] / f"{record['condition']}_seed{record['seed']}"
        if partial and not (out / "completion.json").exists():
            continue
        completion = json.loads((out / "completion.json").read_text())
        assert completion["status"] == "completed" and completion["discrete_probes"] == 20
        assert (completion["version"], completion["condition"], completion["seed"]) == (record["version"], record["condition"], record["seed"])
        for source in completion["sources"]:
            stat = Path(source["path"]).stat()
            assert (stat.st_size, stat.st_mtime_ns) == (source["size_bytes"], source["mtime_ns"])
        prior_path = Path(record["analysis_dir"]) / "probe_models.npz"
        assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == completion["cumulative_probe_sha256"]
        episodes = pd.read_csv(out / "episode_manifest.csv")
        table = pd.read_csv(out / "prediction_fractions.csv")
        assert len(episodes) == 920 and len(table) == 160
        assert (episodes.d_R != episodes.d_B).all()
        assert set(episodes.source_file) == {str(Path(p).resolve()) for p in record["original_rollouts"].values()}
        with np.load(out / "round_end_features.npz") as features, np.load(out / "round_end_probes.npz") as probe, np.load(prior_path) as prior:
            states, steps = features["features"], features["endpoint_steps"]
            assert states.shape == (20, 920, 128) and steps.shape == (20, 920)
            assert np.isfinite(states).all() and np.all(np.diff(steps, axis=0) > 0)
            np.testing.assert_array_equal(features["train"], prior["train"])
            np.testing.assert_array_equal(features["test"], prior["test"])
            np.testing.assert_array_equal(features["labels"], prior["truth"][0])
            np.testing.assert_array_equal(probe["train"], prior["train"])
            np.testing.assert_array_equal(probe["test"], prior["test"])
            np.testing.assert_array_equal(probe["labels"], prior["truth"][0])
            assert probe["weight"].shape == (20, 128, 2) and probe["bias"].shape == (20, 2)
            np.testing.assert_array_equal(probe["class_weights"], np.ones((20,2)))
            assert np.isfinite(probe["weight"]).all() and np.isfinite(probe["bias"]).all()
            np.testing.assert_allclose(probe["probabilities"].sum(axis=2), 1, atol=2e-6)
            np.testing.assert_array_equal(probe["predictions"], probe["probabilities"].argmax(axis=2))
            logits = np.einsum("pnh,phk->pnk", states, probe["weight"], optimize=True) + probe["bias"][:, None, :]
            exponents = np.exp(logits - logits.max(axis=2, keepdims=True))
            recomputed = exponents / exponents.sum(axis=2, keepdims=True)
            np.testing.assert_allclose(probe["probabilities"], recomputed, rtol=1e-4, atol=2e-6)
            # Every transition index is checked against raw round metadata. Exact
            # state vectors are additionally read directly for three episodes
            # per input population (first, middle, last).
            for path, group in episodes.groupby("source_file"):
                with h5py.File(path) as raw:
                    done, round_done, round_idx = (raw[k][:] for k in ("dones", "round_done", "round_idx"))
                    for row in group.itertuples():
                        length = int(np.flatnonzero(done[row.source_episode])[0]) + 1
                        endings = np.flatnonzero(round_done[row.source_episode, :length])
                        np.testing.assert_array_equal(steps[:, row.Index], endings)
                        np.testing.assert_array_equal(round_idx[row.source_episode, endings], np.arange(20))
                        assert endings[-1] == length - 1
                    selected = group.iloc[[0, len(group) // 2, len(group) - 1]]
                    for row in selected.itertuples():
                        direct = raw["hidden_state"][row.source_episode, steps[:, row.Index], :]
                        np.testing.assert_array_equal(states[:, row.Index], direct)
            test = prior["test"]
            truth = prior["truth"][0, test]
            np.testing.assert_array_equal(np.bincount(truth, minlength=2), [92, 92])
            for feature_type, predicted in (("cumulative_mean", prior["predictions"][9:29, test]),
                                            ("round_end_state", probe["predictions"][:, test])):
                for r in range(20):
                    for actual in range(2):
                        y = predicted[r, truth == actual]
                        counts = np.bincount(y, minlength=2)
                        subset = table[(table.feature_type == feature_type) & (table["round"] == r + 1)
                                       & (table.ground_truth_class == actual)].sort_values("predicted_class")
                        assert len(subset) == 2
                        np.testing.assert_array_equal(subset.prediction_count, counts)
                        np.testing.assert_array_equal(subset.n_episodes, [len(y)] * 2)
                        np.testing.assert_allclose(subset.prediction_fraction, counts / len(y), atol=1e-12)
                        np.testing.assert_allclose(subset.prediction_fraction.sum(), 1, atol=1e-12)
        checks.append(dict(version=record["version"], condition=record["condition"], seed=record["seed"], status="passed"))
    if not partial:
        assert len(checks) == 45
        combined = pd.read_csv(root / "prediction_fractions_per_seed.csv")
        assert len(combined) == 14400
        summary = pd.read_csv(root / "prediction_fractions_summary.csv")
        assert len(summary) == 1440
        keys = ["version", "condition", "feature_type", "round", "ground_truth_class"]
        np.testing.assert_allclose(summary.groupby(keys)["mean"].sum(), 1, atol=1e-12)
    for relative, digest in manifest["reference_sha256"].items():
        assert hashlib.sha256((root / "reference_source" / relative).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((root / "source/round_state_prediction_comparison.py").read_bytes()).hexdigest() == manifest["script_sha256"]
    payload = dict(status="passed_partial" if partial else "passed", n_policy_entries=len(checks),
        verified_utc=datetime.now(timezone.utc).isoformat(),
        checks=["same_episode_labels_and_train_test_split", "exact_final_valid_transition_per_round",
                "direct_raw_state_checks_three_episodes_per_population", "two_output_probe_shapes",
                "probability_normalization", "softmax_probabilities_recomputed_from_exact_states_and_weights", "all_prediction_counts_recomputed",
                "class_fractions_sum_to_one", "original_cumulative_predictions_reused", "frozen_recipe_hashes"],
        policy_entries=checks)
    (root / ("partial_validation.json" if partial else "independent_validation.json")).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Independent comparison audit passed for {len(checks)} policy entries.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    verify(args.root, partial=args.partial)
