"""Verify completed strict outputs and retention of the preceding analysis."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
old = root.parent / "counterbalanced1096_20261002_235609"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
fingerprints = json.loads((root / "previous_results_sha256.json").read_text())
assert len(fingerprints) == 200
assert all((old / p).is_file() and sha(old / p) == h for p, h in fingerprints.items())
snapshot_sha = sha(root / "strict_runner_source.txt")
upstream = json.loads((root / "upstream_provenance.json").read_text())
assert sha(root / "upstream_do_ablation_plots.py.txt") == upstream["sha256"]
report = {"audited_utc": datetime.now(timezone.utc).isoformat(),
          "previous_result_files_unchanged": len(fingerprints),
          "regression_tests": {"passed": 3, "scope": "exact source weights, predictions, RNG, warm starts, splits, selection and metric"},
          "versions": {}}
split_manifests = []
for version, protocol in (("v1", "fixed_v1"), ("v2", "online_v2")):
    folder = root / f"{version}_balanced_training"
    metadata = json.loads((folder / "analysis_metadata.json").read_text())
    prior = json.loads((old / folder.name / "analysis_metadata.json").read_text())
    assert metadata["n_checkpoints"] == 15 and metadata["training_seeds"] == list(range(1, 6))
    assert metadata["arguments"]["allocation_protocol"] == protocol
    assert metadata["entrypoint_sha256"] == snapshot_sha
    assert metadata["upstream"]["source_sha256"] == upstream["sha256"]
    sources = lambda m: {(v["condition"], v["training_seed"]): v["sources"] for v in m["validation"]}
    assert sources(metadata) == sources(prior)
    manifest = pd.read_csv(folder / "episode_manifest.csv")
    tables = {"reference_t": pd.read_csv(folder / "probe_timestep_per_seed.csv"),
              "round_idx": pd.read_csv(folder / "probe_by_round_per_seed.csv")}
    assert len(tables["reference_t"]) == 810 and len(tables["round_idx"]) == 1800
    orientation = pd.read_csv(folder / "orientation_probe_per_seed.csv")
    assert len(orientation) == 1305
    masks = sorted((folder / "probe_splits").glob("*.npz"))
    assert len(masks) == 30
    assert len(list((folder / "checkpoint_traces").glob("*.csv"))) == 30
    assert len(list((folder / "probe_models").glob("*.npz"))) == 91
    trace_rows = 0
    for file in masks:
        condition, ending = file.stem.rsplit("_seed", 1)
        seed, axis = ending.split("_", 1)
        episodes = manifest[(manifest.condition == condition) & (manifest.training_seed == int(seed))]
        assert len(episodes) == 920
        labels = episodes[["d_R", "d_B"]].to_numpy()
        targets = {"d_R": labels[:, 0], "d_B": labels[:, 1], "larger_delay_task": labels.argmax(axis=1)}
        metric_rows = pd.concat([tables[axis], orientation[orientation[axis].notna()]], ignore_index=True)
        metric_rows = metric_rows[(metric_rows.condition == condition) & (metric_rows.training_seed == int(seed))]
        traces = pd.read_csv(folder / "checkpoint_traces" / f"{file.stem}.csv")
        trace_rows += len(traces)
        previously_trained = {k: np.zeros(920, bool) for k in targets}
        with np.load(file) as values:
            for target, step, train, test, pred in zip(values["target"], values["step"], values["train_mask"],
                                                     values["test_mask"], values["predictions"]):
                assert train.sum() == 736 and test.sum() == 184
                assert not (train & test).any() and (train | test).all()
                y = targets[str(target)]
                for label in np.unique(y):
                    assert (train & (y == label)).sum() == int(.8 * (y == label).sum())
                rows = metric_rows[(metric_rows.target == target) & (metric_rows[axis] == step)]
                assert len(rows) == 3
                prior_count = int((test & previously_trained[str(target)]).sum())
                assert (rows.n_test_previously_used_for_training == prior_count).all()
                previously_trained[str(target)] |= train
                width = 1 if target == "larger_delay_task" else 9
                for row in rows.itertuples():
                    subset = test.copy()
                    if row.capability_subset != "all":
                        subset &= episodes.capability_slice.to_numpy() == ("train" if row.capability_subset == "familiar" else "test")
                    error = np.abs(pred[subset] - y[subset])
                    assert row.n_test == subset.sum()
                    assert np.isclose(row.distance_accuracy, np.mean(1 - error / width), atol=1e-6)
                    assert np.isclose(row.exact_accuracy, np.mean(error == 0), atol=1e-6)
                    assert np.isclose(row.mae, error.mean(), atol=1e-6)
                checks = traces[(traces.target == target) & (traces[axis] == step)]
                assert checks.checked_update.tolist() == list(range(1, 1002, 20))
                assert checks.checked_score.notna().all() and checks.selected.sum() == 1
                best = checks.iloc[int(np.argmax(checks.checked_score.to_numpy()))]
                assert best.checked_update == rows.iloc[0].selected_update
                assert bool(best.selected)
    assert trace_rows == 66555
    for file in (folder / "probe_models").glob("*.npz"):
        with np.load(file) as values:
            assert np.isfinite(values["weight"]).all() and np.isfinite(values["bias"]).all()
    split_manifests.append(json.loads((folder / "probe_split.json").read_text()))
    endpoints = {}
    for axis, endpoint in (("reference_t", 400), ("round_idx", 19)):
        rows = tables[axis]
        rows = rows[(rows[axis] == endpoint) & (rows.capability_subset == "all")]
        assert len(rows) == 30
        endpoints[axis] = {"probes": 30, "test_episodes_per_fit": 184,
                           "previously_trained_min": int(rows.n_test_previously_used_for_training.min()),
                           "previously_trained_max": int(rows.n_test_previously_used_for_training.max())}
    report["versions"][version] = {"checkpoints": 15, "timestep_metric_rows": 810, "round_metric_rows": 1800,
                                    "orientation_metric_rows": 1305, "trace_rows": trace_rows,
                                    "source_inputs_match_previous": True, "saved_predictions_match_metrics": True,
                                    "scalar_stratification_verified": True, "endpoint_probe_overlap": endpoints,
                                    "elapsed_seconds": metadata["elapsed_seconds"]}
assert split_manifests[0] == split_manifests[1]
report["corresponding_splits_identical_between_versions"] = True
for stem in ("v1_v2_time_all_seeds", "v1_v2_round_all_seeds"):
    assert (root / "comparison" / f"{stem}.png").is_file()
for stem in ("strict_vs_previous_time", "strict_vs_previous_round"):
    assert (root / "protocol_comparison" / f"{stem}.png").is_file()
sizes = [p.stat().st_size for p in root.rglob("*") if p.is_file()]
assert max(sizes) < 100_000_000
report["largest_artifact_bytes"] = max(sizes)
report["all_checks_passed"] = True
(root / "completion_audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
