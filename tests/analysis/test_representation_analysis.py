"""Regression tests for masking, grouping, selection and independent probes."""

import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis import representation_analysis as rep


@pytest.fixture
def rollout_files(tmp_path):
    paths = {}
    for name, pairs in rep.capability_populations().items():
        labels = np.repeat(np.array(sorted(pairs), dtype=int), 20, axis=0)
        E, T = len(labels), 23
        hidden = np.broadcast_to(np.arange(T)[None, :, None], (E, T, 128)).astype(np.float32).copy()
        hidden[:, 20:] = 1e6  # finite scan padding must never contribute
        capability = np.repeat(labels[:, None, :], T, axis=1)
        capability[:, 20:] = -99  # constancy is checked only on valid states
        dones = np.zeros((E, T), bool)
        dones[:, 19:] = True
        round_done = np.ones((E, T), bool)
        rounds = np.broadcast_to(np.minimum(np.arange(T), 19), (E, T))
        path = tmp_path / f"{name}.h5"
        with h5py.File(path, "w") as f:
            for key, value in {"hidden_state": hidden, "capability": capability,
                               "dones": dones, "round_done": round_done, "round_idx": rounds,
                               "capability_pool": np.array(sorted(pairs)),
                               "capability_index_per_ep": np.repeat(np.arange(len(pairs)), 20)}.items():
                f[key] = value
        paths[name] = path
    return paths


def test_terminal_inclusive_prefix_and_round_means(rollout_files):
    data = rep.load_checkpoint_rollouts(rollout_files)
    assert data.labels.shape == (920, 2)
    assert np.all(data.lengths == 20)
    assert np.all(data.timestep_features[0] == 0)
    assert np.all(data.timestep_features[1:] == 9.5)
    assert np.all(data.final50_features == 9.5)
    np.testing.assert_allclose(data.round_features[:, 0, 0], np.arange(20) / 2)
    train, test = rep.split_masks(data, rep.make_probe_split(0))
    assert train.sum() == 736 and test.sum() == 184
    assert not np.any(train & test)
    for pair in np.unique(data.labels, axis=0):
        mask = np.all(data.labels == pair, axis=1)
        assert train[mask].sum() == 16 and test[mask].sum() == 4


@pytest.mark.parametrize("malformation", ["no_done", "changing_capability", "nonfinite_hidden", "bad_round", "wrong_profile"])
def test_malformed_rollouts_fail_loudly(rollout_files, malformation):
    with h5py.File(rollout_files["train"], "r+") as f:
        if malformation == "no_done":
            f["dones"][0] = False
        elif malformation == "changing_capability":
            f["capability"][0, 2, 0] = 99
        elif malformation == "nonfinite_hidden":
            f["hidden_state"][0, -1, 0] = np.nan
        elif malformation == "bad_round":
            f["round_idx"][0, 3] = 7
        else:
            f["capability"][0, :20] = [0, 0]
    with pytest.raises(ValueError, match="Malformed rollout"):
        rep.load_checkpoint_rollouts(rollout_files)


def test_split_is_seeded_and_never_splits_capability_profiles():
    split = rep.make_probe_split(0)
    assert split == rep.make_probe_split(0)
    assert split != rep.make_probe_split(1)
    assert len(split["train_reps"]) == 16 and len(split["test_reps"]) == 4
    assert set(split["train_reps"]).isdisjoint(split["test_reps"])
    assert set(split["permutation"]) == set(range(20))
    with pytest.raises(ValueError, match="final done"):
        rep.episode_valid_length(np.zeros(5, bool))
    assert rep.episode_valid_length(np.array([False, True, True])) == 2


def test_distance_metric_uses_ordered_delays_not_exact_accuracy():
    true, pred = np.array([0, 5, 9]), np.array([9, 4, 9])
    metrics = rep.evaluate_probe(true, pred, np.ones(3, bool))
    assert metrics["mae"] == pytest.approx(10 / 3)
    assert metrics["exact_accuracy"] == pytest.approx(1 / 3)
    assert metrics["distance_accuracy"] == pytest.approx(1 - 10 / 27)


def test_batched_probes_equal_independent_fits_and_decode_linear_signal():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(17)
    labels = np.tile(np.arange(10), 10)
    features = rng.normal(0, 0.02, (2, 100, 128)).astype(np.float32)
    for p in range(2):
        features[p, np.arange(100), labels + p * 10] = 2
    train = np.arange(100) < 80
    targets = np.stack([labels, labels])
    predictions, params = rep.train_linear_probe(features, targets, train, [11, 12])
    assert np.all(predictions[:, ~train] == targets[:, ~train])
    for p in range(2):
        single, single_params = rep.train_linear_probe(features[p:p+1], targets[p:p+1], train, [11 + p])
        np.testing.assert_array_equal(single[0], predictions[p])
        np.testing.assert_allclose(single_params["weight"][0], params["weight"][p], atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(single_params["bias"][0], params["bias"][p], atol=2e-5, rtol=2e-5)


def test_behavior_selection_prefers_final_training_return(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    for condition in rep.CONDITIONS:
        for seed in range(1, 6):
            (logs / f"{condition}_{seed}.out").write_text(
                f"[main] training done in 10s\n ep_return_mean: [+99.0, +{seed}.0]\n"
                f"[main] saved params (seed 0) -> /example/{condition}_seed{seed}.safetensors\n")
            (tmp_path / f"{condition}_seed{seed}_summary.json").write_text(
                json.dumps({"train": {"mean_ep_return": 6 - seed}, "test": {"mean_ep_return": 999 * seed}}))
    selected = rep.select_paper_style_seed(tmp_path, logs)
    assert all(info["selected_seed"] == 5 and info["exact_paper_style_final_training_return"]
               for info in selected.values())
    # One missing final-training metric forces a consistent evaluation proxy
    # for the whole condition, never mixing incomparable metrics across seeds.
    (logs / f"{rep.CONDITIONS[0]}_1.out").write_text("failed before saving\n")
    selected = rep.select_paper_style_seed(tmp_path, logs)
    assert selected[rep.CONDITIONS[0]]["selected_seed"] == 1
    assert selected[rep.CONDITIONS[0]]["eval_return_proxy"]


def test_bootstrap_unit_is_five_policy_seeds():
    rows = [{"condition": rep.CONDITIONS[0], "target": "d_R", "reference_t": 400,
             "capability_subset": "all", "training_seed": seed, "distance_accuracy": 0.75,
             "exact_accuracy": 0.2, "mae": 2.25} for seed in range(1, 6)]
    result = rep.summarize_probes(pd.DataFrame(rows), "reference_t", 30_000).iloc[0]
    assert result.n_training_seeds == 5
    assert result.distance_accuracy_std == 0
    assert result.distance_accuracy_ci_low == result.distance_accuracy_ci_high == 0.75
    with pytest.raises(ValueError, match="all five"):
        rep.summarize_probes(pd.DataFrame(rows[:-1]), "reference_t", 30_000)


def test_condition_contrasts_preserve_paired_seed_variation():
    rows = []
    for seed in range(1, 6):
        for condition in rep.CONDITIONS:
            rows.append({"condition": condition, "target": "d_R", "reference_t": 400,
                         "capability_subset": "all", "training_seed": seed,
                         "distance_accuracy": 0.5 + seed * 0.04 + (0.1 if condition == rep.CONDITIONS[0] else 0)})
    contrasts = rep.summarize_condition_differences(pd.DataFrame(rows), "reference_t", 30_000)
    assert len(contrasts) == 2
    np.testing.assert_allclose(contrasts.distance_accuracy_difference_mean, 0.1)
    np.testing.assert_allclose(contrasts.difference_ci_low, 0.1)
    np.testing.assert_allclose(contrasts.difference_ci_high, 0.1)
    assert contrasts.n_positive_seed_differences.tolist() == [5, 5]


def test_discovery_excludes_mlp_and_requires_all_fifteen(tmp_path):
    for condition in (*rep.CONDITIONS, "mlp_diverse_influence"):
        for seed in range(1, 6):
            for name in ("train", "test"):
                (tmp_path / f"{condition}_seed{seed}_{name}.h5").touch()
    assert len(rep.discover_rollout_files(tmp_path)) == 15
    (tmp_path / f"{rep.CONDITIONS[0]}_seed1_train.h5").unlink()
    with pytest.raises(FileNotFoundError, match="missing/malformed"):
        rep.discover_rollout_files(tmp_path)
