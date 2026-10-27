"""Meaningful invariants for discrete-state extraction and prediction fractions."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import round_state_prediction_comparison as analysis


def test_fractions_count_errors_instead_of_correct_rejection():
    truth = np.array([0, 0, 0, 0, 1, 1])
    predicted = np.array([0, 1, 1, 1, 0, 0])
    fractions, counts, n = analysis.prediction_fractions(truth, predicted, 0)
    np.testing.assert_array_equal(counts, [1, 3])
    np.testing.assert_array_equal(fractions, [.25, .75])
    assert n == 4 and fractions.sum() == 1
    with pytest.raises(ValueError, match="Empty"):
        analysis.prediction_fractions([1, 1], [1, 1], 0)
    with pytest.raises(ValueError, match="Invalid"):
        analysis.prediction_fractions([0, 2], [0, 1], 0)


def test_round_end_state_includes_terminal_but_excludes_padding():
    dones = np.array([False] * 5 + [True, True, True])
    endings = np.array([False, True, False, False, True, True, True, True])
    round_ids = np.array([0, 0, 1, 1, 1, 2, 0, 0])
    indices, length = analysis.round_end_indices(dones, endings, round_ids, n_rounds=3)
    np.testing.assert_array_equal(indices, [1, 4, 5])
    assert length == 6
    states = np.repeat(np.arange(8)[:, None], 128, axis=1).astype(float)
    selected = states[indices]
    np.testing.assert_array_equal(selected[:, 0], [1, 4, 5])
    assert selected[-1, 0] != states[:length].mean(axis=0)[0]
    assert selected.max() < 6


def test_missing_or_mislabeled_rounds_fail():
    with pytest.raises(ValueError, match="ordered"):
        analysis.round_end_indices([False, True], [False, True], [0, 0], n_rounds=2)
    with pytest.raises(ValueError, match="terminal"):
        analysis.round_end_indices([False, False], [False, True], [0, 0], n_rounds=1)


def test_summary_sums_to_one_and_preserves_paired_intervals():
    rows = []
    for feature in analysis.FEATURE_NAMES:
        for seed in range(1, 6):
            fractions = [.2 + seed / 100, .8 - seed / 100]
            for predicted in range(2):
                rows.append(dict(version="v1_matched", condition="rnn_diverse_influence", feature_type=feature,
                    round=1, ground_truth_class=1, predicted_class=predicted, training_seed=seed,
                    prediction_fraction=fractions[predicted]))
    table = pd.DataFrame(rows).sample(frac=1, random_state=13)
    summary = analysis.summarize(table)
    np.testing.assert_allclose(summary.groupby("feature_type")["mean"].sum(), 1)
    first = summary[summary.feature_type == "cumulative_mean"].sort_values("predicted_class")
    second = summary[summary.feature_type == "round_end_state"].sort_values("predicted_class")
    np.testing.assert_array_equal(first[["mean", "ci_low", "ci_high"]], second[["mean", "ci_low", "ci_high"]])
