"""Check the strict runner against the unchanged released probe routine."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval import representation_analysis_strict as strict


def test_recording_preserves_source_weights_predictions_rng_and_best_checkpoint():
    upstream = strict.load_upstream()
    runner = strict.RecordedUpstreamProbe(0)
    jax = upstream["jax"]
    x = np.random.default_rng(7).normal(size=(100, 128)).astype(np.float32)
    y = np.repeat(np.arange(10), 10)
    np.random.seed(12)
    expected, _, score = upstream["train_probe"](jax.random.PRNGKey(0), x, y, n_iterations=40)
    expected_rng = np.random.get_state()
    np.random.seed(12)
    params, predictions, record = runner.fit(x, y, n_iterations=40)
    actual_rng = np.random.get_state()
    for expected_leaf, actual_leaf in zip(jax.tree_util.tree_leaves(expected.params), jax.tree_util.tree_leaves(params)):
        np.testing.assert_array_equal(expected_leaf, actual_leaf)
    np.testing.assert_array_equal(predictions, np.asarray(expected.apply_fn(expected.params, x)).argmax(axis=1))
    np.testing.assert_array_equal(expected_rng[1], actual_rng[1])
    assert expected_rng[2:] == actual_rng[2:]
    assert record["selected_update"] == int(expected.step)
    assert record["upstream_best_score"] == float(score)
    assert [r[0] for r in record["checks"]] == [1, 21, 41]
    best = max(score for _, score in record["checks"])
    assert record["selected_update"] == next(step for step, score in record["checks"] if score == best)
    for label in np.unique(y):
        assert np.count_nonzero(y[record["train_ids"]] == label) == 8
        assert np.count_nonzero(y[record["test_ids"]] == label) == 2


def test_warm_start_resets_optimizer_and_split_changes():
    runner = strict.RecordedUpstreamProbe(0)
    x = np.random.default_rng(9).normal(size=(100, 128)).astype(np.float32)
    y = np.repeat(np.arange(10), 10)
    np.random.seed(0)
    params, _, first = runner.fit(x, y, n_iterations=20)
    second_params, _, second = runner.fit(x * .9, y, params, n_iterations=20)
    assert second["selected_update"] in (1, 21)  # optimizer/update count restarted
    assert not np.array_equal(first["test_ids"], second["test_ids"])
    assert np.intersect1d(first["train_ids"], second["test_ids"]).size > 0
    upstream = strict.load_upstream()
    jax = upstream["jax"]
    # Replay the exact NumPy state reached after the first source split.
    np.random.seed(0)
    upstream["train_probe"](jax.random.PRNGKey(0), x, y, n_iterations=20)
    expected, _, _ = upstream["train_probe"](jax.random.PRNGKey(0), x * .9, y,
                                            params=params, n_iterations=20)
    for a, b in zip(jax.tree_util.tree_leaves(second_params), jax.tree_util.tree_leaves(expected.params)):
        np.testing.assert_array_equal(a, b)


def test_distance_score_uses_label_range_and_reports_exact_accuracy_separately():
    labels, predictions = np.array([0, 4, 9]), np.array([0, 5, 9])
    scores = strict.score_subset(labels, predictions, np.ones(3, bool), 9)
    assert scores["distance_accuracy"] == pytest.approx(1 - 1 / 27)
    assert scores["exact_accuracy"] == pytest.approx(2 / 3)
    assert scores["mae"] == pytest.approx(1 / 3)
