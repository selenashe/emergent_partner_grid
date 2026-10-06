"""Verify interpretation of geometry information, ties, and capability weighting."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_prep.check_geometry_goal_dependence import (
    DEFAULT_CORPUS, assignment_times, capability_profiles, correlation,
    load_layouts, prediction_limit,
)


def test_fastest_assignment_uses_the_opposite_partner_goal_delay():
    layout = dict(ego_to_red=2, ego_to_blue=6, partner_to_red=4, partner_to_blue=7)
    times, labels = assignment_times([layout], [(1, 4), (4, 1)])
    np.testing.assert_array_equal(times, [[[32, 8], [14, 17]]])
    np.testing.assert_array_equal(labels, [[-1, 1]])


def test_equal_times_are_ties_not_red_labels():
    layout = dict(ego_to_red=4, ego_to_blue=4, partner_to_red=4, partner_to_blue=4)
    times, labels = assignment_times([layout], [(1, 1)])
    np.testing.assert_array_equal(times, [[[8, 8]]])
    assert labels[0, 0] == 0
    limits = prediction_limit(labels, ["grid"])
    assert limits["non_tied_accuracy"] is None
    assert limits["tie_credited_accuracy"] == 1
    assert correlation([2], labels.ravel()) is None


def test_ties_are_excluded_from_correlations_and_separately_credited():
    assert correlation([0, 999, 1], [-1, 0, 1]) == pytest.approx(1)
    limits = prediction_limit(np.array([[1, 0, -1]]), ["grid"])
    assert limits["non_tied_accuracy"] == .5
    assert limits["tie_credited_accuracy"] == pytest.approx(2 / 3)


def test_zero_linear_correlation_can_hide_perfect_geometry_prediction():
    # A U-shaped coordinate relationship cancels Pearson r exactly.
    labels = np.array([[1, 1], [-1, -1], [-1, -1], [1, 1]])
    assert correlation(np.repeat([0, 1, 2, 3], 2), labels.ravel()) == pytest.approx(0)
    assert prediction_limit(labels, ["a", "b", "c", "d"])["non_tied_accuracy"] == 1
    assert prediction_limit(labels, [0] * 4)["non_tied_accuracy"] == .5


def test_per_grid_capability_balance_rules_out_any_geometry_only_classifier():
    labels = np.array([[1, -1], [-1, 1]])
    assert prediction_limit(labels, ["grid-a", "grid-b"])["non_tied_accuracy"] == .5
    assert prediction_limit(labels, ["same", "same"])["non_tied_accuracy"] == .5


def test_authoritative_pools_and_actual_training_corpus_remain_balanced():
    profiles, _ = capability_profiles()
    assert len(profiles) == len(set(profiles)) == 46
    layouts, _ = load_layouts(DEFAULT_CORPUS)
    assert len(layouts) == 1096
    _, labels = assignment_times(layouts, profiles)
    train = labels[:, :24]
    assert np.all((train == 1).sum(axis=1) == 12)
    assert np.all((train == -1).sum(axis=1) == 12)
    assert not np.any(train == 0)
    limits = prediction_limit(train, [r["geometry_sha256"] for r in layouts])
    assert limits["non_tied_accuracy"] == .5
    # Guard the result on the full 46-profile audit, including held-out pairs.
    assert np.all((labels == 1).sum(axis=1) == 23)
    assert np.all((labels == -1).sum(axis=1) == 23)
    assert not np.any(labels == 0)
    assert np.all(labels == np.sign(np.array(profiles)[:, 0] - np.array(profiles)[:, 1])[None, :])
