"""Check phase masks and interpretation of joint movement/allocation actions."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from plot_early_action_probabilities import action_mask, joint_marginals


def test_supplied_initial_assignment_is_forced_but_learned_initial_choice_is_not():
    initial = np.array([True, True, False])
    recorded = np.array([1, 2, 1])
    for version, condition, expected in (
        ("v1_matched", "rnn_diverse_influence", [[13,14], [13,14], [0,3,6,9,12]]),
        ("v2", "rnn_diverse_influence", [[13], [14], [1,2,4,5,7,8,10,11,13,14]]),
        ("v3", "rnn_diverse_influence", [[13,14], [13,14], [1,2,4,5,7,8,10,11,13,14]]),
        ("v3", "rnn_diverse_noinfluence", [[13], [14], [1,2,4,5,7,8,10,11,13,14]]),
    ):
        actual = action_mask(version, condition, initial, recorded)
        assert [np.flatnonzero(row).tolist() for row in actual] == expected


def test_joint_marginals_preserve_correlated_move_and_allocation():
    p = np.zeros((2, 15)); p[0, 1] = .3; p[0, 14] = .7; p[1, 9] = 1
    move, allocation = joint_marginals(p)
    np.testing.assert_allclose(move, [[.3,0,0,0,.7],[0,0,0,1,0]])
    np.testing.assert_allclose(allocation, [[0,.3,.7],[1,0,0]])
