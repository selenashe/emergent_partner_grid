"""Protect assignment direction and terminal-inclusive within-round switch counts."""

import numpy as np
import pytest

from eval.analyze_assignment_behaviors import extract_arrays, round_metrics


def arrays():
    shape = (2, 11)
    dones = np.zeros(shape, bool)
    dones[:, 8:] = True
    endings = np.zeros(shape, bool)
    endings[:, [2, 5, 8]] = True
    endings[:, 9:] = True  # Scan padding must not count as extra rounds.
    starts = np.zeros(shape, bool)
    starts[:, [0, 3, 6]] = True
    starts[:, 9] = True
    assignment = np.array([[1, 2, 2, 1, 1, 1, 2, 2, 1, 99, 99],
                           [2, 2, 2, 1, 1, 1, 2, 2, 2, 99, 99]])
    switched = np.zeros(shape, bool)
    switched[0, [1, 8]] = True
    switched[:, 9:] = True
    return dict(dones=dones, round_done=endings, is_t0=starts,
                round_idx=np.tile(np.r_[np.repeat(np.arange(3), 3), 0, 0], (2, 1)),
                round_time=np.tile(np.r_[np.tile([1, 2, 3], 3), 1, 2], (2, 1)),
                partner_assignment=assignment, assignment_changed=switched)


def test_terminal_switch_counts_and_cross_round_changes_do_not():
    data = extract_arrays(arrays(), "v2", True, rounds=3)
    np.testing.assert_array_equal(data['switches'], [[1, 0, 1], [0, 0, 0]])
    np.testing.assert_array_equal(data['initial_ego_red'], [[True, True, False], [False, True, False]])
    np.testing.assert_array_equal(data['final_ego_red'], [[False, True, True], [False, True, False]])
    np.testing.assert_array_equal(data['round_steps'], np.full((2, 3), 3))


def test_goal_complement_and_equal_episode_weighting():
    data = extract_arrays(arrays(), "v2", True, rounds=3)
    data.update(profile=np.array([0, 1]), pool=np.array([[1, 4], [4, 1]]))
    metrics = round_metrics(data)
    np.testing.assert_allclose(metrics['final_partner_faster'], [.5, .5, 0])
    np.testing.assert_allclose(metrics['switches_per_round'], [.5, 0, .5])
    np.testing.assert_allclose(metrics['cumulative_switches'], [.5, .5, 1])
    np.testing.assert_allclose(metrics['final_ego_red_given_partner_faster_red'], [0, 1, 1])
    np.testing.assert_allclose(metrics['final_ego_red_given_partner_faster_blue'], [0, 1, 0])


@pytest.mark.parametrize('version,influence', [('v1_matched', True), ('v2', False), ('v3', False)])
def test_uncontrollable_assignments_cannot_switch(version, influence):
    with pytest.raises(ValueError, match='Fixed/no-influence'):
        extract_arrays(arrays(), version, influence, rounds=3)


@pytest.mark.parametrize('problem', ['no_done', 'missing_end', 'wrong_time', 'wrong_flag', 'unset_assignment'])
def test_invalid_records_fail(problem):
    a = arrays()
    if problem == 'no_done':
        a['dones'][0] = False
    elif problem == 'missing_end':
        a['round_done'][0, 2] = False
    elif problem == 'wrong_time':
        a['round_time'][0, 1] = 17
    elif problem == 'wrong_flag':
        a['assignment_changed'][0, 3] = True
    else:
        a['partner_assignment'][0, 1] = 0
    with pytest.raises(ValueError):
        extract_arrays(a, 'v3', True, rounds=3)
