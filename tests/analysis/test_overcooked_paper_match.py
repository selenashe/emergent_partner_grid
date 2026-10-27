"""Protect the paper's training selection and episode-level uncertainty semantics."""

import numpy as np
import pytest

from eval.plot_overcooked_paper_match import (
    VERSIONS, STYLES, bootstrap_mean_ci, episode_returns, select_candidates,
)


def candidates():
    return [dict(version=v, condition=c, seed=s, training_return=float(s))
            for v in VERSIONS for c, _, _ in STYLES for s in range(1, 11)]


def test_seed_pool_and_training_only_selection():
    rows = candidates()
    # Held-out rankings intentionally favor the worst training seed.
    for row in rows:
        row['held_out_return'] = -row['training_return']
    assert {r['seed'] for r in select_candidates(rows, range(1, 11))} == {10}
    assert {r['seed'] for r in select_candidates(rows, range(1, 6))} == {5}
    # Each version/condition makes its own selection, rather than a common seed.
    rows[0]['training_return'] = 100
    selected = select_candidates(rows, range(1, 11))
    assert selected[0]['seed'] == 1
    assert all(r['seed'] == 10 for r in selected[1:])


def test_missing_candidate_and_tie_resolution():
    rows = candidates()
    rows[0]['training_return'] = 10
    assert select_candidates(rows, range(1, 11))[0]['seed'] == 1
    with pytest.raises(ValueError, match='Missing or duplicate'):
        select_candidates(rows[1:], range(1, 11))
    with pytest.raises(ValueError, match='Missing or duplicate'):
        select_candidates(rows + [rows[0]], range(1, 11))


def rollout():
    rewards = np.tile(np.r_[np.ones(20), np.nan, 9999.], (2, 1))
    dones = np.zeros(rewards.shape, bool)
    dones[:, 19:] = True
    complete = np.ones(rewards.shape, bool)
    indices = np.tile(np.r_[np.arange(20), 0, 1], (2, 1))
    return rewards, dones, complete, indices


def test_terminal_reward_included_and_scan_padding_ignored():
    rewards, dones, complete, indices = rollout()
    rewards[1, :20] = -.01
    np.testing.assert_allclose(episode_returns(rewards, dones, complete, indices), [20., -.2])


@pytest.mark.parametrize('issue', ['no_done', 'missing_round', 'repeated_round', 'nonfinite'])
def test_invalid_episode_fails(issue):
    rewards, dones, complete, indices = rollout()
    if issue == 'no_done':
        dones[0] = False
    elif issue == 'missing_round':
        complete[0, 0] = False
    elif issue == 'repeated_round':
        indices[0, 0] = 1
    else:
        rewards[0, 0] = np.nan
    with pytest.raises(ValueError):
        episode_returns(rewards, dones, complete, indices)


def test_bootstrap_resamples_episode_rows_across_profiles():
    # Each profile has a constant outcome. Stratifying by profile would produce
    # a zero-width interval; pooling all episodes produces genuine uncertainty.
    values = np.repeat([0., 10.], 220)
    mean, (low, high) = bootstrap_mean_ci(values)
    assert mean == 5
    assert 4 < low < mean < high < 6
    assert bootstrap_mean_ci(values) == bootstrap_mean_ci(values)
    # Duplicating evaluation rows shrinks a row-bootstrap interval. This guards
    # against bootstrapping training-seed averages or partner-profile means.
    _, (low2, high2) = bootstrap_mean_ci(np.tile(values, 4))
    assert high2 - low2 < high - low


def test_bootstrap_constant_data_and_invalid_values():
    assert bootstrap_mean_ci(np.ones(440)) == (1., [1., 1.])
    for values in ([1.], [0., np.nan], np.ones((3, 2))):
        with pytest.raises(ValueError):
            bootstrap_mean_ci(values)
