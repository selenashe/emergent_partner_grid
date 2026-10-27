"""Check cumulative reward extraction, failed rounds, and scan padding."""

import numpy as np
import pytest

from eval.plot_round_return_progression import extract_round_returns


def fixture_arrays():
    durations = np.array([list(range(2, 22)), [3]*9 + [100] + [4]*10])
    successes = np.ones((2, 20), dtype=bool)
    successes[1, 9] = False
    shape = (2, 600)
    rewards = np.full(shape, 50.0)
    dones = np.zeros(shape, dtype=bool)
    round_done = np.zeros(shape, dtype=bool)
    indices = np.full(shape, 19)
    times = np.zeros(shape, dtype=int)
    success = np.zeros(shape, dtype=bool)
    for ep in range(2):
        start = 0
        for r, length in enumerate(durations[ep]):
            end = start+length
            rewards[ep, start:end] = -.01
            indices[ep, start:end] = r
            times[ep, start:end] = np.arange(1, length+1)
            round_done[ep, end-1] = True
            if successes[ep, r]:
                rewards[ep, end-1] = 1
                success[ep, end-1] = True
            start = end
        dones[ep, start-1:] = True
        # Simulate spurious terminal flags after the first final done.
        round_done[ep, start:] = True
        success[ep, start:] = True
    return (rewards, dones, round_done, indices, times, success), durations, successes


def test_round_prefixes_include_timeouts_and_ignore_post_terminal_scan():
    arrays, durations, successes = fixture_arrays()
    cumulative, incremental, actual_steps, actual_success = extract_round_returns(*arrays)
    expected = np.where(successes, 1-.01*(durations-1), -1.)
    np.testing.assert_allclose(incremental, expected, atol=1e-12)
    np.testing.assert_allclose(cumulative, np.cumsum(expected, axis=1), atol=1e-12)
    np.testing.assert_array_equal(actual_steps, durations)
    np.testing.assert_array_equal(actual_success, successes)
    assert incremental[1, 9] == pytest.approx(-1)


def test_rejects_a_missing_round_boundary():
    arrays, _, _ = fixture_arrays()
    arrays[2][0, 1] = False
    with pytest.raises(ValueError, match="exactly 20"):
        extract_round_returns(*arrays)


def test_rejects_incorrect_round_order():
    arrays, _, _ = fixture_arrays()
    arrays[3][0, 1] = 1
    with pytest.raises(ValueError, match="ordered"):
        extract_round_returns(*arrays)


def five_seed_data():
    seeds = np.repeat(np.arange(1,6),20)
    values = (seeds*10 + np.tile(np.arange(20),5))[:,None] + np.arange(20)[None,:]
    return dict(training_seed=seeds, capability=np.tile([1,4],(100,1)),
                cumulative_return=values.astype(float), round_return=values.astype(float))


def test_five_seed_single_partner_view_and_within_seed_variance():
    from eval.plot_round_return_progression import make_views, statistics
    data = five_seed_data()
    views = make_views({('train','v3','rnn_single_influence'):data})
    single = views['training_profiles'][('v3','rnn_single_influence')]
    assert len(single['training_seed']) == 100
    rows, per_seed, _ = statistics(views)
    row = next(r for r in rows if r['view']=='training_profiles' and r['round']==1)
    assert row['n_policy_seeds'] == 5
    assert row['n_episodes'] == 100
    assert row['round_return_mean'] == pytest.approx(39.5)
    assert row['round_return_within_seed_episode_sd'] == pytest.approx(np.std(np.arange(20),ddof=1))
    assert row['round_return_seed_mean_sd'] == pytest.approx(np.std(np.arange(1,6)*10,ddof=1))
    assert len(per_seed) == 200  # five learners × 20 rounds × two views


def test_single_partner_view_requires_twenty_episodes_from_each_seed():
    from eval.plot_round_return_progression import make_views
    data = five_seed_data()
    data['capability'][0] = [4,1]
    with pytest.raises(ValueError,match='20 .* episodes per seed'):
        make_views({('train','v3','rnn_single_influence'):data})
