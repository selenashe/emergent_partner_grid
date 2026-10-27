"""Duration measurements must count round initialization and ignore scan padding."""

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.plot_step_distributions import read_durations


@pytest.fixture
def duration_rollout(tmp_path):
    # Two rounds: one succeeds after three transitions, one times out after five.
    # Final-done padding deliberately has arbitrary success, time, and reward.
    path = tmp_path / "durations.h5"
    shape = (1, 11)
    dones = np.zeros(shape, bool)
    dones[:, 7:] = True
    round_done = np.zeros(shape, bool)
    round_done[:, [2, 7, 8, 9, 10]] = True
    success = np.zeros(shape, bool)
    success[:, [2, 8, 9, 10]] = True
    rewards = np.full(shape, -.01)
    rewards[success] = 1.
    rewards[:, 8:] = 999.
    is_t0 = np.zeros(shape, bool)
    is_t0[:, [0, 3]] = True
    round_time = np.array([[1, 2, 3, 1, 2, 3, 4, 5, 999, 999, 999]])
    round_idx = np.array([[0, 0, 0, 1, 1, 1, 1, 1, 999, 999, 999]])
    with h5py.File(path, "w") as f:
        for key, value in dict(dones=dones, round_done=round_done, success=success,
                               rewards=rewards, is_t0=is_t0, round_time=round_time,
                               round_idx=round_idx, capability_pool=np.array([[1, 4]]),
                               capability_index_per_ep=np.array([0]),
                               capability=np.broadcast_to([1, 4], shape + (2,))).items():
            f[key] = value
    return path


def test_round_initialization_terminal_and_padding(duration_rollout):
    sample = read_durations(duration_rollout, rounds_per_episode=2, max_steps=5)
    assert sample["round_steps"].tolist() == [[3, 5]]
    assert sample["episode_steps"].tolist() == [8]
    assert sample["round_success"].tolist() == [[True, False]]
    assert sample["episode_returns"].tolist() == pytest.approx([.93])


@pytest.mark.parametrize("problem", ["round_time", "early_failure", "initialization", "missing_done"])
def test_invalid_duration_records_are_rejected(duration_rollout, problem):
    with h5py.File(duration_rollout, "r+") as f:
        if problem == "round_time":
            f["round_time"][0, 2] = 2
        elif problem == "early_failure":
            f["success"][0, 2] = False
        elif problem == "initialization":
            f["is_t0"][0, 3] = False
        else:
            f["dones"][:] = False
    with pytest.raises(ValueError):
        read_durations(duration_rollout, rounds_per_episode=2, max_steps=5)
