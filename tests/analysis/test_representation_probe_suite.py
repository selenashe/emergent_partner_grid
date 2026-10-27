"""Timing, action masking, and held-out independence of representation probes."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from representation_probe_features import masked_probabilities, representation_features
from relative_speed_representation import train_probes


def test_previous_memory_uses_pre_observation_carry_and_holds_its_final_valid_value():
    hidden = np.arange(2*5*3, dtype=np.float32).reshape(2,5,3) + 1
    obs = np.zeros((2,5,4)); legal = np.ones((2,5,2), bool)
    actual = representation_features("previous_memory", hidden, obs, legal, [3,5], [0,1,2,3,4])
    np.testing.assert_array_equal(actual[0], 0)
    np.testing.assert_array_equal(actual[1], hidden[:,0])
    np.testing.assert_array_equal(actual[2], hidden[:,1])
    np.testing.assert_array_equal(actual[3:,0], np.broadcast_to(hidden[0,1], (2,3)))
    np.testing.assert_array_equal(actual[4,1], hidden[1,3])


def test_current_observation_never_reads_padding_or_adds_history():
    obs = np.arange(2*5*4).reshape(2,5,4)
    actual = representation_features("current_observation", np.zeros((2,5,3)), obs,
                                     np.ones((2,5,2)), [2,4], [0,1,3,4])
    np.testing.assert_array_equal(actual[:,0], obs[0,[0,1,1,1]])
    np.testing.assert_array_equal(actual[:,1], obs[1,[0,1,3,3]])
    assert actual.dtype == np.float32


def test_masked_probabilities_ignore_illegal_scores_and_common_logit_offsets():
    logits = np.array([[1., 2., 1e6], [-1e6, 0., 3.]])
    legal = np.array([[True,True,False], [False,False,True]])
    p = masked_probabilities(logits, legal)
    np.testing.assert_allclose(p.sum(axis=-1), 1)
    np.testing.assert_array_equal(p[~legal], 0)
    np.testing.assert_allclose(p[0,:2], [1/(1+np.e),np.e/(1+np.e)])
    np.testing.assert_allclose(p, masked_probabilities(logits+1000,legal))
    with pytest.raises(ValueError,match="nonempty"):
        masked_probabilities(logits, np.zeros_like(legal))


def test_recovery_preserves_original_pool_order_instead_of_sorting_review_membership(tmp_path):
    import h5py
    from representation_probe_suite import recovery_pools
    paths = {}
    expected = {"train":[[7,1],[1,7]],"test":[[3,5],[0,5],[1,6]]}
    for name,pool in expected.items():
        path = tmp_path / (name+".h5")
        with h5py.File(path,"w") as h: h.create_dataset("capability_pool",data=pool)
        paths[name] = str(path)
    suite = dict(profiles=[dict(red_delay=0,blue_delay=1,requires_new_profile_rollout=True),
                           dict(red_delay=3,blue_delay=5,requires_new_profile_rollout=False)])
    pools = recovery_pools(suite,dict(original_rollouts=paths))
    assert pools["train"] == expected["train"]
    assert pools["test"] == expected["test"]
    assert pools["additional"] == [(0,1)]


@pytest.mark.parametrize("dimension", [1,15,128,249])
def test_affine_probe_supports_each_width_and_never_fits_held_out_labels(dimension):
    x = np.zeros((1,8,dimension), np.float32)
    x[0,:,0] = [-2, -1, 1, 2, -3, -2, 2, 3]
    labels = np.array([[0,0,1,1,0,0,1,1]])
    fit = np.array([True,True,True,True,False,False,False,False])
    pred, prob, models = train_probes(x, labels, fit, [10000], steps=150)
    changed = labels.copy(); changed[:,~fit] = 1 - changed[:,~fit]
    _, _, control = train_probes(x, changed, fit, [10000], steps=150)
    np.testing.assert_array_equal(pred, labels)
    np.testing.assert_allclose(prob.sum(axis=-1), 1, atol=1e-6)
    np.testing.assert_array_equal(models["weight"], control["weight"])
    np.testing.assert_array_equal(models["bias"], control["bias"])
    assert models["weight"].shape == (1,dimension,2)
