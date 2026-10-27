"""Capability transfer must never fit on novel profiles or held-out episodes."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"eval"))
from analyze_training_capability_decoding import capability_split


def episode_data():
    profiles=np.array([(r,b) for r in range(10) for b in range(10) if r!=b])
    pairs=np.repeat(profiles,20,axis=0);reps=np.tile(np.arange(20),90)
    test=np.isin(reps,[1,9,14,15]);base_train=~test
    fit=np.array(sorted([(f,s) for f in (1,2,3) for s in (4,7,8,9)]+[(s,f) for f in (1,2,3) for s in (4,7,8,9)]))
    return pairs,reps,base_train,test,fit


def test_capability_transfer_fits_only_actual_training_profiles_and_preserves_test_episodes():
    pairs,reps,base_train,base_test,fit=episode_data()
    train,test,unused,fit_profile=capability_split(pairs,base_train,base_test,fit)
    assert (train.sum(),test.sum(),unused.sum())==(384,360,1056)
    np.testing.assert_array_equal(test,base_test)
    np.testing.assert_array_equal(np.unique(pairs[train],axis=0),fit)
    np.testing.assert_array_equal(np.bincount((pairs[train,1]<pairs[train,0]).astype(int)),[192,192])
    assert not train[~fit_profile].any()
    assert not (train&test).any()
    assert (train.astype(int)+test+unused==1).all()
    assert set(reps[train])==set(range(20))-{1,9,14,15}


def test_single_partner_actual_training_support_cannot_supply_both_faster_goal_labels():
    pairs,reps,base_train,base_test,fit=episode_data()
    with pytest.raises(ValueError,match="both faster-goal classes"):
        capability_split(pairs,base_train,base_test,np.array([[1,4]]))
