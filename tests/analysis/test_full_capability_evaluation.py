"""Coverage and scientific invariants for the approved 90-profile expansion."""
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"eval"))
import full_capability_evaluation as full


@pytest.fixture
def rollout(tmp_path):
    path=tmp_path/"profile.h5"
    E,T=20,2000
    with h5py.File(path,"w") as f:
        f.attrs["allocation_protocol"]="online_v3"
        f["capability_pool"]=np.array([[1,4]],np.int32)
        f["capability_index_per_ep"]=np.zeros(E,np.int32)
        f["capability"]=np.broadcast_to([1,4],(E,T,2)).copy()
        done=np.zeros((E,T),bool);done[:,19:]=True;f["dones"]=done
        rounds=np.full((E,T),19,np.int32);rounds[:,:20]=np.arange(20);f["round_idx"]=rounds
        ends=np.zeros((E,T),bool);ends[:,:20]=True;f["round_done"]=ends
        hidden=np.full((E,T,128),10000,np.float32)
        hidden[:,:20]=np.broadcast_to(np.arange(20)[None,:,None],(E,20,128));f["hidden_state"]=hidden
        rewards=np.full((E,T),-100,np.float32);rewards[:,:20]=1;f["rewards"]=rewards
        success=np.zeros((E,T),bool);success[:,:20]=True;f["success"]=success
        f["partner_assignment"]=np.full((E,T),2,np.int32)
        f["is_t0"]=np.zeros((E,T),bool)
    return path


def test_extract_includes_terminal_excludes_padding_and_uses_actual_partner_goal(rollout):
    result=full.extract(rollout,{(1,4)},protocol="online_v3")
    assert result["features"].shape==(9,20,128)
    np.testing.assert_array_equal(result["features"][0],0)
    np.testing.assert_array_equal(result["features"][1:],9.5)
    assert set(result["episodes"].valid_length)=={20}
    assert set(result["behavior"].episode_return)=={20}
    assert set(result["behavior"].round_success_rate)=={1}
    assert set(result["behavior"].fraction_partner_assigned_faster)=={1}
    np.testing.assert_array_equal(result["repetitions"],np.arange(20))


def test_extract_rejects_protocol_mismatch_missing_terminal_and_wrong_rounds(rollout):
    with pytest.raises(ValueError,match="protocol"):
        full.extract(rollout,{(1,4)},protocol="fixed_allocation")
    with h5py.File(rollout,"r+") as f: f["round_idx"][0,4]=3
    with pytest.raises(ValueError,match="round indices"):
        full.extract(rollout,{(1,4)})
    with h5py.File(rollout,"r+") as f: f["dones"][0]=False
    with pytest.raises(ValueError,match="terminal"):
        full.extract(rollout,{(1,4)})


def test_profile_accuracy_uses_only_four_held_out_episodes_not_fitting_examples():
    pairs=np.repeat([[1,4],[4,1]],20,axis=0)
    test=np.tile(np.arange(20)<4,2)
    truth=(pairs[:,1]<pairs[:,0]).astype(int)
    predictions=np.broadcast_to(1-truth,(9,40)).copy()
    predictions[:,test]=truth[test]
    probabilities=np.stack([1-predictions,predictions],axis=-1).astype(float)
    table=full.profile_metrics(predictions,probabilities,pairs,test)
    assert len(table)==18
    assert set(table.accuracy)=={1}
    assert set(table.n_episodes)=={4}
    assert set(table.mean_correct_class_probability)=={1}
    with pytest.raises(ValueError,match="four held-out"):
        full.profile_metrics(predictions,probabilities,pairs,np.ones(40,bool))


def test_summary_requires_all_five_independent_policy_seeds():
    table=pd.DataFrame([dict(version="v2",condition="rnn_diverse_influence",d_R=5,d_B=6,cutoff=50,
        training_seed=seed,n_episodes=4,accuracy=.75) for seed in range(1,6)])
    result=full.summarize_profiles(table).iloc[0]
    assert result.accuracy_mean==result.accuracy_ci_low==result.accuracy_ci_high==.75
    with pytest.raises(ValueError,match="five policy seeds"):
        full.summarize_profiles(table.iloc[:-1])
    with pytest.raises(ValueError,match="Duplicate"):
        full.summarize_profiles(pd.concat([table,table.iloc[:1]]))


def test_review_has_all_90_unequal_profiles_and_complete_novel_cross_product():
    path=Path(__file__).resolve().parents[2]/"eval/evaluation_plans/t0fix_direct_input_capability_review_20261022/capability_review.json"
    review=full.read(path)
    profiles=review["proposed_profiles"]
    assert len(profiles)==90
    assert {(p["red_delay"],p["blue_delay"]) for p in profiles}=={(r,b) for r in range(10) for b in range(10) if r!=b}
    both={(p["red_delay"],p["blue_delay"]) for p in profiles if p["category"]=="both_delays_novel"}
    assert both=={(r,b) for r in (0,5,6) for b in (0,5,6) if r!=b}
    assert sum(p["requires_new_profile_rollout"] for p in profiles)==44
    assert [(p["red_delay"],p["blue_delay"]) for p in profiles if p["seen_by_single_partner_training"]]==[(1,4)]
