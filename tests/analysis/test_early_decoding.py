"""Temporal ordering and descriptive onset rules for dense saved-rollout analysis."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"eval"))
from analyze_early_decoding import behavior_history,N_READ,partial_correlation
from plot_early_decoding import timing,first_run


def test_outcomes_only_enter_the_history_at_the_next_observation():
    goal=np.full((2,N_READ),2,int);goal[:,3:]=1
    assignment=np.full((2,N_READ),1,int);assignment[:,3:]=2
    round_done=np.zeros((2,N_READ),bool);round_done[:,4]=True
    data=dict(pairs=np.array([[1,4],[1,4]]),lengths=np.full(2,N_READ),
        partner_assignment=assignment,ego_alloc_action=assignment,partner_goal=goal,
        round_idx=np.zeros((2,N_READ),int),is_t0=np.zeros((2,N_READ),bool),round_done=round_done,
        round_time=np.broadcast_to(np.arange(N_READ),(2,N_READ)).copy())
    result=behavior_history(data)
    assert not result['any_switch'][3].any()
    assert result['any_switch'][4].all()
    assert (result['first_switch']==4).all()
    assert not result['both_goals_sampled'][3].any()
    assert result['both_goals_sampled'][4].all()
    assert not result['round_finished'][4].any()
    assert result['round_finished'][5].all()
    assert result['action_faster'][3].mean()==1
    assert result['allocation_history_baseline'][3].mean()==0
    assert result['allocation_history_baseline'][4].mean()==1
    assert result['future_faster_next10'][2].mean()==1
    data['ego_alloc_action']=np.zeros((2,N_READ),int)
    retained=behavior_history(data)
    np.testing.assert_array_equal(retained['request_faster'],retained['action_faster'])
    assert not retained['request_available'].any()


def test_onset_definitions_identify_a_known_jump_and_reject_small_rises():
    curve=np.full(101,.5);curve[8:]=.8
    result=timing(curve)
    assert result['peak_step']==8
    assert result['peak5_start']<=8<=result['peak5_start']+5
    assert result['half_rise_step']==8
    assert result['peak_step_gain']==pytest.approx(.3)
    assert np.isnan(timing(np.full(101,.51))['half_rise_step'])
    assert np.isnan(first_run(np.zeros(101,bool),5))


def test_partial_correlation_removes_a_shared_observed_control():
    rng=np.random.default_rng(12);z=rng.normal(size=1000)
    x=2*z+rng.normal(size=1000);y=-3*z+rng.normal(size=1000)
    assert np.corrcoef(x,y)[0,1]<-.7
    value,n=partial_correlation(x,y,z[:,None])
    assert n==1000 and abs(value)<.1
