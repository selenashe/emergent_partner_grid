"""Protect terminal accounting, tails, forced masks and conditional oracle semantics."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from eval.episode_return_analysis import Oracle, discount_returns, extract, tail_selection, trend
from eval.episode_return_math import action_mask, case_selection


def rollout():
    # Two 3-tick rounds; the second resets its assignment. Extra scan rewards
    # after the first terminal must never become part of the episode return.
    a={k:np.zeros((1,8),dtype=int) for k in ('round_done','round_idx','round_time','is_t0','success','partner_assignment','partner_goal','ego_alloc_action','layout_idx')}
    a['dones']=np.array([[0,0,0,0,0,1,1,1]],bool)
    a['round_idx'][0]=[0,0,0,1,1,1,0,0]
    a['round_time'][0]=[1,2,3,1,2,3,1,1]
    a['is_t0'][0,[0,3]]=1;a['round_done'][0,[2,5]]=1;a['success'][0,[2,5]]=1
    a['partner_assignment'][0]=[1,1,1,2,2,1,0,0]
    a['rewards']=np.array([[-.01,-.01,1,-.01,-.01,1,999,999]],np.float32)
    return a


def test_terminal_reward_padding_and_round_reset_switches():
    d=extract(rollout(),'v3',True,rounds=2)
    np.testing.assert_allclose(d['returns'],[[.98,.98]],atol=1e-7)
    np.testing.assert_array_equal(d['switches'],[[0,1]])
    assert d['lengths'].tolist()==[6]


def test_uncontrollable_changes_and_missing_round_boundaries_fail():
    with pytest.raises(ValueError,match='Uncontrollable'):
        extract(rollout(),'v3',False,rounds=2)
    a=rollout();a['round_done'][0,5]=0
    with pytest.raises(ValueError,match='Incomplete rounds'):extract(a,'v3',True,rounds=2)


def test_actual_penalty_must_match_recorded_reward_ledger():
    with pytest.raises(ValueError,match='Reward ledger'):extract(rollout(),'v3',True,penalty=.03,rounds=2)


def test_discounted_target_stops_at_terminal_including_reward():
    d=discount_returns(np.array([[2.,4.,1000.]]),np.array([[False,True,True]]),.5)
    np.testing.assert_array_equal(d,[[4.,4.,0.]])


def test_decile_ties_are_stable_and_tie_inclusive_sensitivity_is_distinct():
    values=np.array([0.,0.,0.,1.,2.,3.,4.,5.,6.,7.])
    mask,rank,tied,meta=tail_selection(values,np.array(['train','test','test']+['train']*7),np.array([0,2,1,3,4,5,6,7,8,9]))
    assert np.flatnonzero(mask).tolist()==[2]
    assert tied.sum()==3 and meta['boundary_tied_episodes']==3 and mask.sum()==1
    assert rank[2]==1


def test_paired_whole_episode_trend_preserves_constant_change():
    early=np.arange(15.)
    d=trend(early,early+1)
    assert d['trend']=='increasing' and d['delta_ci_low']==d['delta_ci_high']==1


def test_case_roles_choose_four_distinct_tail_episodes():
    ep=pd.DataFrame(dict(lowest_decile=[True]*5,episode_return=[-5,-4,-3,-2,-1],population=['train']*5,
                         episode_index=range(5),switches=[0,0,10,0,0],late_minus_early=[0,0,0,2,0]))
    cases=case_selection(ep)
    assert len(cases)==4 and cases.episode_index.nunique()==4
    assert cases.episode_index.tolist()==[0,4,2,3]


def test_supplied_initial_allocation_is_forced_but_learned_choice_is_free():
    phase=np.array([True,False]);alloc=np.array([2,1])
    v2=action_mask('v2','rnn_diverse_influence',phase,alloc)
    v3=action_mask('v3','rnn_diverse_influence',phase,alloc)
    assert np.flatnonzero(v2[0]).tolist()==[14]
    assert np.flatnonzero(v3[0]).tolist()==[13,14]
    assert np.flatnonzero(v3[1]).tolist()==[1,2,4,5,7,8,10,11,13,14]


def test_oracle_conditions_on_actual_complementary_goal_and_no_influence():
    oracle=object.__new__(Oracle)
    row=SimpleNamespace(fixed_partner_red_steps=10,fixed_partner_blue_steps=1000,
        online_initial_partner_red_steps=9,online_initial_partner_blue_steps=10)
    oracle.tables={'test':{(0,1,4):row}}
    initial,best,ri,rb=oracle.values('test',0,1,4,1,'v3',False)
    assert (initial,best,ri)==(1000,10,-1)
    assert rb==pytest.approx(.91)
    assert oracle.values('test',0,1,4,1,'v3',True)==pytest.approx((10,9,.91,.92))
