"""Event outcomes must enter only the following observation's information set."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"eval"))
from analyze_decoding_behavior import behavioral_history,STEPS
from analyze_replayed_motion import motion_events


def test_initial_stay_goal_does_not_count_as_exposure_and_events_enter_next_observation():
    n=len(STEPS);goal=np.ones((1,n),int);goal[:,1:4]=2
    init=np.zeros((1,n),bool);init[:,0]=True
    done=np.zeros((1,n),bool);done[:,6]=True
    data=dict(pairs=np.array([[1,4]]),lengths=np.array([200]),partner_goal=goal,is_t0=init,
        round_idx=np.zeros((1,n),int),round_done=done,success=done,
        partner_assignment=np.ones((1,n),int))
    b=behavioral_history(data)
    assert not b['any_active_goal'][0,1]
    assert b['any_active_goal'][0,2]
    assert not b['both_active_goals'][0,4]
    assert b['both_active_goals'][0,5]
    assert not b['any_goal_switch'][0,4]
    assert b['any_goal_switch'][0,5]
    assert not b['round_finished'][0,6]
    assert b['round_finished'][0,7]
    assert b['first_round_feedback'][0]==7


def test_motion_history_distinguishes_waiting_from_second_move_and_resets_on_goal_change():
    T=12;pre=np.zeros((1,T,2,2),int);post=pre.copy()
    pre[0,2:10,1,0]=1;pre[0,10:,1,0]=2;post[:]=pre
    post[0,1,1,0]=1;post[0,9,1,0]=2
    goal=np.full((1,T),2,int);goal[:,0]=0
    init=np.zeros((1,T),bool);init[:,0]=True
    data=dict(pre_positions=pre,post_positions=post,partner_goal=goal,is_t0=init,
        round_done=np.zeros((1,T),bool),pre_round=np.zeros((1,T),int),pre_clock=np.zeros((1,T),int),
        red_goal=np.full((1,T,2),4,int),blue_goal=np.full((1,T,2),5,int))
    m=motion_events(data)
    assert not m['first_partner_move_seen'][0,1]
    assert m['first_partner_move_seen'][0,2]
    assert not m['four_waits_after_move_seen'][0,5]
    assert m['four_waits_after_move_seen'][0,6]
    assert not m['second_partner_move_seen'][0,9]
    assert m['second_partner_move_seen'][0,10]
    assert not m['fast_repeat_seen'].any()
    data['partner_goal'][:,4:]=1
    assert not motion_events(data)['four_waits_after_move_seen'].any()
