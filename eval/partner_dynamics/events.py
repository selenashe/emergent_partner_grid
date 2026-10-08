"""Instrumented movement evidence, event-aligned updates and behavioral weighting."""
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from .data import write_json
from .numeric import grouped_ci
from .geometry import savefig
import matplotlib.pyplot as plt


def run(out,config):
    with h5py.File(out/'capture.h5') as f:
        hidden=np.asarray(f['hidden_state']);before=np.asarray(f['carry_before']);valid=np.asarray(f['valid'])
        rounds=np.asarray(f['pre_state/round_idx']);goal=np.asarray(f['info/partner_goal']);times=np.asarray(f['pre_state/time'])
        obs_partner=np.asarray(f['observation/grid'])[...,-1];assignment=np.asarray(f['info/partner_assignment']);request=np.asarray(f['info/ego_alloc_action'])
        cap=np.asarray(f['capability_profile']);rep=np.asarray(f['repetition'])
    events=[];exposure=[]
    for b in range(len(hidden)):
        evidence=np.zeros(2,int);last_movement={};intervals={1:[],2:[]}
        for t in range(int(valid[b].sum())):
            same_round=t>0 and rounds[b,t]==rounds[b,t-1]
            moved=same_round and np.any(obs_partner[b,t]!=obs_partner[b,t-1]) and obs_partner[b,t].sum()>0 and obs_partner[b,t-1].sum()>0
            if moved:
                g=int(goal[b,t-1])
                if g in (1,2):
                    evidence[g-1]+=1
                    # An interval is informative about observed cadence only when no intervening goal change occurred.
                    prev=last_movement.get(g)
                    if prev is not None and rounds[b,prev]==rounds[b,t] and np.all(goal[b,prev:t]==g):intervals[g].append(t-prev)
                    last_movement[g]=t
            starts=times[b,t]==0
            request_event=same_round and request[b,t]!=request[b,t-1] and request[b,t] in (1,2) and not starts
            effective_event=same_round and assignment[b,t]!=assignment[b,t-1] and assignment[b,t-1] in (1,2)
            kinds=[]
            if starts:
                kinds.append('round_start')
                kinds.append('supplied_initial_assignment' if out.parent.name=='online_v2' else 'learned_initial_allocation')
            if moved:kinds.append('observed_partner_movement')
            if request_event:kinds.append('allocation_request_change')
            if effective_event:kinds.append('effective_assignment_change')
            for kind in kinds:
                for offset in range(-2,3):
                    j=t+offset
                    if j<0 or j>=len(valid[b]) or not valid[b,j] or rounds[b,j]!=rounds[b,t]:continue
                    events.append(dict(episode=b,repetition=int(rep[b]),d_red=int(cap[b,0]),d_blue=int(cap[b,1]),round=int(rounds[b,t]),event=kind,event_t=t,offset=offset,t=j,hidden_update_norm=float(np.linalg.norm(hidden[b,j]-before[b,j])),red_movement_evidence=int(evidence[0]),blue_movement_evidence=int(evidence[1])))
        exposure.append(dict(episode=b,repetition=int(rep[b]),d_red=int(cap[b,0]),d_blue=int(cap[b,1]),red_movements=int(evidence[0]),blue_movements=int(evidence[1]),red_intervals=intervals[1],blue_intervals=intervals[2],both_goals_observed=bool(np.all(evidence>0)),valid_steps=int(valid[b].sum())))
    pd.DataFrame(events).to_csv(out/'instrumented_event_index.csv',index=False);write_json(out/'movement_exposure.json',exposure)
    summary=[]
    frame=pd.DataFrame(events)
    for keys,part in frame.groupby(['event','offset']):
        ci=grouped_ci(part.hidden_update_norm,part.episode,config['analysis_seed'],config['bootstrap'])
        summary.append(dict(event=keys[0],offset=int(keys[1]),**ci,n_event_samples=len(part)))
    pd.DataFrame(summary).to_csv(out/'event_update_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    for event,part in pd.DataFrame(summary).groupby('event'):
        ax.errorbar(part.offset,part['mean'],yerr=[part['mean']-part.low,part.high-part['mean']],label=event)
    ax.set(xlabel='Environment steps from event',ylabel='Hidden-state update norm');ax.legend(fontsize=7);savefig(fig,out,'event_aligned_updates')
    # Correct the v2 field originally measured at forced t=0. Its first learned
    # allocation at t=1 is input-matched because ALL t=0 branches take the same
    # forced action and start from exactly the same physical state and key.
    outcome_path=out/'intervention_outcomes.json';arms=json.loads(outcome_path.read_text())
    if out.parent.name=='online_v2':
        for arm in arms:
            arm['forced_t0_probability_l1']=arm['open_loop_probability_l1']
            # Allocation probability deltas remain exact input-matched at t=1.
            arm['open_loop_first_learned_allocation_red_delta']=arm['allocation_red_probability_delta']
            arm['input_match_convention']='Identical physical state, forced t=0 action, and step key imply identical t=1 observation'
    write_json(outcome_path,arms)
    metrics_path=out/'episode_metrics.json'
    if metrics_path.exists():
        metrics=json.loads(metrics_path.read_text());n=np.asarray([r['n_steps'] for r in metrics]);fractions=np.asarray([r['faster_fraction'] for r in metrics]);weights=np.maximum(n-20,1) if out.parent.name=='online_v2' else np.full(len(n),20)
        write_json(out/'behavior_weighting.json',dict(equal_profile_equal_episode_faster_fraction=float(fractions.mean()),exposure_weighted_faster_fraction=float(np.average(fractions,weights=weights)),n_episodes=len(metrics),round_success_rate=float(np.mean([r['success'] for r in metrics])),episode_return=float(np.mean([r['reward'] for r in metrics])),total_valid_steps=int(n.sum()),both_goal_evidence_capture_fraction=float(np.mean([r['both_goals_observed'] for r in exposure]))))
    return dict(observed_movements=sum(r['red_movements']+r['blue_movements'] for r in exposure),both_goals_observed_fraction=float(np.mean([r['both_goals_observed'] for r in exposure])),n_event_windows=len(events))
