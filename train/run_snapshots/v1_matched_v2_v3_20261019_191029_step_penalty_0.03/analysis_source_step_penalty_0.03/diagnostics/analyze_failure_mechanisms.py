"""Inspect actual switch clocks, movement, profiles, and replayable failures."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.learning_diagnostics import targets,target_name,write_json,sha
from diagnostics.analyze_completed_learning import ledger,aggregate,save_compressed_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args()
    import numpy as np
    import pandas as pd
    args.out_dir.mkdir(parents=True,exist_ok=True)
    rows=targets();profiles=[];mechanisms=[];cases=[];early=[]
    for index in (2,3,7,4,5,9,11,6,10):
        row=rows[index];folder=args.run_dir/'runs'/target_name(row)
        base,_=ledger(folder,915,'heldout','baseline')
        for profile,g in base.groupby('profile'):
            dr,db=int(g.d_R.iloc[0]),int(g.d_B.iloc[0])
            profiles.append(dict(index=index,target=target_name(row),profile=int(profile),d_R=dr,d_B=db,
                partner_red_faster=dr<db,has_zero_delay=min(dr,db)==0,
                success=float(g.success.mean()),return_=float(g.return_.sum()/g.episode.nunique()),
                switches=float(g.switches.mean()),fraction_switched=float((g.switches>0).mean()),
                matching=float(g.partner_matching.mean())))
        if index in (2,3,7):
            summary=json.loads((folder/'evaluation_summary.json').read_text())
            for a in summary:
                if a['population']=='heldout' and a['update'] in (16,32,64,915):
                    early.append(dict(index=index,target=target_name(row),**a))
        path=folder/'evaluation/update0915_heldout_baseline.npz'
        with np.load(path) as z:
            valid=z['valid'];time=z['time'];rd=z['round_done'];r=z['round_idx'];assignment=z['partner_assignment'];applied=z['effective_assignment']
            switched=valid&(time>0)&(assignment>0)&(applied!=assignment)
            assert int(switched.sum())==int(base.switches.sum())
            dest_delay=np.where(applied==1,z['capability'][...,1],z['capability'][...,0])
            ctr=z['partner_move_ctr'];pm=z['partner_move_effective'];em=z['ego_move_effective']
            pos=z['agent_pos'];post=z['post/agent_pos']
            move_phase=valid&(time>0)&(~rd)
            ego_moved=(pos[:,:,0]!=post[:,:,0]).any(axis=-1)
            partner_moved=(pos[:,:,1]!=post[:,:,1]).any(axis=-1)
            ego_request=move_phase&(em!=4)
            episode=np.broadcast_to(np.arange(valid.shape[1])[None,:],valid.shape)
            group=episode*20+r
            first=np.full(valid.shape[1]*20,10000,np.int32)
            np.minimum.at(first,group[switched],time[switched])
            first=first[first<10000]
            total=int(switched.sum())
            rec=dict(index=index,target=target_name(row),switches=total,
                first_switch_median_tick=float(np.median(first)) if len(first) else None,
                fraction_switched_rounds_first_switch_after_tick10=float(np.mean(first>10)) if len(first) else None,
                fraction_switches_destination_delay_positive=float((dest_delay[switched]>0).mean()) if total else None,
                fraction_switches_partner_stays=float((pm[switched]==4).mean()) if total else None,
                fraction_switches_when_clock_ready=float((ctr[switched]==0).mean()) if total else None,
                fraction_ego_movement_requests_without_movement=float((~ego_moved[ego_request]).mean()) if ego_request.any() else None,
                fraction_partner_stays_during_movement=float((~partner_moved[move_phase]).mean()),
                fraction_ego_stays_during_movement=float((~ego_moved[move_phase]).mean()),
                movement_denominator='Valid movement-phase transitions excluding round boundaries; requested STAY excluded from blocked-request metric')
            mechanisms.append(rec)
        if index in (6,5,9):
            mode='hold_after_first_request'
            intervention,_=ledger(folder,915,'heldout',mode)
            base_ep=base.groupby('episode').agg(successes=('success','sum'),packet=('repetition','first'),
                profile=('profile','first'),d_R=('d_R','first'),d_B=('d_B','first'))
            int_ep=intervention.groupby('episode').success.sum()
            base_ep['successes_with_hold']=int_ep
            base_ep['gain']=int_ep-base_ep.successes
            order=base_ep.sort_values(['gain','successes'],ascending=[False,True])
            for ep,s in order.head(5).iterrows():
                g=base[(base.episode==ep)&~base.success]
                if len(g)==0:continue
                worst=g.sort_values('switches',ascending=False).iloc[0]
                cases.append(dict(index=index,target=target_name(row),episode=int(ep),profile=int(s.profile),packet=int(s.packet),
                    d_R=int(s.d_R),d_B=int(s.d_B),baseline_successes=int(s.successes),hold_successes=int(s.successes_with_hold),
                    success_gain=int(s.gain),round=int(worst['round'])+1,layout_index=int(worst.layout_index),
                    round_switches=int(worst.switches),first_tick=int(worst.first_tick),last_tick=int(worst.last_tick),
                    baseline_trace=str(path),hold_trace=str(folder/'evaluation/update0915_heldout_hold_after_first_request.npz'),
                    caveat='Hold intervention starts at episode beginning; this is not a local branch at the selected round.'))
        print('[mechanisms]',index,target_name(row),flush=True)
    pd.DataFrame(profiles).to_csv(args.out_dir/'final_profile_behaviors.csv',index=False)
    pd.DataFrame(mechanisms).to_csv(args.out_dir/'switch_clock_mechanisms.csv',index=False)
    pd.DataFrame(early).to_csv(args.out_dir/'early_fixed_assignment_controls.csv',index=False)
    write_json(args.out_dir/'replayable_failure_cases.json',dict(cases=cases,script_sha256=sha(__file__)))
    # Inspect historical on-policy batches exactly where v1_matched first separates.
    training=[]
    for index in (2,3):
        folder=args.run_dir/'runs'/target_name(rows[index])
        for u in range(8,21):
            with np.load(folder/'training'/f'update{u:04d}.npz') as z:
                value=z['value'].astype(float);target=z['targets'].astype(float);adv=z['advantages'].astype(float)
                time=z['pre_step_time'];cap=z['info/capability'];alloc=z['info/partner_assignment'];action=z['action']
                done=z['info/round_done'].astype(bool);success=z['info/success'].astype(bool)
                rec=dict(index=index,seed=int(rows[index]['seed']),update=u,transitions=u*65536,
                    completed_rounds=int(done.sum()),success=float(success[done].mean()),
                    critic_explained_variance=float(1-np.var(target-value)/np.var(target)),
                    advantage_mean=float(adv.mean()),advantage_sd=float(adv.std()),
                    advantage_positive_fraction=float((adv>0).mean()),
                    old_value_mean=float(value.mean()),target_mean=float(target.mean()))
                for label_,mask in [('initial_choice',time==0),('movement',time>0)]:
                    rec[label_+'_advantage_mean']=float(adv[mask].mean())
                    rec[label_+'_positive_fraction']=float((adv[mask]>0).mean())
                losses=z['epoch_minibatch_losses']
                rec['sgd_first_epoch_loss']=float(losses[0].mean());rec['sgd_last_epoch_loss']=float(losses[-1].mean())
                training.append(rec)
    pd.DataFrame(training).to_csv(args.out_dir/'matched_early_onpolicy_diagnostics.csv',index=False)

if __name__=='__main__':main()
