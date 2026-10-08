"""Validation-selected driven dynamics, separate regimes and paired outcome summaries."""
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from .data import write_json
from .numeric import Ridge,grouped_ci
from .geometry import savefig
import matplotlib.pyplot as plt


def evaluate(model,segments,delay):
    # Audit guide:
    # Predict the next delay-lifted hidden state using both previous lifted memory and
    # its aligned observed input. Score mean squared error against actual next states,
    # and compare with simply keeping the state unchanged. This measures predictive
    # validity on supplied segments rather than environment reward.
    #
    from .input_dsa import pairs,arrays
    a=arrays(model);x,y,u=pairs(segments,delay)
    pred=x@a['A_havok_dmd'].T+u@a['B_havok_dmd'].T
    width=segments[0][0].shape[1]
    return float(np.mean((pred[:,:width]-y[:,:width])**2)),float(np.mean((x[:,:width]-y[:,:width])**2))


def run(directory,config):
    # Audit guide:
    # Fit discovery-only hidden/input projections, split captures by repetition, and
    # choose driven-model rank and delay using validation segments. Compare held-out
    # prediction with persistence, input-omitting, and input-only controls before
    # interpreting regime distances. Fit separate early/late and faster-goal regimes
    # with fixed policy weights. Cross-policy comparisons use the seeded raw-observation
    # input basis; summarize paired carry-intervention outcomes separately from
    # observational dynamics.
    #
    import torch
    torch.set_num_threads(2)
    from .input_dsa import fit,pairs,arrays,distance
    summaries=[];all_effects=[]
    for path in sorted(directory.glob('*/capture.h5')):
        out=path.parent
        print('Dynamics',directory.name,out.name,flush=True)
        from .events import run as event_analysis
        event_summary=event_analysis(out,config)
        with h5py.File(path) as f:
            hidden=np.asarray(f['hidden_state']);encoded=np.asarray(f['encoded_input']);valid=np.asarray(f['valid']);rounds=np.asarray(f['pre_state/round_idx']);rep=np.asarray(f['repetition']);cap=np.asarray(f['capability_profile'])
            raw=np.concatenate([np.asarray(f['observation/grid']).reshape(*hidden.shape[:2],-1),np.asarray(f['observation/last_allocation']),np.asarray(f['observation/is_t0'])[...,None]],axis=-1)
        discovery=rep<config['capture_repetitions']-2;validation=rep==config['capture_repetitions']-2;test=rep==config['capture_repetitions']-1
        htrain=hidden[discovery][valid[discovery]];etrain=encoded[discovery][valid[discovery]]
        center=htrain.mean(0);ev,basis=np.linalg.eigh(np.cov((htrain-center).T));basis=basis[:,np.argsort(ev)[::-1][:40]]
        ecenter=etrain.mean(0);ee,ebasis=np.linalg.eigh(np.cov((etrain-ecenter).T));ebasis=ebasis[:,np.argsort(ee)[::-1][:20]]
        h=(hidden-center)@basis;e=(encoded-ecenter)@ebasis
        segments=[]
        for b in range(len(hidden)):
            for r in range(config['capture_history_rounds']):
                ids=np.flatnonzero(valid[b]&(rounds[b]==r))
                if len(ids)<3:continue
                if not np.all(np.diff(ids)==1):raise ValueError('Noncontiguous real trajectory')
                # Post-observation hidden[t] is driven to hidden[t+1] by encoded[t+1].
                inputs=np.r_[e[b,ids[1:]],e[b,ids[-1:]]]
                segments.append(dict(episode=b,round=r,orientation=bool(cap[b,0]<cap[b,1]),h=h[b,ids],u=inputs))
        def select(mask,group=None):
            return [(s['h'],s['u']) for s in segments if mask[s['episode']] and (group is None or group(s))]
        trainseg=select(discovery);valseg=select(validation);testseg=select(test)
        fits=[];best=None
        for delay in config['delays']:
            for rank in config['ranks']:
                model=fit(trainseg,rank,delay);mse,persist=evaluate(model,valseg,delay)
                radius=float(np.abs(np.linalg.eigvals(arrays(model)['A_v'])).max())
                fits.append(dict(delay=delay,rank=rank,validation_mse=mse,validation_persistence_mse=persist,spectral_radius=radius))
                if best is None or mse<best[0]:best=(mse,model,rank,delay)
        _,model,rank,delay=best;operators=arrays(model);mse,persist=evaluate(model,testseg,delay)
        x,y,u=pairs(trainseg,delay);xt,yt,ut=pairs(testseg,delay)
        autonomous=Ridge().fit(x,y,1.);inputonly=Ridge().fit(u,y,1.)
        width=h.shape[-1]
        auto_mse=float(np.mean((autonomous.predict(xt)[:,:width]-yt[:,:width])**2));input_mse=float(np.mean((inputonly.predict(ut)[:,:width]-yt[:,:width])**2))
        # Short-horizon rollout follows identical observed input streams; no invented environment reward.
        short=[]
        for hs,us in testseg:
            if len(hs)<=delay+5:continue
            from .input_dsa import embed_signal_torch
            lifted=np.asarray(embed_signal_torch(hs.astype(np.float32),delay));z=lifted[0]
            errs=[]
            for t in range(4):
                z=operators['A_havok_dmd']@z+operators['B_havok_dmd']@us[delay-1+t]
                errs.append(float(np.mean((z[:width]-lifted[t+1,:width])**2)))
            short.append(errs)
        pd.DataFrame(fits).to_csv(out/'dynamics_rank_delay.csv',index=False)
        np.savez_compressed(out/'dynamics_operators.npz',**operators,hidden_center=center,hidden_basis=basis,input_center=ecenter,input_basis=ebasis,rank=rank,delay=delay)
        regimes={}
        for name,group in [('early',lambda s:s['round']<2),('late',lambda s:s['round']>=2),('faster_red',lambda s:s['orientation']),('faster_blue',lambda s:not s['orientation'])]:
            fitreg=fit(select(discovery,group),rank,delay);arr=arrays(fitreg);regimes[name]=arr
            np.savez_compressed(out/('operators_'+name+'.npz'),**arr)
        distances=[]
        for a,b in [('early','late'),('faster_red','faster_blue')]:
            distances.append(dict(first=a,second=b,**distance(regimes[a],regimes[b])))
        pd.DataFrame(distances).to_csv(out/'input_dsa_distances.csv',index=False)
        # Common raw-observation basis across networks: seeded projection contains all observation leaves.
        raw_dim=raw.shape[-1];projection=np.random.default_rng(config['analysis_seed']).normal(size=(raw_dim,20))/np.sqrt(raw_dim)
        raw_input=raw@projection
        raw_segments=[]
        for s in segments:
            ids=np.flatnonzero(valid[s['episode']]&(rounds[s['episode']]==s['round']))
            rc=raw_input[s['episode'],ids];raw_segments.append(dict(**s,raw_u=np.r_[rc[1:],rc[-1:]]))
        rtrain=[(s['h'],s['raw_u']) for s in raw_segments if discovery[s['episode']]];rtest=[(s['h'],s['raw_u']) for s in raw_segments if test[s['episode']]]
        common=fit(rtrain,20,1);raw_mse,raw_persist=evaluate(common,rtest,1)
        np.savez_compressed(out/'common_input_operators.npz',**arrays(common))
        episode_errors=[]
        for episode in np.flatnonzero(test):
            epseg=[(s['h'],s['u']) for s in segments if s['episode']==episode]
            ep_mse,ep_persistence=evaluate(model,epseg,delay)
            episode_errors.append(dict(episode=int(episode),mse=ep_mse,persistence=ep_persistence))
        write_json(out/'dynamics_episode_errors.json',episode_errors)
        episode_ci=grouped_ci([row['mse'] for row in episode_errors],[row['episode'] for row in episode_errors],config['analysis_seed'],config['bootstrap'])
        summary=dict(hidden_training_pca_variance=float(np.sort(ev)[-40:].sum()/ev.sum()),input_training_pca_variance=float(np.sort(ee)[-20:].sum()/ee.sum()),prediction_target='Current-state first block of delay embedding; already observed past blocks excluded from score',equal_episode_error=episode_ci,event_evidence=event_summary,policy=out.name,selected_rank=rank,selected_delay=delay,test_mse=mse,persistence_mse=persist,input_omitting_dmd_mse=auto_mse,input_only_mse=input_mse,
            test_improvement_over_persistence=float(1-mse/persist),short_horizon_mse=np.asarray(short).mean(0).tolist(),spectral_radius=float(np.abs(np.linalg.eigvals(operators['A_v'])).max()),
            design_condition_number=float(np.linalg.cond(model.Omega.numpy())),common_raw_input_mse=raw_mse,common_raw_input_persistence_mse=raw_persist,
            dsa_distances=distances,n_train_segments=len(trainseg),n_test_segments=len(testseg),
            interpretation='Global linear approximation; regime differences reflect visited states of fixed weights. Similarity and observational fits do not establish causal memory use.')
        write_json(out/'dynamics_summary.json',summary);summaries.append(summary)
        effects=json.loads((out/'intervention_outcomes.json').read_text());frame=pd.DataFrame(effects)
        frame.to_csv(out/'paired_intervention_outcomes.csv',index=False)
        for keys,part in frame[frame.magnitude>0].groupby(['donor_kind','kind','magnitude']):
            ci=grouped_ci(part.reward_delta,part.episode,config['analysis_seed'],config['bootstrap'])
            all_effects.append(dict(policy=out.name,donor_kind=keys[0],kind=keys[1],magnitude=float(keys[2]),reward_delta=float(part.reward_delta.mean()),reward_ci_low=ci['low'],reward_ci_high=ci['high'],allocation_delta=float(part.allocation_red_probability_delta.mean()),probability_l1=float(part.open_loop_probability_l1.mean()),n_recipients=part.episode.nunique()))
        fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].bar(['Driven','Persistence','Input omitted','Input only'],[mse,persist,auto_mse,input_mse]);axes[0].tick_params(axis='x',rotation=35);axes[0].set(ylabel='Held-out lifted-state MSE',title='Predictive validity before interpretation')
        for kind,part in frame[(frame.donor_kind=='opposite')&(frame.magnitude>0)].groupby('kind'):
            means=part.groupby('magnitude').reward_delta.mean();axes[1].plot(means.index,means.values,'o-',label=kind)
        axes[1].set(xlabel='Perturbation magnitude',ylabel='Paired round reward difference');axes[1].legend(fontsize=7);savefig(fig,out,'dynamics_and_interventions')
    if not summaries:
        write_json(directory/'dynamics_summary.json',dict(status='blocked',reason='No richer captures available; encoded inputs cannot be reconstructed from historical hidden states alone.'));return
    pd.DataFrame(summaries).drop(columns=['dsa_distances']).to_csv(directory/'dynamics_policy_summary.csv',index=False)
    pd.DataFrame(all_effects).to_csv(directory/'intervention_summary.csv',index=False)
    # Cross-network distances use the common raw-observation input basis, never learned encoder coordinates.
    policies=[s['policy'] for s in summaries];matrices={k:np.zeros((len(policies),len(policies))) for k in ('joint','intrinsic','input')}
    loaded={p:dict(np.load(directory/p/'common_input_operators.npz')) for p in policies}
    for i,a in enumerate(policies):
        for j in range(i):
            dist=distance(loaded[a],loaded[policies[j]])
            for k,v in dist.items():matrices[k][i,j]=matrices[k][j,i]=v
    for k,matrix in matrices.items():pd.DataFrame(matrix,index=policies,columns=policies).to_csv(directory/('common_input_'+k+'_distances.csv'))
    write_json(directory/'dynamics_summary.json',dict(status='completed',policies=len(summaries),median_test_improvement_over_persistence=float(np.median([s['test_improvement_over_persistence'] for s in summaries])),
        limitations=['InputDSA uses pinned official DMDc/controllability components with explicit safe transition pairing.', 'Unmodified upstream list concatenation would create cross-trajectory transitions; adapter corrects this boundary assembly.', 'Native distance ordering is joint/state/control; exported fields are named explicitly.', 'Regime matrices reflect the same nonlinear weights visited in different states.', 'Common raw input projection is a fixed lossy basis; inspect its held-out prediction error before cross-policy comparisons.', 'Only the prespecified early four-round diagnostic capture is fitted; full late-episode input dynamics require additional collection.', 'Bootstrap of paired recipients is shown per policy; effects must also be compared across learner seeds.']))
