"""Reconstruct current actor/critic and start-observation diagnostics from saved data."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from safetensors.numpy import load_file
from scipy.special import expit

from eval.episode_return_analysis import (BUILD, FIELDS, discount_returns, extract,
    identity, key, layout_grids, require, sha, write)


def action_mask(version, condition, phase, observed_allocation):
    phase=np.asarray(phase,bool);a=np.asarray(observed_allocation)
    ids=np.arange(15)
    legal=np.broadcast_to(ids%3==0 if version=='v1_matched' else ids%3!=0,phase.shape+(15,)).copy()
    forced=version=='v2' or (version=='v3' and condition=='rnn_diverse_noinfluence')
    initial=ids==(12+a[...,None]) if forced else np.broadcast_to(np.isin(ids,[13,14]),legal.shape)
    return np.where(phase[...,None],initial,legal)


def dense(x,params,index):
    prefix=f'params,Dense_{index},'
    return x@params[prefix+'kernel']+params[prefix+'bias']


def heads(hidden,params,legal,mlp=False):
    a=1 if mlp else 0;c=3 if mlp else 2
    logits=dense(np.maximum(dense(hidden,params,a),0),params,a+1)
    logits=np.where(legal,logits,-np.inf)
    require(np.all(legal.any(-1)),'No legal action')
    probs=np.exp(logits-np.max(logits,axis=-1,keepdims=True))
    probs/=probs.sum(-1,keepdims=True)
    values=dense(np.maximum(dense(hidden,params,c),0),params,c+1)[...,0]
    return probs,values


def gru(x,previous,params):
    """The frozen Flax GRU equations, with direct 249-value inputs."""
    stem=next(k.rsplit('ir,kernel',1)[0] for k in params if k.endswith('ir,kernel'))
    def affine(value,name):
        return value@params[stem+name+',kernel']+params.get(stem+name+',bias',0)
    reset=expit(affine(x,'ir')+affine(previous,'hr'))
    update=expit(affine(x,'iz')+affine(previous,'hz'))
    candidate=np.tanh(affine(x,'in')+reset*affine(previous,'hn'))
    return (1-update)*candidate+update*previous


def observation_inputs(grids,layout,allocation,phase):
    layout=np.asarray(layout,int);allocation=np.asarray(allocation,int)
    grid=grids[layout].reshape(layout.shape+(245,))
    return np.concatenate((grid,np.eye(3,dtype=np.float32)[allocation],np.asarray(phase,np.float32)[...,None]),axis=-1)


def case_selection(episodes):
    """Four different tail episodes with deterministic, descriptive roles."""
    tail=episodes.loc[episodes.lowest_decile].copy();require(len(tail)>=4,'Insufficient tail episodes')
    selected=[]
    for role,column,ascending in [('worst_return','episode_return',True),('near_cutoff','episode_return',False),
                                 ('most_switches','switches',False),('largest_late_gain','late_minus_early',False)]:
        row=tail.drop(index=selected).sort_values([column,'population','episode_index'],ascending=[ascending,True,True],kind='stable').iloc[0]
        selected.append(row.name)
    result=tail.loc[selected].copy();result['case_role']=['worst_return','near_cutoff','most_switches','largest_late_gain']
    return result


def _round_reduce(values,starts,ends,operation='mean'):
    E,R=starts.shape;result=np.empty((E,R),np.float64)
    for e in range(E):
        for rd in range(R):
            v=values[e,starts[e,rd]:ends[e,rd]+1]
            result[e,rd]=np.sum(v) if operation=='sum' else np.mean(v)
    return result


def analyze_policy(record,root=BUILD):
    cache=Path(root)/'t0fix_direct_input/cache'/key(record)
    target=cache/'math';target.mkdir(exist_ok=True)
    if (target/'completion.json').exists():
        cached=json.loads((target/'completion.json').read_text())
        if cached.get('producer_sha256')==sha(__file__):return str(target)
    print('[actor/critic]',key(record),flush=True)
    require(sha(record['checkpoint'])==record['checkpoint_sha256'] and sha(record['config'])==record['config_sha256'],'Changed policy')
    params=load_file(record['checkpoint']);cfg=json.loads(Path(record['config']).read_text())
    require(cfg['ACTIVATION']=='relu','Unexpected activation')
    mlp=record['condition'].startswith('mlp');influence=bool(cfg['INFLUENCE'])
    paths,grids=layout_grids(record['layouts_dir'])
    episodes=pd.read_csv(cache/'episode_metrics.csv.gz');rounds=pd.read_csv(cache/'round_metrics.csv.gz')
    round_lookup=rounds.set_index(['population','episode_index','round'])
    cases=case_selection(episodes);cases.to_csv(cache/'representative_cases.csv',index=False)
    initial_rows=[];round_rows=[];phase_rows=[];episode_rows=[];probe_rows=[];case_rows=[];switch_rows=[];mlp_rows=[]
    single={};maximum_hidden_error=0.;states=0
    for population in ('train','test'):
        with h5py.File(record[population+'_file'],'r') as f:
            a={k:f[k][:] for k in FIELDS};pairs=f['capability'][:,0,:]
            d=extract(a,record['version'],influence)
            E,T=a['dones'].shape;discounted=discount_returns(a['rewards'],a['dones'],record['gamma'])
            start=d['starts'];end=d['ends'];ep=np.arange(E)[:,None]
            values=np.zeros((E,T),np.float32);prob=np.zeros((E,T,15),np.float32)
            if not mlp:
                at_start=np.empty((E,20,128),np.float32)
                before_start=np.empty_like(at_start)
                after_start=np.empty_like(at_start)
                # Read only the necessary time prefix per episode block. No complete
                # hidden-state archive is copied to the new analysis output.
                for lo in range(0,E,16):
                    hi=min(E,lo+16);stop=int(d['lengths'][lo:hi].max())
                    hidden=f['hidden_state'][lo:hi,:stop]
                    local=np.arange(hi-lo)[:,None];begin=start[lo:hi]
                    at_start[lo:hi]=hidden[local,begin]
                    before_start[lo:hi]=hidden[local,np.maximum(begin-1,0)]
                    before_start[lo:hi][begin==0]=0
                    after_start[lo:hi]=hidden[local,begin+1]
                    valid=d['alive'][lo:hi,:stop]
                    legal=action_mask(record['version'],record['condition'],a['is_t0'][lo:hi,:stop][valid],a['ego_alloc_action'][lo:hi,:stop][valid])
                    p,v=heads(hidden[valid],params,legal)
                    prob[lo:hi,:stop][valid]=p;values[lo:hi,:stop][valid]=v
                    states+=int(valid.sum())
            probe_for_population=[]
            for offset in (0,1):
                times=start+offset
                li=a['layout_idx'][ep,times]
                phase=np.full(times.shape,offset==0)
                if offset==1:
                    alloc=a['ego_alloc_action'][ep,start]
                elif record['version']=='v2' or (record['version']=='v3' and not influence):
                    alloc=a['ego_alloc_action'][ep,start]
                else:alloc=np.zeros_like(start)
                x=observation_inputs(grids,li,alloc,phase).reshape(-1,249)
                legal=action_mask(record['version'],record['condition'],phase,alloc).reshape(-1,15)
                if mlp:
                    h=np.maximum(dense(x,params,0),0);p,v=heads(h,params,legal,True)
                    prob[ep,times]=p.reshape(E,20,15);values[ep,times]=v.reshape(E,20)
                    variants={}
                    flip=x.copy();flip[:,-1]=1-flip[:,-1]
                    _,phase_value=heads(np.maximum(dense(flip,params,0),0),params,legal,True)
                else:
                    previous=(before_start if offset==0 else at_start).reshape(-1,128)
                    factual=at_start if offset==0 else after_start
                    h=gru(x,previous,params);error=np.max(np.abs(h-factual.reshape(-1,128)),axis=1)
                    maximum_hidden_error=max(maximum_hidden_error,float(error.max()))
                    require(error.max()<.01,'Start-observation reconstruction differs from saved hidden states')
                    p,v=heads(h,params,legal)
                    variants={}
                    for label,xx,hh in [('zero_history',x,np.zeros_like(previous)),('message_red',x.copy(),previous),('message_blue',x.copy(),previous)]:
                        if label.startswith('message_'):
                            xx[:,245:248]=0;xx[:,246 if label=='message_red' else 247]=1
                        ph,vh=heads(gru(xx,hh,params),params,legal)
                        variants[label]=(ph.reshape(-1,5,3).sum(1)[:,1],vh)
                allocations=p.reshape(-1,5,3).sum(1)
                pop_ep=episodes.loc[episodes.population==population].set_index('episode_index')
                for e in range(E):
                    dr,db=map(int,pairs[e]);group=str(pop_ep.loc[e,'group']);tail=bool(pop_ep.loc[e,'lowest_decile'])
                    for rd in range(20):
                        ix=e*20+rd;t=int(times[e,rd]);red=float(allocations[ix,1]);fast=red if db<dr else float(allocations[ix,2])
                        common=dict(identity(record),population=population,episode_index=e,round=rd+1,round_tick=offset,
                            delay_red=dr,delay_blue=db,layout_index=int(li[e,rd]),last_allocation=int(alloc[e,rd]),group=group,lowest_decile=tail)
                        if offset==0:
                            initial_rows.append(dict(common,value=float(v[ix]),discounted_return=float(discounted[e,t]),
                                value_bias=float(v[ix]-discounted[e,t]),p_red=red,decision_p_partner_faster=fast,
                                **{f'p_action_{k:02d}':float(p[ix,k]) for k in range(15)}))
                        if mlp:
                            mlp_rows.append(dict(common,value=float(v[ix]),discounted_return=float(discounted[e,t]),
                                value_bias=float(v[ix]-discounted[e,t]),p_red=red,p_partner_faster=fast,
                                phase_bit_only_value_delta=float(phase_value[ix]-v[ix]),
                                **{f'p_action_{k:02d}':float(p[ix,k]) for k in range(15)}))
                        else:
                            zero=variants['zero_history'][0][ix]
                            probe_rows.append(dict(common,p_red_factual=red,p_red_zero_history=float(zero),
                                p_red_if_message_red=float(variants['message_red'][0][ix]),p_red_if_message_blue=float(variants['message_blue'][0][ix]),
                                value_factual=float(v[ix]),value_zero_history=float(variants['zero_history'][1][ix]),
                                memory_probability_effect=float(red-zero),fixed_mask_message_probability_effect=float(variants['message_red'][0][ix]-variants['message_blue'][0][ix]),
                                p_partner_faster_factual=fast,p_partner_faster_zero_history=float(zero if db<dr else (1-zero if record['version']!='v1_matched' or offset==0 else 0)),
                                allocation_choice=bool(record['version']!='v1_matched' or offset==0),
                                factual_hidden_max_abs_difference=float(error[ix]),reconstruction_within_0_01=bool(error[ix]<.01),
                                message_probe_changes_legal_mask=bool(offset==0 and (record['version']=='v2' or (record['version']=='v3' and not influence)))))
            if not mlp:
                allocations=prob.reshape(E,T,5,3).sum(2)
                red=allocations[:,:,1];blue=allocations[:,:,2]
                previous=np.concatenate([a['partner_assignment'][:,:1],a['partner_assignment'][:,:-1]],axis=1)
                requested_switch=np.where(previous==1,blue,red)
                pswitch=np.where(a['is_t0']|~d['alive'],0,requested_switch)*(1 if influence and record['version']!='v1_matched' else 0)
                entropy=-np.sum(np.where(prob>0,prob*np.log(np.maximum(prob,1e-30)),0),axis=-1)
                following=np.concatenate([values[:,1:],np.zeros((E,1),np.float32)],axis=1)
                td=a['rewards']+record['gamma']*following*~a['dones']-values
                bias=values-discounted
                meanp=_round_reduce(pswitch,start,end);sump=_round_reduce(pswitch,start,end,'sum')
                meanent=_round_reduce(entropy,start,end);meanbias=_round_reduce(bias,start,end)
                rmse=np.sqrt(_round_reduce(bias*bias,start,end));meantd=_round_reduce(td,start,end)
                for e in range(E):
                    dr,db=map(int,pairs[e]);length=int(d['lengths'][e]);pop=pop_ep.loc[e]
                    episode_rows.append(dict(identity(record),population=population,episode_index=e,
                        episode_return=float(pop.episode_return),timeouts=int(pop.timeouts),switches=int(pop.switches),
                        initial_value=float(values[e,0]),discounted_return=float(discounted[e,0]),
                        initial_value_bias=float(values[e,0]-discounted[e,0]),mean_p_switch=float(pswitch[e,:length].mean()),
                        critic_rmse=float(np.sqrt(np.mean(bias[e,:length]**2)))))
                    for rd in range(20):
                        s,t=int(start[e,rd]),int(end[e,rd]);decision=min(s+(record['version']=='v2'),t)
                        rr=round_lookup.loc[(population,e,rd+1)]
                        subsequent=pswitch[e,s+1+(record['version']=='v2'):t+1]
                        round_rows.append(dict(identity(record),population=population,episode_index=e,round=rd+1,delay_red=dr,delay_blue=db,
                            timeout=bool(rr.timeout),switches=int(rr.switches),steps=int(rr.steps),round_return=float(rr.round_return),
                            final_partner_faster=bool(rr.final_partner_faster),decision_tick=int(decision-s),decision_p_red=float(red[e,decision]),
                            decision_p_partner_faster=float(red[e,decision] if db<dr else blue[e,decision]),
                            initial_value=float(values[e,s]),initial_discounted_return=float(discounted[e,s]),initial_value_bias=float(bias[e,s]),
                            mean_p_switch=float(meanp[e,rd]),mean_subsequent_p_switch=float(subsequent.mean()) if len(subsequent) else 0.,
                            expected_switch_sum=float(sump[e,rd]),mean_entropy=float(meanent[e,rd]),mean_td=float(meantd[e,rd]),
                            terminal_td=float(td[e,t]),critic_rmse=float(rmse[e,rd]),burst_10plus=bool(rr.switches>=10)))
                    for label,mask in [('initial',a['is_t0'][e,:length]),('movement',~a['is_t0'][e,:length])]:
                        phase_rows.append(dict(identity(record),population=population,episode_index=e,phase=label,n=int(mask.sum()),
                            mean_p_switch=float(pswitch[e,:length][mask].mean()),mean_entropy=float(entropy[e,:length][mask].mean()),
                            mean_critic_bias=float(bias[e,:length][mask].mean()),critic_rmse=float(np.sqrt(np.mean(bias[e,:length][mask]**2))),mean_td=float(td[e,:length][mask].mean())))
                case_ids=set(cases.loc[cases.population==population,'episode_index'].astype(int))
                for selected,event_flag in [(d['switched'],'switch'),(np.isin(np.arange(E),list(case_ids))[:,None]&d['alive'],'case')]:
                    ee,tt=np.nonzero(selected)
                    # Write large switch tables per policy, avoiding object-heavy full
                    # trajectory dataframes across the entire 60-policy cohort.
                    if len(ee):
                        table=pd.DataFrame(dict(version=record['version'],condition=record['condition'],seed=record['seed'],population=population,
                            episode_index=ee,episode_tick=tt,round=a['round_idx'][ee,tt]+1,round_tick=a['round_time'][ee,tt]-1,
                            delay_red=pairs[ee,0],delay_blue=pairs[ee,1],is_t0=a['is_t0'][ee,tt],ego_assignment_after=a['partner_assignment'][ee,tt],
                            actual_switch=d['switched'][ee,tt],reward=a['rewards'][ee,tt],value=values[ee,tt],discounted_return=discounted[ee,tt],
                            value_bias=bias[ee,tt],one_step_td_residual=td[ee,tt],entropy=entropy[ee,tt],p_red=red[ee,tt],
                            p_partner_faster_request=np.where(pairs[ee,1]<pairs[ee,0],red[ee,tt],blue[ee,tt]),
                            p_switch=pswitch[ee,tt],p_switch_request=requested_switch[ee,tt],
                            **{f'p_action_{k:02d}':prob[ee,tt,k] for k in range(15)}))
                        (switch_rows if event_flag=='switch' else case_rows).append(table)
            if record['condition'] in ('rnn_diverse_influence','rnn_single_influence'):
                pop=episodes.loc[episodes.population==population].sort_values('episode_index')
                single[population]=dict(capability=pairs,group=pop.group.to_numpy(dtype='U20'),population=np.full(E,population,dtype='U5'),
                    episode_index=np.arange(E),episode_steps=d['lengths'],episode_return=d['returns'].sum(1),round_return=d['returns'],
                    round_success=d['success'],round_steps=d['steps'],round_switches=d['switches'],
                    initial_partner_faster=rounds.loc[rounds.population==population,'initial_partner_faster'].to_numpy().reshape(E,20),
                    final_partner_faster=rounds.loc[rounds.population==population,'final_partner_faster'].to_numpy().reshape(E,20),
                    initial_value=values[:,0],discounted_return=discounted[:,0],layout_start=a['layout_idx'][ep,start])
    for name,data in [('initial_calibration.csv',initial_rows),('rnn_round_math.csv.gz',round_rows),('rnn_episode_math.csv',episode_rows),
                      ('rnn_phase_math.csv',phase_rows),('observation_channel_probes.csv.gz',probe_rows),('mlp_start_probabilities.csv.gz',mlp_rows)]:
        if data:pd.DataFrame(data).to_csv(target/name,index=False)
    for name,data in [('switch_event_math.csv.gz',switch_rows),('case_action_probabilities.csv.gz',case_rows)]:
        if data:pd.concat(data,ignore_index=True).to_csv(target/name,index=False)
    if single:
        out=Path(root)/'single_partner_generalization/cache';out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/(key(record)+'.npz'),**{k:np.concatenate([single[p][k] for p in ('train','test')]) for k in single['train']})
        write(out/(key(record)+'.json'),dict(identity(record),checkpoint_sha256=record['checkpoint_sha256'],config_sha256=record['config_sha256'],
            episodes=920,groups=['training_partner','other_reference','held_out'],value_source='frozen critic head on original saved post-observation GRU states'))
    write(target/'completion.json',dict(identity(record),status='completed',saved_hidden_states_analyzed=states,
        maximum_start_hidden_reconstruction_error=maximum_hidden_error,observation_probes=len(probe_rows),mlp_start_observations=len(mlp_rows),
        initial_calibration_rows=len(initial_rows),new_rollouts=False,policy_training=False,
        critic_target='whole-episode discounted recorded reward, terminal included, zero terminal bootstrap',
        observation_probe='zero pre-observation GRU history or vary own previous-allocation channel; fixed factual action mask',
        mlp_scope='exact round-start and immediately following stationary observation; no fabricated full-trajectory MLP activations',producer_sha256=sha(__file__)))
    return str(target)


def reference_check(record,path):
    """Independent comparison to the executed frozen Flax network, no env rollout."""
    sys.path.insert(0,str(Path(__file__).parent))
    import full_capability_evaluation as full
    evaluator=full.load_evaluator(record)
    jax,jnp=evaluator.jax,evaluator.jnp
    cfg=json.loads(Path(record['config']).read_text());params=load_file(record['checkpoint'])
    _,grids=layout_grids(record['layouts_dir']);rng=np.random.default_rng(2026)
    n=40;li=np.arange(n);phase=np.tile([True,False],n//2)
    forced=record['version']=='v2' or (record['version']=='v3' and not cfg['INFLUENCE'])
    allocation=np.where(phase, np.tile([1,2],n//2) if forced else 0, np.tile([1,2],n//2))
    x=observation_inputs(grids,li,allocation,phase)
    previous=rng.normal(0,.2,(n,128)).astype(np.float32)
    legal=action_mask(record['version'],record['condition'],phase,allocation)
    mlp=record['condition'].startswith('mlp')
    h=np.maximum(dense(x,params,0),0) if mlp else gru(x,previous,params)
    wanted_p,wanted_v=heads(h,params,legal,mlp)
    network,init_hidden,_=evaluator.trainer_mod.build_network(cfg,15)
    obs=dict(grid=jnp.asarray(grids[li])[None],last_allocation=jnp.asarray(np.eye(3,dtype=np.float32)[allocation])[None],is_t0=jnp.asarray(phase,dtype=jnp.float32)[None])
    actual_h,actual_p,actual_v=network.apply(evaluator.load_params(record['checkpoint']),jnp.asarray(previous), (obs,jnp.zeros((1,n),bool)))
    np.testing.assert_allclose(np.asarray(actual_p.probs).reshape(n,15),wanted_p,atol=5e-6,rtol=3e-5)
    np.testing.assert_allclose(np.asarray(actual_v).reshape(n),wanted_v,atol=2e-5,rtol=3e-5)
    if not mlp:np.testing.assert_allclose(np.asarray(actual_h).reshape(n,128),h,atol=5e-6,rtol=3e-5)
    write(path,dict(identity(record),status='passed',observations=n,original_frozen_network=True,
        max_probability_error=float(np.max(np.abs(np.asarray(actual_p.probs).reshape(n,15)-wanted_p))),
        max_value_error=float(np.max(np.abs(np.asarray(actual_v).reshape(n)-wanted_v))),new_rollouts=False))
