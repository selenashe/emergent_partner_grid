"""Isolated frozen-source collection, exact replay, local derivatives and paired tests."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import hashlib
import numpy as np
import h5py
from .data import ROOT,resolve_path,write_json,sha


def load_frozen(config,version):
    # Audit guide:
    # Load the manifest-selected historical trainer and environment in an isolated
    # subprocess context. Check that the imported environment actually comes from that
    # frozen tree. Active online-v2 code cannot substitute for a fixed-v1 checkpoint.
    #
    manifest=json.loads((ROOT/'train/manifests'/('sbatch_'+config['batch']+'.json')).read_text())
    source=resolve_path(manifest['versions'][version]['frozen_source_root'])
    sys.path.insert(0,str(source));sys.path.insert(0,str(source/'baselines/IPPO'))
    spec=importlib.util.spec_from_file_location('partner_dynamics_frozen_trainer',source/'baselines/IPPO/ippo_rnn_coordination_grid.py')
    trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
    from jaxmarl.environments.coordination_grid import CoordinationGrid
    from jaxmarl.wrappers.baselines import load_params
    from jaxmarl.environments.coordination_grid import coordination_grid as envmod
    if Path(envmod.__file__).resolve().is_relative_to(source.resolve()) is False:
        raise ValueError('Frozen environment import isolation failed')
    return trainer,CoordinationGrid,load_params,manifest,source


def diagnostic_layouts(env,count):
    """Prespecified distance-gap strata, independent of policy performance."""
    # Audit guide:
    # Choose physical contexts by coordinate-distance strata before inspecting policy
    # performance. This diversity selection uses Manhattan coordinates rather than BFS
    # distances and does not redefine the corpus or oracle.
    #
    e=np.asarray(env.ego_starts);p=np.asarray(env.partner_starts)
    # Coordinate-distance strata are used only to choose diverse physical contexts.
    gap=np.sum(np.abs(e-np.asarray(env.red_goals)),1)-np.sum(np.abs(e-np.asarray(env.blue_goals)),1)
    order=np.lexsort((np.arange(len(gap)),np.sum(np.abs(e-p),1),gap))
    return order[np.linspace(0,len(order)-1,count,dtype=int)].tolist()


def run(config,version,out,force=False):
    # Audit guide:
    # Collect short checkpoint histories with detailed inputs, memory, state, and random
    # keys, then verify exact one-step replay. Learn a projector from discovery
    # histories, check derivatives against finite differences, and compare paired carry
    # perturbations on held-out recipients. This routine performs targeted diagnostic
    # evaluation, not policy retraining. Early-history, nuisance-memory, and forced-
    # initialization limitations remain explicit in saved summaries.
    #
    os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('OMP_NUM_THREADS','2')
    import jax
    import jax.numpy as jnp
    import distrax
    from dataclasses import fields
    from .adapters import PolicyAdapter
    from .interventions import partner_projector,perturb
    from .numeric import grouped_ci
    trainer,Env,load_params,manifest,source=load_frozen(config,version)
    checkpoint_root=resolve_path(manifest['versions'][version]['checkpoint_root'])
    profiles=np.asarray(config['capture_profiles'],np.int32)
    reps=config['capture_repetitions'];caps=np.repeat(profiles,reps,axis=0);B=len(caps)
    if reps<4:raise ValueError('Need at least 4 histories per profile: 2 discovery, 1 validation, 1 test')
    histories=config['capture_history_rounds'];max_capture=histories*100
    kwargs0=None;adapter=None;roll=None
    for condition in config['conditions']:
        cfg=json.loads((checkpoint_root/f'{condition}_seed{config["seeds"][0]}_config.json').read_text())
        kwargs=dict(cfg['ENV_KWARGS']);kwargs['layouts_dir']=str(resolve_path(kwargs['layouts_dir']));kwargs['augment_symmetries']=False;kwargs['influence']=bool(cfg['INFLUENCE']);kwargs['partner_capability_pairs']=profiles.tolist()
        env=Env(**kwargs)
        layouts=diagnostic_layouts(env,config['diagnostic_layout_count'])
        write_json(out/'diagnostic_layouts.json',dict(indices=layouts,selection='Coordinate-distance gap strata; no checkpoint outcomes used',n_rounds=histories,profiles=profiles.tolist(),repetitions=reps))
        sequences=np.asarray([[layouts[(rep+r)%len(layouts)] for r in range(20)] for rep in range(reps)],np.int32)
        seq=jnp.asarray(np.tile(sequences,(len(profiles),1)))
        # Same physical layout histories and random initialization keys across profiles.
        reset_keys=jnp.tile(jax.random.split(jax.random.PRNGKey(config['analysis_seed']),reps),(len(profiles),1))
        observations,states=(jax.vmap(env.reset_from_schedule)(jnp.asarray(caps),seq,reset_keys) if version=='v2' else jax.vmap(env.reset_from_schedule)(jnp.asarray(caps),seq))
        network,initialize,_=trainer.build_network(cfg,env.n_ego_actions)
        adapter=PolicyAdapter(trainer,network,cfg)
        state_names=[f.name for f in fields(states)]
        def select(probs,key):
            # Original categorical selection; retain the exact shape used by original evaluator.
            dist=distrax.Categorical(probs=probs[None])
            if hasattr(trainer,'select_action'):
                return trainer.select_action(dist,key,cfg.get('ACTION_SELECTION','categorical'))[0]
            return dist.sample(seed=key)[0]
        def body(params,carry,t):
            obs,state,h,ended,key=carry
            nh,probs,value,e,logits=adapter.full(params,h,obs['agent_0'],jnp.zeros(B,bool))
            key,ka,ks=jax.random.split(key,3);sk=jax.random.split(ks,B);actions=select(probs,ka)
            obs2,state2,reward,done,info=jax.vmap(env.step_env,in_axes=(0,0,{'agent_0':0}))(sk,state,{'agent_0':actions})
            final=ended|(state2.round_idx>=histories)|done['__all__']
            nh=jnp.where(ended[:,None],h,nh)
            record=dict(carry_before=h,hidden_state=nh,encoded_input=e,observation=obs['agent_0'],pre_state={n:getattr(state,n) for n in state_names},post_state={n:getattr(state2,n) for n in state_names},probabilities=probs,logits=logits,value=value,flat_action=actions,action_key=jnp.tile(ka[None],(B,1)),step_key=sk,rewards=reward['agent_0'],dones=done['__all__'],valid=~ended,history_complete=final&~ended,info=info,reset_flag=jnp.zeros(B,bool))
            state2=jax.tree_util.tree_map(lambda a,b:jnp.where(ended.reshape((B,)+(1,)*(a.ndim-1)),a,b),state,state2)
            obs2=jax.tree_util.tree_map(lambda a,b:jnp.where(ended.reshape((B,)+(1,)*(a.ndim-1)),a,b),obs,obs2)
            return (obs2,state2,nh,final,key),record
        @jax.jit
        def rollout(params):
            return jax.lax.scan(lambda c,t:body(params,c,t),(observations,states,initialize(B),jnp.zeros(B,bool),jax.random.PRNGKey(config['analysis_seed']+10)),jnp.arange(max_capture))
        @jax.jit
        def branch(params,state,h,key):
            """One physical round, identical recipient state/key across perturbation arms."""
            initial_round=state.round_idx
            obs=env.get_obs(state)
            def step(carry,_):
                obs,state,h,finished,key,total,success,steps,switches=carry
                nh,probs,value,e,logits=adapter.full(params,h[None],jax.tree_util.tree_map(lambda a:a[None],obs['agent_0']),jnp.zeros(1,bool))
                key,ka,ks=jax.random.split(key,3);action=select(probs,ka)[0]
                obs2,state2,reward,done,info=env.step_env(ks,state,{'agent_0':action})
                valid=~finished
                now_done=(state2.round_idx!=initial_round)|done['__all__']
                switches= switches + (info.get('assignment_changed',jnp.bool_(False)) & valid).astype(jnp.int32)
                return (obs2,state2,nh[0],finished|now_done,key,total+jnp.where(valid,reward['agent_0'],0.),success|(info['success']&valid),steps+valid.astype(jnp.int32),switches),probs[0]
            initial=(obs,state,h,jnp.bool_(False),key,jnp.float32(0),jnp.bool_(False),jnp.int32(0),jnp.int32(0))
            carry,probs=jax.lax.scan(step,initial,None,length=env.max_steps)
            return dict(reward=carry[5],success=carry[6],steps=carry[7],switches=carry[8],first_probabilities=probs[0],learned_probabilities=probs[1] if version=='v2' else probs[0])
        for seed in config['seeds']:
            directory=out/f'{condition}_seed{seed}';directory.mkdir(parents=True,exist_ok=True)
            if (directory/'capture_summary.json').exists() and not force:
                print('Cached capture',directory.name,flush=True);continue
            print('Collect',version,condition,seed,flush=True)
            path=checkpoint_root/f'{condition}_seed{seed}.safetensors';params=load_params(path)
            final,record=rollout(params);jax.block_until_ready(final[2]);record_np=jax.tree_util.tree_map(lambda a:np.swapaxes(np.asarray(a),0,1),record)
            final_h=np.asarray(final[2]);valid=record_np['valid'];lengths=valid.sum(1)
            if not np.all(np.asarray(final[3])):raise ValueError('Capture history did not finish within bound')
            # Verify normalized encoder/cell/readout and exact categorical replay at valid samples.
            replay_errors=[];sample_agreement=[]
            for b in range(B):
                t=min(5,int(lengths[b])-1);obs={k:jnp.asarray(v[b,t])[None] for k,v in record_np['observation'].items()}
                h=jnp.asarray(record_np['carry_before'][b,t])[None];e=jnp.asarray(record_np['encoded_input'][b,t])[None]
                nh=adapter.update(params,h,e);p,v,legal=adapter.readout(params,nh,obs)
                replay_errors.extend([float(np.max(np.abs(np.asarray(nh[0])-record_np['hidden_state'][b,t]))),float(np.max(np.abs(np.asarray(p[0])-record_np['probabilities'][b,t]))),float(np.max(np.abs(np.asarray(v[0])-record_np['value'][b,t])))])
                # Original action key samples the full B-slot batch; reproduce that shape, not a changed single-slot draw.
                replay=select(jnp.asarray(record_np['probabilities'][:,t]),jnp.asarray(record_np['action_key'][0,t]))
                sample_agreement.append(bool(np.asarray(replay)[b]==record_np['flat_action'][b,t]))
            if max(replay_errors)>1e-4 or not all(sample_agreement):raise ValueError('One-step replay equivalence failed')
            def save_tree(group,tree):
                for k,v in tree.items():
                    if isinstance(v,dict):save_tree(group.create_group(k),v)
                    else:group.create_dataset(k,data=v,compression='gzip',compression_opts=1)
            with h5py.File(directory/'capture.h5','w') as f:
                f.attrs.update(dataset_kind='new_targeted_evaluation',protocol=manifest['versions'][version]['allocation_protocol'],checkpoint_sha256=sha(path),source=str(source),history_rounds=histories)
                save_tree(f,record_np);f['capability_profile']=caps;f['repetition']=np.tile(np.arange(reps),len(profiles));f['history_final_carry']=final_h;f['history_length']=lengths
            discovery=np.tile(np.arange(reps)<reps-2,len(profiles));validation=np.tile(np.arange(reps)==reps-2,len(profiles));test=np.tile(np.arange(reps)==reps-1,len(profiles))
            q,reference=partner_projector(final_h[discovery],caps[discovery]);np.savez_compressed(directory/'partner_projector.npz',basis=q,reference=reference)
            # Validation of discovery-only projector against label contrasts (no intervention outcomes used).
            orient=(caps[:,0]<caps[:,1]).astype(float);centroids=[final_h[discovery&(orient==o)].mean(0) for o in (0,1)]
            projection=final_h@q
            centers=np.asarray(centroids)@q
            pred=np.linalg.norm(projection[:,None]-centers[None],axis=-1).argmin(1)
            validation_accuracy=float((pred[validation]==orient[validation]).mean())
            derivative=[]
            rng=np.random.default_rng(config['analysis_seed'])
            for b in np.flatnonzero(test):
                t=min(10,int(lengths[b])-2);h=jnp.asarray(record_np['carry_before'][b,t]);e=jnp.asarray(record_np['encoded_input'][b,t]);jh=np.asarray(adapter.jac_h(params,h,e));je=np.asarray(adapter.jac_e(params,h,e));direction=rng.normal(size=h.shape);direction/=np.linalg.norm(direction);eps=1e-3
                fd=np.asarray((adapter.update(params,(h+eps*direction)[None],e[None])-adapter.update(params,(h-eps*direction)[None],e[None]))[0]/(2*eps))
                fde=np.asarray((adapter.update(params,h[None],(e+eps*direction)[None])-adapter.update(params,h[None],(e-eps*direction)[None]))[0]/(2*eps))
                product=np.eye(len(h));gains=[]
                for offset in range(4):
                    hh=jnp.asarray(record_np['carry_before'][b,t+offset]);ee=jnp.asarray(record_np['encoded_input'][b,t+offset]);product=np.asarray(adapter.jac_h(params,hh,ee))@product;gains.append(float(np.linalg.svd(product,compute_uv=False)[0]))
                derivative.append(dict(episode=int(b),jh_direction_relative_error=float(np.linalg.norm(fd-jh@direction)/max(np.linalg.norm(jh@direction),1e-8)),je_direction_relative_error=float(np.linalg.norm(fde-je@direction)/max(np.linalg.norm(je@direction),1e-8)),jh_gain=float(np.linalg.svd(jh,compute_uv=False)[0]),je_gain=float(np.linalg.svd(je,compute_uv=False)[0]),spectral_radius=float(np.abs(np.linalg.eigvals(jh)).max()),partner_gain=float(np.linalg.norm(jh@q)/max(np.linalg.norm(q),1e-8)),finite_horizon_gain=gains))
            if max(r['jh_direction_relative_error'] for r in derivative)>.02 or max(r['je_direction_relative_error'] for r in derivative)>.02:raise ValueError('Derivative finite difference check failed')
            write_json(directory/'jacobian_summary.json',derivative)
            intervention_rows=[];zero_verified=True
            for b in np.flatnonzero(test):
                rep=b%reps;layout_seq=jnp.asarray(sequences[rep]);key=jax.random.PRNGKey(config['analysis_seed']+500+rep)
                # Physically valid new-round state, preserving recipient capability and identical random assignment per history pair.
                state=env._build_state_for(layout_seq[histories],**({'key':key} if version=='v2' else {}),capability=jnp.asarray(caps[b]),round_idx=jnp.int32(histories),episode_layout_seq=layout_seq)
                h=final_h[b];obs=env.get_obs(state)['agent_0'];obs_batch=jax.tree_util.tree_map(lambda a:a[None],obs)
                baseline=branch(params,state,jnp.asarray(h),key);baseline={k:np.asarray(v) for k,v in baseline.items()}
                nh,p0,v0,e0,logits0=adapter.full(params,jnp.asarray(h)[None],obs_batch,jnp.zeros(1,bool))
                opposite=np.flatnonzero(discovery&(orient!=orient[b]))[rep%(discovery.sum()//2)]
                same_mag=np.flatnonzero(discovery&(orient==orient[b])&np.any(caps!=caps[b],axis=1))[0]
                sham=np.flatnonzero(discovery&np.all(caps==caps[b],axis=1))[0]
                for donor_kind,donor_idx in [('opposite',opposite),('same_orientation_different_delay',same_mag),('same_profile',sham)]:
                    for kind in (['whole','partner','attenuate','random','nonpartner'] if donor_kind!='same_profile' else ['sham']):
                        # Fixed random direction across the magnitude sweep.
                        perturb_seed=int(rng.integers(0,2**31))
                        for magnitude in config['intervention_magnitudes']:
                            hp=perturb(h,final_h[donor_idx],q,reference,kind,magnitude,np.random.default_rng(perturb_seed))
                            arm=branch(params,state,jnp.asarray(hp),key);arm={k:np.asarray(v) for k,v in arm.items()}
                            nhp,pp,vp,_,_=adapter.full(params,jnp.asarray(hp)[None],obs_batch,jnp.zeros(1,bool))
                            # Input-matched open-loop response and paired closed-loop behavior are separate fields.
                            zero=all(np.array_equal(arm[k],baseline[k]) for k in baseline)
                            if magnitude==0:zero_verified &= zero
                            probs=arm['learned_probabilities'];pbase=baseline['learned_probabilities']
                            requested_partner_red=np.arange(len(probs))%3==2
                            support=float(np.min(np.linalg.norm(final_h[discovery]-hp,axis=1)))
                            intervention_rows.append(dict(episode=int(b),recipient_profile=caps[b].tolist(),donor_profile=caps[donor_idx].tolist(),history_steps=int(lengths[b]),donor_history_steps=int(lengths[donor_idx]),donor_kind=donor_kind,kind=kind,magnitude=magnitude,perturb_norm=float(np.linalg.norm(hp-h)),support_distance=support,
                                open_loop_probability_l1=float(np.abs(np.asarray(pp[0])-np.asarray(p0[0])).sum()),open_loop_value_delta=float(vp[0]-v0[0]),allocation_red_probability_delta=float(probs[requested_partner_red].sum()-pbase[requested_partner_red].sum()),reward_delta=float(arm['reward']-baseline['reward']),success_delta=int(arm['success'])-int(baseline['success']),steps_delta=int(arm['steps']-baseline['steps']),switch_delta=int(arm['switches']-baseline['switches']),baseline_reward=float(baseline['reward']),zero_identity=bool(zero) if magnitude==0 else None))
            if not zero_verified:raise ValueError('Zero intervention changed original trajectory')
            write_json(directory/'intervention_outcomes.json',intervention_rows)
            write_json(directory/'capture_summary.json',dict(episodes=B,valid_steps=int(valid.sum()),profiles=profiles.tolist(),history_rounds=histories,layout_indices=layouts,max_replay_error=max(replay_errors),sampled_action_replay=all(sample_agreement),zero_intervention_identity=zero_verified,projector_validation_accuracy=validation_accuracy,projector_rank=q.shape[1],jacobian_samples=len(derivative),intervention_branches=len(intervention_rows),split='repetitions 0,1 discovery; 2 validation; 3 held-out test',
                limits=['4-round histories are targeted early-experience diagnostics, not all 20 rounds.', 'Whole-carry swaps can change nuisance memories; projector is label-predictive and needs control comparisons for causal specificity.', 'History durations vary by capability and performance; matched round exposure is not matched step exposure.', 'v2 first learned allocation follows forced t=0, so its closed-loop t=1 observations can diverge across arms.', 'Current input-matched test measures t=0 network/value response; forced v2 allocation probabilities require a later replay test.']))
            print('Captured',directory.name,'valid steps',valid.sum(),'replay error',max(replay_errors),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--version',choices=['v1','v2'],required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--force',action='store_true');args=parser.parse_args()
    run(json.loads(args.config.read_text()),args.version,args.output,args.force)

if __name__=='__main__':main()
