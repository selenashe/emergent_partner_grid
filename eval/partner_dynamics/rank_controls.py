"""Rank- and norm-matched complement-subspace interventions with original keys."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import h5py
from .data import resolve_path,write_json
from .collect import load_frozen,diagnostic_layouts


def run(config,version,directory):
    os.environ.setdefault('JAX_PLATFORMS','cpu')
    import jax
    import jax.numpy as jnp
    import distrax
    from .adapters import PolicyAdapter
    trainer,Env,load_params,manifest,_=load_frozen(config,version)
    checkpoints=resolve_path(manifest['versions'][version]['checkpoint_root']);histories=config['capture_history_rounds'];reps=config['capture_repetitions']
    for condition in config['conditions']:
        cfg=json.loads((checkpoints/f'{condition}_seed{config["seeds"][0]}_config.json').read_text());kwargs=dict(cfg['ENV_KWARGS']);kwargs['layouts_dir']=str(resolve_path(kwargs['layouts_dir']));kwargs['augment_symmetries']=False;kwargs['influence']=bool(cfg['INFLUENCE']);kwargs['partner_capability_pairs']=config['capture_profiles'];env=Env(**kwargs)
        network,_,_=trainer.build_network(cfg,env.n_ego_actions);adapter=PolicyAdapter(trainer,network,cfg);layouts=diagnostic_layouts(env,config['diagnostic_layout_count']);sequences=np.asarray([[layouts[(rep+r)%len(layouts)] for r in range(20)] for rep in range(reps)],np.int32)
        @jax.jit
        def branch(params,state,h,key):
            initial_round=state.round_idx;obs=env.get_obs(state)
            def step(carry,_):
                obs,state,h,finished,key,total,success,steps=carry
                nh,prob,value,e,logits=adapter.full(params,h[None],jax.tree_util.tree_map(lambda a:a[None],obs['agent_0']),jnp.zeros(1,bool))
                key,ka,ks=jax.random.split(key,3);dist=distrax.Categorical(probs=prob[None]);action=trainer.select_action(dist,ka,cfg.get('ACTION_SELECTION','categorical'))[0,0] if hasattr(trainer,'select_action') else dist.sample(seed=ka)[0,0]
                obs2,state2,reward,done,info=env.step_env(ks,state,{'agent_0':action});valid=~finished
                return (obs2,state2,nh[0],finished|(state2.round_idx!=initial_round)|done['__all__'],key,total+jnp.where(valid,reward['agent_0'],0.),success|(info['success']&valid),steps+valid.astype(jnp.int32)),prob[0]
            final,p=jax.lax.scan(step,(obs,state,h,jnp.bool_(False),key,jnp.float32(0),jnp.bool_(False),jnp.int32(0)),None,length=env.max_steps)
            return final[5],final[6],final[7],p[1] if version=='v2' else p[0]
        for seed in config['seeds']:
            out=directory/f'{condition}_seed{seed}';params=load_params(checkpoints/f'{condition}_seed{seed}.safetensors')
            with h5py.File(out/'capture.h5') as f:h=np.asarray(f['history_final_carry']);caps=np.asarray(f['capability_profile']);rep=np.asarray(f['repetition'])
            q=np.load(out/'partner_projector.npz')['basis'];rng=np.random.default_rng(config['analysis_seed']+777);matrix=rng.normal(size=q.shape);matrix-=q@(q.T@matrix);control=np.linalg.qr(matrix)[0][:,:q.shape[1]]
            np.savez_compressed(out/'rank_matched_control.npz',basis=control,rank=q.shape[1]);assert np.linalg.norm(q.T@control)<1e-8
            discovery=rep<reps-2;orient=caps[:,0]<caps[:,1];test=np.flatnonzero(rep==reps-1);rows=[];original=json.loads((out/'intervention_outcomes.json').read_text())
            for b in test:
                r=int(rep[b]);sequence=jnp.asarray(sequences[r]);key=jax.random.PRNGKey(config['analysis_seed']+500+r);state=env._build_state_for(sequence[histories],**({'key':key} if version=='v2' else {}),capability=jnp.asarray(caps[b]),round_idx=jnp.int32(histories),episode_layout_seq=sequence)
                baseline=branch(params,state,jnp.asarray(h[b]),key);baseline=[np.asarray(x) for x in baseline]
                if abs(float(baseline[0])-next(x['baseline_reward'] for x in original if x['episode']==b))>1e-5:raise ValueError('Control baseline replay failed')
                opposite=np.flatnonzero(discovery&(orient!=orient[b]))[r%(discovery.sum()//2)];same=np.flatnonzero(discovery&(orient==orient[b])&np.any(caps!=caps[b],axis=1))[0]
                for donor_kind,donor in [('opposite',opposite),('same_orientation_different_delay',same)]:
                    difference=h[donor]-h[b];partner=q@(q.T@difference);delta=control@(control.T@difference);delta*=np.linalg.norm(partner)/max(np.linalg.norm(delta),1e-12)
                    for magnitude in config['intervention_magnitudes']:
                        arm=branch(params,state,jnp.asarray(h[b]+magnitude*delta),key);arm=[np.asarray(x) for x in arm]
                        if magnitude==0 and not all(np.array_equal(a,v) for a,v in zip(arm,baseline)):raise ValueError('Rank control zero-identity failure')
                        red=np.arange(len(arm[3]))%3==2
                        rows.append(dict(episode=int(b),recipient_profile=caps[b].tolist(),donor_profile=caps[donor].tolist(),donor_kind=donor_kind,kind='rank_matched_nonpartner',magnitude=magnitude,rank=int(q.shape[1]),perturb_norm=float(magnitude*np.linalg.norm(delta)),reward_delta=float(arm[0]-baseline[0]),success_delta=int(arm[1])-int(baseline[1]),steps_delta=int(arm[2]-baseline[2]),allocation_red_probability_delta=float(arm[3][red].sum()-baseline[3][red].sum()),first_learned_probability_l1=float(np.abs(arm[3]-baseline[3]).sum())))
            write_json(out/'rank_matched_control_outcomes.json',rows);print('Rank control',version,out.name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--version',choices=['v1','v2'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(json.loads(a.config.read_text()),a.version,a.output)
