"""Identical-input replay from history-derived and perturbed carries."""
import argparse
import json
from pathlib import Path
import os
import numpy as np
import h5py
from .data import ROOT,resolve_path,write_json
from .collect import load_frozen,diagnostic_layouts


def run(config,version,directory):
    # Audit guide:
    # Replay recipient and perturbed memories against the same baseline observations at
    # initialization and the next tick. Use the later learned-action probabilities for
    # v2 because its initialization choice is forced; use the initialization choice for
    # v1. Reconstruct the same donor/magnitude/random-control perturbations as capture,
    # verify zero-change identity, and distinguish this fixed-input action response from
    # paired free-running reward outcomes.
    #
    os.environ.setdefault('JAX_PLATFORMS','cpu')
    import jax
    import jax.numpy as jnp
    import distrax
    from .adapters import PolicyAdapter
    from .interventions import perturb
    trainer,Env,load_params,manifest,source=load_frozen(config,version)
    checkpoints=resolve_path(manifest['versions'][version]['checkpoint_root'])
    for condition in config['conditions']:
        cfg=json.loads((checkpoints/f'{condition}_seed{config["seeds"][0]}_config.json').read_text())
        kwargs=dict(cfg['ENV_KWARGS']);kwargs['layouts_dir']=str(resolve_path(kwargs['layouts_dir']));kwargs['augment_symmetries']=False;kwargs['influence']=bool(cfg['INFLUENCE']);kwargs['partner_capability_pairs']=config['capture_profiles']
        env=Env(**kwargs);network,_,_=trainer.build_network(cfg,env.n_ego_actions);adapter=PolicyAdapter(trainer,network,cfg);step=jax.jit(env.step_env)
        layouts=diagnostic_layouts(env,config['diagnostic_layout_count']);reps=config['capture_repetitions'];histories=config['capture_history_rounds']
        sequences=np.asarray([[layouts[(rep+r)%len(layouts)] for r in range(20)] for rep in range(reps)],np.int32)
        for seed in config['seeds']:
            out=directory/f'{condition}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
            if not (out/'capture.h5').exists():continue
            params=load_params(checkpoints/f'{condition}_seed{seed}.safetensors')
            with h5py.File(out/'capture.h5') as f:
                hfinal=np.asarray(f['history_final_carry']);caps=np.asarray(f['capability_profile']);rep=np.asarray(f['repetition'])
            source_outcomes=json.loads((out/'intervention_outcomes.json').read_text());test=np.flatnonzero(rep==reps-1);discovery=rep<reps-2;orient=caps[:,0]<caps[:,1]
            projector=dict(np.load(out/'partner_projector.npz'));q=projector['basis'];reference=projector['reference']
            rng=np.random.default_rng(config['analysis_seed'])
            # Reproduce the recorded perturbation directions exactly: derivative checks consumed one direction per test history.
            for _ in test:rng.normal(size=hfinal.shape[1])
            outcomes=[];comparison_error=0.
            for b in test:
                r=int(rep[b]);layout_seq=jnp.asarray(sequences[r]);key=jax.random.PRNGKey(config['analysis_seed']+500+r)
                state=env._build_state_for(layout_seq[histories],**({'key':key} if version=='v2' else {}),capability=jnp.asarray(caps[b]),round_idx=jnp.int32(histories),episode_layout_seq=layout_seq)
                observation=env.get_obs(state)['agent_0'];ob=jax.tree_util.tree_map(lambda a:a[None],observation)
                nh,p,v,e,logits=adapter.full(params,jnp.asarray(hfinal[b])[None],ob,jnp.zeros(1,bool))
                original=distrax.Categorical(probs=p[None]).sample(seed=jax.random.split(key,3)[1])[0,0]
                obs2,_,_,_,_=step(jax.random.split(key,3)[2],state,{'agent_0':original});ob2=jax.tree_util.tree_map(lambda a:a[None],obs2['agent_0'])
                nh2,p2,v2,e2,_=adapter.full(params,nh,ob2,jnp.zeros(1,bool))
                opposite=np.flatnonzero(discovery&(orient!=orient[b]))[r%(discovery.sum()//2)];same=np.flatnonzero(discovery&(orient==orient[b])&np.any(caps!=caps[b],axis=1))[0];sham=np.flatnonzero(discovery&np.all(caps==caps[b],axis=1))[0]
                for donor_kind,donor in [('opposite',opposite),('same_orientation_different_delay',same),('same_profile',sham)]:
                    for kind in (['whole','partner','attenuate','random','nonpartner'] if donor_kind!='same_profile' else ['sham']):
                        perturb_seed=int(rng.integers(0,2**31))
                        for magnitude in config['intervention_magnitudes']:
                            hp=perturb(hfinal[b],hfinal[donor],q,reference,kind,magnitude,np.random.default_rng(perturb_seed))
                            hh,pp,vv,_,_=adapter.full(params,jnp.asarray(hp)[None],ob,jnp.zeros(1,bool))
                            hh2,pp2,vv2,_,_=adapter.full(params,hh,ob2,jnp.zeros(1,bool))
                            reference_prob=np.asarray(p2[0] if version=='v2' else p[0]);arm_prob=np.asarray(pp2[0] if version=='v2' else pp[0]);allocation_ids=np.arange(len(arm_prob))%3==2
                            delta=float(arm_prob[allocation_ids].sum()-reference_prob[allocation_ids].sum())
                            original_row=next(row for row in source_outcomes if row['episode']==b and row['donor_kind']==donor_kind and row['kind']==kind and row['magnitude']==magnitude)
                            comparison_error=max(comparison_error,abs(delta-original_row['allocation_red_probability_delta']))
                            if abs(np.linalg.norm(hp-hfinal[b])-original_row['perturb_norm'])>1e-5:raise ValueError('Perturbation replay norm mismatch')
                            outcomes.append(dict(episode=int(b),donor_kind=donor_kind,kind=kind,magnitude=magnitude,perturb_seed=perturb_seed,first_learned_probability_l1=float(np.abs(arm_prob-reference_prob).sum()),allocation_red_probability_delta=delta,value_delta=float(vv2[0]-v2[0]) if version=='v2' else float(vv[0]-v[0]),
                                first_hidden_delta=float(np.linalg.norm(np.asarray(hh-nh))),second_hidden_delta=float(np.linalg.norm(np.asarray(hh2-nh2))),
                                replay_mode='Identical baseline observations at t=0 and t=1; free-running reward is analyzed separately'))
            if comparison_error>1e-4:raise ValueError(f'First learned open/closed-loop probability mismatch: {comparison_error}')
            write_json(out/'open_loop_outcomes.json',outcomes);write_json(out/'open_loop_validation.json',dict(first_learned_probability_matches_closed_loop_max_error=comparison_error,zero_identity=all(row['first_learned_probability_l1']==0 for row in outcomes if row['magnitude']==0),
                interpretation='v2 t=1 observations are physically identical because the supplied t=0 action is forced; v1 first legal allocation is t=0.',n_branches=len(outcomes)))
            print('Input-matched replay',version,out.name,'max error',comparison_error,flush=True)


def main():
    # Audit guide:
    # Load the recorded targeted-capture configuration and select one frozen protocol
    # for replay. Write replay outcomes beside capture artifacts. This diagnostic uses
    # existing checkpoint weights and collected histories; it does not train another
    # policy.
    #
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--version',choices=['v1','v2'],required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(json.loads(args.config.read_text()),args.version,args.output)
if __name__=='__main__':main()
