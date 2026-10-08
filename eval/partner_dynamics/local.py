"""Exact input-matched GRU/readout sensitivities from different natural histories."""
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
    from .adapters import PolicyAdapter
    trainer,Env,load_params,manifest,_=load_frozen(config,version)
    checkpoints=resolve_path(manifest['versions'][version]['checkpoint_root']);reps=config['capture_repetitions'];histories=config['capture_history_rounds']
    for condition in config['conditions']:
        cfg=json.loads((checkpoints/f'{condition}_seed{config["seeds"][0]}_config.json').read_text());kwargs=dict(cfg['ENV_KWARGS']);kwargs['layouts_dir']=str(resolve_path(kwargs['layouts_dir']));kwargs['augment_symmetries']=False;kwargs['influence']=bool(cfg['INFLUENCE']);kwargs['partner_capability_pairs']=config['capture_profiles'];env=Env(**kwargs)
        network,_,_=trainer.build_network(cfg,env.n_ego_actions);adapter=PolicyAdapter(trainer,network,cfg);layouts=diagnostic_layouts(env,config['diagnostic_layout_count']);sequences=np.asarray([[layouts[(rep+r)%len(layouts)] for r in range(20)] for rep in range(reps)],np.int32)
        prob_jac=jax.jit(jax.jacfwd(lambda params,h,ob:adapter._readout(params,h[None],ob)[0][0],argnums=1))
        value_jac=jax.jit(jax.grad(lambda params,h,ob:adapter._readout(params,h[None],ob)[1][0],argnums=1))
        for seed in config['seeds']:
            out=directory/f'{condition}_seed{seed}';params=load_params(checkpoints/f'{condition}_seed{seed}.safetensors')
            with h5py.File(out/'capture.h5') as f:h=np.asarray(f['history_final_carry']);caps=np.asarray(f['capability_profile']);rep=np.asarray(f['repetition'])
            q=np.load(out/'partner_projector.npz')['basis'];discovery=rep<reps-2;orientation=caps[:,0]<caps[:,1];rows=[]
            for b in np.flatnonzero(rep==reps-1):
                r=int(rep[b]);sequence=jnp.asarray(sequences[r]);key=jax.random.PRNGKey(config['analysis_seed']+500+r);state=env._build_state_for(sequence[histories],**({'key':key} if version=='v2' else {}),capability=jnp.asarray(caps[b]),round_idx=jnp.int32(histories),episode_layout_seq=sequence)
                ob=jax.tree_util.tree_map(lambda a:a[None],env.get_obs(state)['agent_0']);nh,p,v,e,_=adapter.full(params,jnp.asarray(h[b])[None],ob,jnp.zeros(1,bool));encoded=e[0]
                jh=np.asarray(adapter.jac_h(params,jnp.asarray(h[b]),encoded));je=np.asarray(adapter.jac_e(params,jnp.asarray(h[b]),encoded))
                jp=np.asarray(prob_jac(params,nh[0],ob));jv=np.asarray(value_jac(params,nh[0],ob))
                opposite=np.flatnonzero(discovery&(orientation!=orientation[b]))[r%(discovery.sum()//2)];same=np.flatnonzero(discovery&(orientation==orientation[b])&np.any(caps!=caps[b],axis=1))[0];sham=np.flatnonzero(discovery&np.all(caps==caps[b],axis=1))[0]
                for kind,donor in [('opposite',opposite),('same_orientation_different_delay',same),('same_profile',sham)]:
                    donor_je=np.asarray(adapter.jac_e(params,jnp.asarray(h[donor]),encoded));donor_jh=np.asarray(adapter.jac_h(params,jnp.asarray(h[donor]),encoded))
                    rows.append(dict(episode=int(b),donor_kind=kind,input_jacobian_relative_difference=float(np.linalg.norm(donor_je-je)/max(np.linalg.norm(je),1e-8)),hidden_jacobian_relative_difference=float(np.linalg.norm(donor_jh-jh)/max(np.linalg.norm(jh),1e-8)),partner_policy_sensitivity=float(np.linalg.norm(jp@jh@q)),partner_value_sensitivity=float(np.linalg.norm(jv@jh@q)),input_policy_sensitivity=float(np.linalg.norm(jp@je)),input_value_sensitivity=float(np.linalg.norm(jv@je)),
                        protocol_note='v2 t=0 policy allocation derivatives are zero by forced mask; learned t=1 effects are in open-loop replay' if version=='v2' else 'v1 t=0 is a learned allocation'))
            write_json(out/'input_matched_sensitivity.json',rows);print('Exact sensitivity',version,out.name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--version',choices=['v1','v2'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(json.loads(a.config.read_text()),a.version,a.output)
