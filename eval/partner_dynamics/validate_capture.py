"""Independent replay against the original full network and legal-action capture."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import h5py
from .collect import load_frozen
from .data import resolve_path,write_json

def run(config,version,directory):
    os.environ.setdefault('JAX_PLATFORMS','cpu')
    import jax
    import jax.numpy as jnp
    trainer,Env,load_params,manifest,_=load_frozen(config,version)
    checkpoints=resolve_path(manifest['versions'][version]['checkpoint_root'])
    for condition in config['conditions']:
        cfg=json.loads((checkpoints/f'{condition}_seed{config["seeds"][0]}_config.json').read_text());network,_,_=trainer.build_network(cfg,15)
        @jax.jit
        def original(params,h,obs,key):
            nh,pi,value=network.apply(params,h,(jax.tree_util.tree_map(lambda a:a[None],obs),jnp.zeros((1,len(h)),bool)))
            action=trainer.select_action(pi,key,cfg.get('ACTION_SELECTION','categorical')) if hasattr(trainer,'select_action') else pi.sample(seed=key)
            return nh,pi.probs[0],value[0],action[0]
        for seed in config['seeds']:
            out=directory/f'{condition}_seed{seed}';params=load_params(checkpoints/f'{condition}_seed{seed}.safetensors');error=0.;actions=True;state_error=False
            with h5py.File(out/'capture.h5','r+') as f:
                for t in (0,5,10):
                    h=jnp.asarray(f['carry_before'][:,t]);obs={k:jnp.asarray(v[:,t]) for k,v in f['observation'].items()};key=jnp.asarray(f['action_key'][0,t]);nh,p,v,a=original(params,h,obs,key)
                    error=max(error,float(np.max(np.abs(np.asarray(nh)-f['hidden_state'][:,t]))),float(np.max(np.abs(np.asarray(p)-f['probabilities'][:,t]))),float(np.max(np.abs(np.asarray(v)-f['value'][:,t]))));actions &= np.array_equal(np.asarray(a),f['flat_action'][:,t])
                logits=np.asarray(f['logits']);flat=np.asarray(f['flat_action']);legal=np.isfinite(logits)
                for name,value in [('legal_action_mask',legal),('decoded_move',flat//3),('decoded_request',flat%3)]:
                    if name not in f:f.create_dataset(name,data=value,compression='gzip',compression_opts=1)
                valid=np.asarray(f['valid']);slot,step=np.where(valid)
                if not legal[slot,step,flat[slot,step]].all():raise ValueError('An illegal action was recorded')
            if error>1e-4 or not actions:raise ValueError('Original full-network/action replay failed')
            write_json(out/'original_network_replay.json',dict(max_hidden_probability_value_error=error,exact_original_sampled_actions=bool(actions),timesteps_checked=[0,5,10],legal_action_mask_validated=True,original_frozen_network_used=True))
            print('Original replay',version,out.name,error,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--version',choices=['v1','v2'],required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();run(json.loads(a.config.read_text()),a.version,a.output)
