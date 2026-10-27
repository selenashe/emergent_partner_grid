"""Small CPU functional check; not a historical training reproduction."""
import argparse
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys
from pathlib import Path
BUNDLE=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('GRID_DIAGNOSTICS_REPO_ROOT',str(BUNDLE)))
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(BUNDLE))
from diagnostics.learning_diagnostics import load_target, load_trainer
from diagnostics.record_learning_diagnostics import instrument


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=int,required=True)
    p.add_argument('--eval-fixture-dir',type=Path)
    args=p.parse_args()
    import numpy as np
    import jax
    import wandb
    from flax import serialization
    row,config=load_target(args.target)
    config.update(NUM_ENVS=4,NUM_STEPS=4,NUM_MINIBATCHES=2,TOTAL_TIMESTEPS=915*16)
    captured=[]
    def callback(before,after,traj,advantages,targets,bootstrap,losses,metrics):
        assert traj.action.shape==(4,4)
        assert np.isfinite(advantages).all() and np.isfinite(targets).all()
        assert int(after[0].step)==8 and int(after[4])==1
        assert len(serialization.to_bytes(before))>0 and len(serialization.to_bytes(after))>0
        if args.eval_fixture_dir:
            from diagnostics.learning_diagnostics import target_name,write_json
            fixture=args.eval_fixture_dir/'runs'/target_name(row)
            fixture.mkdir(parents=True,exist_ok=True)
            params=fixture/'params.msgpack'
            params.write_bytes(serialization.to_bytes(after[0].params))
            write_json(fixture/'status.json',dict(state='complete',functional_fixture=True,actual_update=1))
            write_json(fixture/'artifacts.json',[dict(kind='params',update=1,path=str(params))])
            import shutil
            shutil.copyfile(ROOT/'train/seed_diagnostics/v1_matched_v2_v3/learning_divergence/20261015_full_round_recording/evaluation_panel.json',args.eval_fixture_dir/'evaluation_panel.json')
        captured.append(True)
        raise RuntimeError('FUNCTIONAL_SMOKE_COMPLETE')
    trainer,_,_=load_trainer(row['version'],instrument,{'_audit_callback':callback})
    wandb.init(mode='disabled',project=config['PROJECT'],config=config)
    try:
        result=jax.vmap(jax.jit(trainer.make_train(config)))(jax.random.split(jax.random.PRNGKey(config['SEED']),1))
        jax.tree_util.tree_map(lambda x:x.block_until_ready(),result)
    except Exception as exc:
        if not captured or 'FUNCTIONAL_SMOKE_COMPLETE' not in str(exc): raise
    assert captured
    print('FUNCTIONAL SMOKE PASSED:',row['version'],row['condition'])

if __name__=='__main__': main()
