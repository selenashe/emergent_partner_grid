"""Validate historical pilots, full checkpoint restoration, and exact resume.

Resume uses the recorder's callback operands and original 915-update outer scan.
An exception after the first saved resumed update stops that diagnostic prefix;
no arithmetic, scan outputs, PRNG streams, or optimizer schedule are changed.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
BUNDLE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('GRID_DIAGNOSTICS_REPO_ROOT', str(BUNDLE)))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUNDLE))
from diagnostics.learning_diagnostics import load_target, target_name, load_trainer, sha, write_json
from diagnostics.record_learning_diagnostics import instrument


def extract_step(source):
    anchor = '''        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}
'''
    if source.count(anchor) != 1:
        raise ValueError('Unexpected original outer scan')
    return source.replace(anchor, '        _audit_publish_step(_update_step)\n        return runner_state\n')


def run(args):
    import jax
    import numpy as np
    import wandb
    from flax import serialization
    row, config = load_target(args.target)
    out = args.run_dir / 'pilots' / target_name(row)
    status = json.loads((out / 'status.json').read_text())
    fidelity = json.loads((out / 'fidelity.json').read_text())
    assert status['state'] == 'complete' and status['completed_updates'] == list(range(1,33))
    assert fidelity['passed'] and fidelity['checked_updates'] == 32 and fidelity['scalar_checks'] >= 180
    metrics = json.loads((out/'metrics.json').read_text())
    artifacts = json.loads((out / 'artifacts.json').read_text())
    for entry in artifacts:
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Artifact hash mismatch: ' + entry['path'])
        if entry['kind'] == 'training_transitions':
            with np.load(entry['path']) as t:
                assert t['action'].shape == (256,256)
                assert int(t['round_events/completed'].sum()) == int(metrics[str(entry['update'])]['rounds_completed'])
                assert np.array_equal(t['round_events/layout_idx'],t['info/layout_idx'][t['round_events/tick'],t['round_events/worker']])
                assert np.isfinite(t['advantages']).all() and np.isfinite(t['targets']).all()
    result_path=out/'resume_validation.json'
    if result_path.exists() and not (out/'resume_validation_separate_step.json').exists():
        shutil.copyfile(result_path,out/'resume_validation_separate_step.json')
    captured = {}
    def collect(*values):
        captured['update_result'] = values
        raise RuntimeError('RESUMED_UPDATE_CAPTURED')
    trainer, _, _ = load_trainer(row['version'], lambda s:extract_step(instrument(s)),
        {'_audit_publish_step': lambda step: captured.update(step=step), '_audit_callback':collect})
    wandb.init(mode='disabled', project=config['PROJECT'], config=config)
    initial = jax.eval_shape(trainer.make_train(config),
        jax.random.split(jax.random.PRNGKey(config['SEED']), config['NUM_SEEDS'])[0])
    update_fn = captured['step']
    print('[resume validator] abstract state template built; no initialization kernels executed', flush=True)
    before = serialization.from_bytes(initial, (out/'checkpoints/update0015.msgpack').read_bytes())
    expected = serialization.from_bytes(initial, (out/'checkpoints/update0016.msgpack').read_bytes())
    reference = serialization.msgpack_restore((out/'checkpoints/update0015.msgpack').read_bytes())
    restored = serialization.to_state_dict(before)
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(restored)
    reference_leaves, reference_tree = jax.tree_util.tree_flatten(reference)
    assert actual_tree == reference_tree
    assert all(np.array_equal(a,b) for a,b in zip(actual_leaves, reference_leaves))
    if args.check_template_only:
        write_json(out/'template_validation.json',dict(passed=True,checkpoint_leaves=len(actual_leaves),
            template_initialization='jax.eval_shape',validator_sha256=sha(__file__),full_resume_not_tested=True))
        return
    batched = jax.tree_util.tree_map(lambda x: np.expand_dims(x,0), before)
    def resume_scan(state):
        state, metrics_ = jax.lax.scan(update_fn,state,None,length=config['NUM_UPDATES'])
        return {'runner_state':state,'metrics':metrics_}
    try:
        resumed = jax.vmap(jax.jit(resume_scan))(batched)
        jax.tree_util.tree_map(lambda x:x.block_until_ready(),resumed)
        jax.effects_barrier()
    except Exception as exc:
        if 'update_result' not in captured or 'RESUMED_UPDATE_CAPTURED' not in str(exc):
            raise
    _, resumed, traj, advantages, targets_, bootstrap, losses, resumed_metrics = captured['update_result']
    actual_with_paths, td = jax.tree_util.tree_flatten_with_path(resumed)
    expected_leaves, ed = jax.tree_util.tree_flatten(expected)
    assert td == ed
    comparisons=[]
    for i,((path,a),b) in enumerate(zip(actual_with_paths,expected_leaves)):
        a,b=np.asarray(a),np.asarray(b)
        comparisons.append(dict(leaf=i,path=jax.tree_util.keystr(path),shape=list(a.shape),dtype=str(a.dtype),
            exact=bool(np.array_equal(a,b)),max_absolute_difference=float(np.max(np.abs(
                a.astype(np.float64)-b.astype(np.float64)))) if a.size else 0))
    batch_checks=[]
    with np.load(out/'training/update0016.npz') as recorded:
        def compare(key, value):
            if isinstance(value,dict):
                for k,v in value.items(): compare(key+'/'+k if key else k,v)
            else:
                a=np.asarray(value); b=recorded[key]
                batch_checks.append(dict(field=key,exact=bool(np.array_equal(a,b)),
                    max_absolute_difference=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0))
        compare('',traj._asdict())
        for key,value in [('advantages',advantages),('targets',targets_),('bootstrap_value',bootstrap),('epoch_minibatch_losses',losses[0])]:
            compare(key,value)
    metric_checks=[dict(field=k,exact=bool(float(v)==metrics['16'][k])) for k,v in resumed_metrics.items()]
    result=dict(passed=all(c['exact'] for c in comparisons+batch_checks+metric_checks),historical_metrics_match=True,
        resume_from_update=15,compared_with_update=16,comparisons=comparisons,batch_checks=batch_checks,metric_checks=metric_checks,
        validator_sha256=sha(__file__),template_initialization='jax.eval_shape: no numerical initialization',
        compilation_context='original 915-update scan and identical recorder callback operands',
        full_state_fields='parameters, Adam state/step, environment, observations, done masks, update, GRU, RNG, queue/audits, unfinished episode returns/lengths')
    write_json(result_path,result)
    if not result['passed']:
        raise ValueError('Exact resume check failed; inspect named state/batch/metric comparisons in resume_validation.json')
    print(json.dumps(dict(target=target_name(row),passed=True,artifacts_verified=len(artifacts),
        exact_state_leaves=len(comparisons),exact_batch_fields=len(batch_checks),exact_metrics=len(metric_checks))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=int,required=True,choices=range(12))
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--check-template-only',action='store_true')
    run(p.parse_args())

if __name__=='__main__': main()
