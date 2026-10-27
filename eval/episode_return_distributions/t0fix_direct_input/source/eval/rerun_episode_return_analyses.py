"""Reproduce four saved-evaluation analyses for the five-seed direct-input cohort.

Run with the existing representation environment. The reference mode uses the
training environment and compares with each executed frozen Flax network. No
policy training or new environment rollouts are performed. Separate diagnostic
capability probes are fitted to saved states for the selected-policy audit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eval.episode_return_analysis import (BASE, BATCH, BUILD, ROOT, TRAINING_MANIFEST,
    extract_policy, key, records, require, sha, write)
from eval.episode_return_math import analyze_policy, reference_check

SOURCE_FILES=('eval/episode_return_analysis.py','eval/episode_return_math.py',
              'eval/episode_return_reports.py','eval/rerun_episode_return_analyses.py',
              'eval/episode_return_decoding.py','eval/relative_speed_representation.py',
              'eval/representation_analysis.py')
EVAL_PYTHON='/nlp/scr/jshe/miniconda3/envs/emergent_partner_model/bin/python'


def worker(record,root):
    extract_policy(record,root)
    analyze_policy(record,root)
    return key(record)


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    policies=records()
    previous=root/'run.json'
    if previous.exists():
        old=json.loads(previous.read_text())['policies']
        require([(key(p),p['checkpoint_sha256'],p['config_sha256']) for p in old]==[(key(p),p['checkpoint_sha256'],p['config_sha256']) for p in policies],'Run inputs changed')
    write(previous,dict(cohort=BATCH,step_penalty=.01,policies=policies,training_manifest=str(TRAINING_MANIFEST),training_manifest_sha256=sha(TRAINING_MANIFEST)))
    return policies


def references(root):
    import os
    policies=json.loads((Path(root)/'run.json').read_text())['policies']
    choices=[]
    for version in ('v1_matched','v2','v3'):
        for condition in ('rnn_diverse_influence','mlp_diverse_influence'):
            choices.append(next(i for i,p in enumerate(policies) if p['version']==version and p['condition']==condition))
    choices.append(next(i for i,p in enumerate(policies) if p['version']=='v1_matched' and p['condition']=='rnn_diverse_noinfluence'))
    env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',XLA_FLAGS='')
    for i in choices:
        subprocess.run([EVAL_PYTHON,str(Path(__file__).resolve()),'reference','--root',str(root),'--index',str(i)],cwd=ROOT,env=env,check=True)


def freeze(root):
    directory=Path(root)/'t0fix_direct_input/source'
    hashes={}
    for rel in SOURCE_FILES:
        target=directory/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,target);hashes[rel]=sha(target)
    write(directory/'source_inventory.json',hashes)


def validate(root):
    import numpy as np
    import pandas as pd
    from train.cohort_integrity import verify
    root=Path(root)
    manifest=verify(TRAINING_MANIFEST)
    run=json.loads((root/'run.json').read_text()) if (root/'run.json').exists() else json.loads((root/'t0fix_direct_input/analysis_run.json').read_text())
    require(run['cohort']==BATCH and run['step_penalty']==.01,'Wrong analysis cohort')
    states=0;probes=0;mlp_starts=0;max_hidden=0
    for p in run['policies']:
        for name in ('checkpoint','config'):require(sha(p[name])==p[name+'_sha256'],'Policy changed: '+key(p))
        c=root/'t0fix_direct_input/cache'/key(p)
        done=json.loads((c/'completion.json').read_text());math=json.loads((c/'math/completion.json').read_text())
        require(done['episodes']==920 and done['rounds']==18400 and done['tail_episodes']==92,'Incomplete policy coverage')
        require(done['producer_sha256']==sha(ROOT/'eval/episode_return_analysis.py'),'Changed extractor source')
        require(math['producer_sha256']==sha(ROOT/'eval/episode_return_math.py'),'Changed mathematical producer source')
        for source in done['sources']:
            f=Path(source['path']);require(f.stat().st_size==source['size'] and f.stat().st_mtime_ns==source['mtime_ns'],'Original evaluation changed')
        frame=pd.read_csv(c/'episode_metrics.csv.gz')
        require(len(frame)==920 and frame.lowest_decile.sum()==92,'Wrong tail selection')
        require(frame[['population','episode_index']].duplicated().sum()==0,'Duplicate episodes')
        require(np.allclose(frame.oracle_best_return-frame.episode_return,frame.assignment_opportunity_gap+frame.execution_gap,atol=3e-5),'Oracle decomposition mismatch')
        states+=math['saved_hidden_states_analyzed'];probes+=math['observation_probes'];mlp_starts+=math['mlp_start_observations']
        max_hidden=max(max_hidden,math['maximum_start_hidden_reconstruction_error'])
    expected={'lowest_decile_analysis':(12,11040,1104,22080),'lowest_decile_analysis_all_seeds':(60,55200,5520,110400)}
    for folder,(policies,episodes,tail,rounds) in expected.items():
        d=json.loads((root/folder/'validation.json').read_text())
        require((d['policies'],d['episodes'],d['tail_episodes'],d['tail_rounds'])==(policies,episodes,tail,rounds),'Wrong report coverage')
        require(d['reward_gap_decomposition_max_error']<3e-5,'Reward gap mismatch')
        audit=json.loads((root/folder/'math_audit/validation.json').read_text())
        require(audit['all_start_reconstructions_within_0_01'],'Observation reconstruction failed')
        require((root/folder/'report.html').exists() and (root/folder/'math_audit/report.html').exists(),'Missing report')
    reference_files=list((root/'t0fix_direct_input/reference_checks').glob('*.json'))
    require(len(reference_files)==7,'Need seven independent frozen-network reference checks')
    require(all(json.loads(p.read_text())['status']=='passed' for p in reference_files),'Failed frozen-network comparison')
    general=json.loads((root/'single_partner_generalization/validation.json').read_text())
    require(general['policies']==30 and general['episodes']==27600,'Wrong generalization coverage')
    decoder=root/'lowest_decile_analysis/math_audit/decision_decoding'
    decoder_audit=json.loads((decoder/'validation.json').read_text())
    require(decoder_audit['fitted_probes']==36 and decoder_audit['policies']==6,'Missing decision decoders')
    require(decoder_audit['producer_sha256']==sha(ROOT/'eval/episode_return_decoding.py'),'Changed decoding source')
    require(len(list(decoder.glob('*.npz')))==6,'Incorrect decoder model count')
    for p in run['policies']:
        if not (p['selected'] and p['condition'] in ('rnn_diverse_influence','rnn_single_influence')):continue
        with np.load(decoder/(key(p)+'.npz'),allow_pickle=False) as f:
            train,test=f['train'],f['test']
            require((train.sum(),test.sum())==(736,184) and not (train&test).any() and (train|test).all(),'Invalid decoder split')
            require(f['features'].shape==(6,920,128) and np.isfinite(f['features']).all(),'Invalid decoder features')
            require(np.allclose(f['probabilities'].sum(-1),1,atol=1e-6),'Invalid decoder probabilities')
            require(np.array_equal(f['predictions'],f['probabilities'].argmax(-1)),'Decoder predictions mismatch')
            require(np.array_equal(f['truth'][5],np.random.default_rng(20000).permutation(f['truth'][0])),'Invalid shuffled control')
            for pair in np.unique(f['pairs'],axis=0):
                mask=np.all(f['pairs']==pair,axis=1)
                require((int((mask&train).sum()),int((mask&test).sum()))==(16,4),'Decoder profile leakage')
    require(probes==1656000 and mlp_starts==552000,'Wrong observation-probe/MLP coverage')
    inventory=json.loads((root/'t0fix_direct_input/source/source_inventory.json').read_text())
    for rel,digest in inventory.items():require(sha(root/'t0fix_direct_input/source'/rel)==digest,'Frozen producer changed')
    result=dict(status='passed',cohort=BATCH,step_penalty=.01,policies=60,seeds=[1,2,3,4,5],
        episodes=55200,rounds=1104000,all_seed_tail_episodes=5520,all_seed_tail_rounds=110400,
        selected_policies=12,selected_episodes=11040,selected_tail_episodes=1104,selected_tail_rounds=22080,
        rnn_policies=45,mlp_policies=15,generalization_policies=30,
        saved_hidden_states_analyzed=states,observation_probes=probes,mlp_start_observations=mlp_starts,
        maximum_hidden_reconstruction_error=max_hidden,independent_frozen_network_checks=7,
        checkpoint_and_config_hashes_unchanged=120,new_environment_rollouts=False,policy_retraining=False,
        probe_fitting=True,diagnostic_probes_fitted=36,decision_decoding_policies=6)
    write(root/'t0fix_direct_input/completion_audit.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def publish(root):
    root=Path(root);validate(root)
    folders=('t0fix_direct_input','lowest_decile_analysis','lowest_decile_analysis_all_seeds','single_partner_generalization')
    require(root.is_relative_to(BASE) and root!=BASE,'Publish only the isolated build tree')
    for name in folders[1:]:
        target=BASE/name
        if target.exists():
            require((target/'step_penalty_0.03').is_dir(),'Refusing to replace already-current or unknown results: '+str(target))
    require(not (BASE/'t0fix_direct_input').exists(),'Current distribution results already exist')
    # All previous outputs were inspected and the four removals were explicitly
    # authorized. Only these result directories are removed; policy inputs stay.
    removed=0
    for name in ('step_penalty_0.03',)+folders[1:]:
        target=BASE/name
        if target.exists():
            require(target.is_dir() and not target.is_symlink(),'Unexpected deletion target')
            removed+=sum(1 for p in target.rglob('*') if p.is_file() or p.is_symlink())
            shutil.rmtree(target)
    shutil.copyfile(root/'run.json',root/'t0fix_direct_input/analysis_run.json')
    for name in folders:(root/name).rename(BASE/name)
    (root/'run.json').unlink();root.rmdir()
    print('Removed',removed,'old-analysis files; published four current result directories',flush=True)
    validate(BASE)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['run','prepare','worker','reference','references','report','validate','publish'])
    p.add_argument('--root',type=Path,default=BUILD);p.add_argument('--workers',type=int,default=3);p.add_argument('--index',type=int)
    args=p.parse_args();root=args.root.resolve()
    if args.mode=='prepare':prepare(root)
    elif args.mode in ('worker','reference'):
        require(args.index is not None,'Provide --index')
        record=json.loads((root/'run.json').read_text())['policies'][args.index]
        if args.mode=='worker':worker(record,root)
        else:reference_check(record,root/'t0fix_direct_input/reference_checks'/(key(record)+'.json'))
    elif args.mode=='references':references(root)
    elif args.mode=='report':
        from eval.episode_return_reports import build_reports
        from eval.episode_return_decoding import run_decoding
        build_reports(root);run_decoding(root);freeze(root)
    elif args.mode=='validate':validate(root)
    elif args.mode=='publish':publish(root)
    else:
        policies=prepare(root)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs=[pool.submit(worker,record,root) for record in policies]
            for i,f in enumerate(as_completed(jobs),1):print('[complete]',i,'/60',f.result(),flush=True)
        references(root)
        from eval.episode_return_reports import build_reports
        from eval.episode_return_decoding import run_decoding
        build_reports(root);run_decoding(root);freeze(root);validate(root)


if __name__=='__main__':main()
