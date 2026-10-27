"""Independently audit binary specialist-only probes, splits, and UMAPs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def verify(root, partial=False):
    root=Path(root).resolve()
    manifest=json.loads((root/'analysis_run.json').read_text())
    assert partial or manifest['status']=='completed'
    assert manifest['schema_version']==2 and manifest['episodes_per_policy']==920
    assert len(manifest['classes'])==2 and len(manifest['policies'])==45
    identities=set();checks=[]
    for record in manifest['policies']:
        out=Path(record['analysis_dir'])
        if partial and not (out/'completion.json').exists():continue
        identity=(record['version'],record['condition'],record['seed'])
        assert identity not in identities
        identities.add(identity)
        for field in ['checkpoint','config']:
            assert hashlib.sha256(Path(record[field]).read_bytes()).hexdigest()==record[field+'_sha256']
        episodes=pd.read_csv(out/'episode_manifest.csv')
        assert len(episodes)==920 and set(episodes.capability_slice)=={'train','test'}
        assert set(episodes.source_file)=={str(Path(p).resolve()) for p in record['original_rollouts'].values()}
        assert not episodes.duplicated(['source_file','source_episode']).any()
        pairs=episodes[['d_R','d_B']].to_numpy()
        assert np.all(pairs[:,0]!=pairs[:,1])
        truth=(pairs[:,1]<pairs[:,0]).astype(np.int64)
        np.testing.assert_array_equal(episodes['class'],truth)
        counts=episodes.groupby(['d_R','d_B','probe_partition']).size().unstack()
        assert len(counts)==46 and counts['train'].eq(16).all() and counts['test'].eq(4).all()
        permutation=np.random.default_rng(0).permutation(20)
        np.testing.assert_array_equal(episodes.rollout_rep.isin(permutation[:16]),episodes.probe_partition=='train')
        with np.load(out/'probe_models.npz') as probes:
            assert probes['weight'].shape==(30,128,2) and probes['bias'].shape==(30,2)
            probs=probes['probabilities'];preds=probes['predictions'];targets=probes['truth']
            W=probes['weight'];bias=probes['bias']
            assert probs.shape==(30,920,2) and np.isfinite(probs).all()
            assert np.isfinite(W).all() and np.isfinite(bias).all() and np.isfinite(probes['final_train_loss']).all()
            np.testing.assert_allclose(probs.sum(axis=2),1,atol=2e-6)
            np.testing.assert_array_equal(preds,probs.argmax(axis=2))
            np.testing.assert_array_equal(targets[:29],np.broadcast_to(truth,(29,920)))
            np.testing.assert_array_equal(targets[-1],np.random.default_rng(20000).permutation(truth))
            train,test=probes['train'],probes['test']
            assert (train.sum(),test.sum())==(736,184) and np.all(train!=test)
            np.testing.assert_array_equal(train,episodes.probe_partition=='train')
            np.testing.assert_array_equal(np.bincount(truth[train],minlength=2),[368,368])
            np.testing.assert_array_equal(np.bincount(truth[test],minlength=2),[92,92])
            weights=np.stack([736/(2*np.bincount(y[train],minlength=2)) for y in targets])
            np.testing.assert_allclose(probes['class_weights'],weights)
            np.testing.assert_array_equal(weights[:29],np.ones((29,2)))
            table=pd.read_csv(out/'probe_per_seed.csv')
            specs=[('timestep',t,'primary') for t in probes['time_cutoffs']]
            specs+=[('round',t,'primary') for t in range(1,21)]+[('timestep',400,'shuffled_labels')]
            assert len(table)==90
            for i,(axis,cutoff,kind) in enumerate(specs):
                saved=table[(table.axis==axis)&(table.cutoff==cutoff)&(table.kind==kind)&(table.population=='all')]
                assert len(saved)==1
                y,p=targets[i,test],preds[i,test]
                recalls=[np.mean(p[y==k]==k) for k in range(2)]
                np.testing.assert_allclose(saved.accuracy.iloc[0],np.mean(y==p),atol=1e-12)
                np.testing.assert_allclose(saved.balanced_accuracy.iloc[0],np.mean(recalls),atol=1e-12)
            for path,group in episodes.groupby('source_file'):
                population=next(k for k,v in record['original_rollouts'].items() if Path(v).resolve()==Path(path))
                stat=Path(path).stat();saved=record['original_stats'][population]
                assert (stat.st_size,stat.st_mtime_ns)==(saved['size'],saved['mtime_ns'])
                with h5py.File(path) as raw:
                    for row in group.iloc[[0,len(group)//2,len(group)-1]].itertuples():
                        done=raw['dones'][row.source_episode]
                        length=int(np.flatnonzero(done)[0])+1
                        hidden=raw['hidden_state'][row.source_episode,:length]
                        ends=np.flatnonzero(raw['round_done'][row.source_episode,:length])
                        assert len(ends)==20
                        prefix=np.cumsum(hidden,axis=0,dtype=np.float64)
                        times=np.minimum(np.array([1,50,100,150,200,250,300,350,400]),length)
                        cutoffs=np.r_[times,ends+1,times[-1]]
                        features=(prefix[cutoffs-1]/cutoffs[:,None]).astype(np.float32)
                        logits=np.einsum('ph,phk->pk',features,W)+bias
                        exp=np.exp(logits-logits.max(axis=1,keepdims=True));prob=exp/exp.sum(axis=1,keepdims=True)
                        np.testing.assert_allclose(probs[:,row.Index],prob,rtol=1e-4,atol=2e-6)
        with np.load(out/'random_models.npz') as control:
            assert control['weight'].shape==(5,128,2)
            assert control['probabilities'].shape==(5,920,2)
            np.testing.assert_array_equal(control['class_weights'],np.ones((5,2)))
        embedding=pd.read_csv(out/'umap.csv')
        assert len(embedding)==920 and np.isfinite(embedding[['umap_x','umap_y']]).all().all()
        np.testing.assert_array_equal(embedding['class'],truth)
        np.testing.assert_array_equal(embedding[['d_R','d_B']],pairs)
        checks.append(dict(version=identity[0],condition=identity[1],seed=identity[2],status='passed'))
    expected={(v,c,s) for v in manifest['versions'] for c in manifest['conditions'] for s in range(1,6)}
    if not partial:
        assert identities==expected
        assert len(pd.read_csv(root/'probe_per_seed.csv'))==4050
        assert len(pd.read_csv(root/'umap_all_seeds.csv'))==41400
    for relative,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((root/'source'/relative).read_bytes()).hexdigest()==digest
    result=dict(status='passed_partial' if partial else 'passed',n_policies=len(checks),verified_utc=datetime.now(timezone.utc).isoformat(),
        checks=['original_46_specialist_profiles_only','no_tied_delay_pairs','16_4_split_per_profile','368_fit_92_test_per_class',
            'two_output_models_and_probabilities','heldout_metrics_recomputed','raw_prefix_features_reconstruct_probabilities',
            'training_only_loss_weights','920_point_umaps','source_and_input_hashes'],policies=checks)
    (root/('partial_validation.json' if partial else 'independent_validation.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(f'Independent specialist-only audit passed for {len(checks)} policies.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--partial',action='store_true')
    args=parser.parse_args();verify(args.root,args.partial)
