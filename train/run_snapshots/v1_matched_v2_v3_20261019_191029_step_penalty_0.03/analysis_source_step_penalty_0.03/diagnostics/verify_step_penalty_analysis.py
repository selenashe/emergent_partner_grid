"""Exercise five-seed aggregation, probe preparation, and 0.03 reward accounting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


def verify(path):
    manifest=json.loads(Path(path).read_text());root=Path(manifest['analysis_root'])
    os.environ['GRID_PROJECT_ROOT']=str(root)
    sys.path.insert(0,str(root));sys.path.insert(0,str(root/'eval'))
    from eval import relative_speed_representation as relative
    from eval import round_state_prediction_comparison as states
    from eval import run_relative_speed_representation as runner
    from eval.plot_round_return_progression import extract_round_returns
    from eval.optimal_training_return import round_return
    checks=[]
    rows=[dict(version='v3',condition='rnn_diverse_influence',axis='round',cutoff=20,
        kind='primary',population='all',training_seed=i,
        balanced_accuracy=.5+.05*i,accuracy=.5+.05*i,recall_0=.5+.05*i,recall_1=.5+.05*i) for i in range(1,6)]
    out=relative.summary(pd.DataFrame(rows))
    np.testing.assert_allclose(out.balanced_accuracy_mean,[.65])
    assert out.balanced_accuracy_ci_low.iloc[0] <= .65 <= out.balanced_accuracy_ci_high.iloc[0]
    checks.append('five-seed cumulative-probe bootstrap and coverage')
    fractions=pd.DataFrame([dict(version='v3',condition='rnn_diverse_influence',feature_type='round_end_state',
        round=20,ground_truth_class=0,predicted_class=0,training_seed=i,prediction_fraction=.5+.05*i) for i in range(1,6)])
    result=states.summarize(fractions)
    assert result.n_seeds.iloc[0]==5
    np.testing.assert_allclose(result['mean'],[.65])
    checks.append('five-seed discrete-state bootstrap and coverage')
    for function,table in [(relative.summary,pd.DataFrame(rows)),(states.summarize,fractions)]:
        try:function(table[table.training_seed!=5])
        except ValueError:pass
        else:raise AssertionError('Missing policy seed was accepted')
    checks.append('both aggregators reject incomplete policy coverage')
    success=np.tile([False,True],20)[None,:]
    reward=np.where(success,1.,-.03)
    done=np.zeros_like(success);done[0,-1]=True
    cumulative,incremental,steps,flags=extract_round_returns(reward,done,success,np.repeat(np.arange(20),2)[None,:],np.tile([1,2],20)[None,:],success)
    np.testing.assert_allclose(cumulative[0,-1],19.4)
    np.testing.assert_allclose(incremental,.97)
    np.testing.assert_array_equal(steps,2)
    assert flags.all()
    np.testing.assert_allclose([round_return(2),round_return(100),round_return(101)],[.97,-1.97,-3])
    checks.append('0.03 per-tick and oracle reward accounting including timeout')
    with tempfile.TemporaryDirectory(prefix='step_penalty_analysis_') as temporary:
        tmp=Path(temporary); source=tmp/'scientific_source';(source/'analysis').mkdir(parents=True)
        evaluator=source/'analysis/evaluate_partner_modelling.py';evaluator.write_text('pass\n')
        config=tmp/'config.json';config.write_text('{}')
        checkpoint=tmp/'checkpoint.safetensors';checkpoint.write_text('dummy test fixture')
        layouts=tmp/'layouts';layouts.mkdir()
        rollouts={p:tmp/(p+'.h5') for p in ['train','test']}
        for p in rollouts.values():p.touch()
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        policies=[]
        for version in runner.VERSIONS:
            for condition in runner.CONDITIONS:
                for seed in range(1,6):
                    policies.append(dict(version=version,condition=condition,seed=seed,
                        source_root=str(source),evaluator_sha256=sha(evaluator),config=str(config),config_sha256=sha(config),
                        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),layouts_dir=str(layouts),allocation_protocol='test_fixture',
                        original_rollouts={p:str(q) for p,q in rollouts.items()},
                        original_stats={p:dict(size=q.stat().st_size,mtime_ns=q.stat().st_mtime_ns) for p,q in rollouts.items()}))
        inputs=tmp/'inputs.json';inputs.write_text(json.dumps(dict(policies=policies)))
        prepared=runner.prepare(tmp/'representations',inputs)
        saved=json.loads(prepared.read_text())
        assert saved['n_policies']==45 and len(saved['policies'])==45 and saved['seeds']==list(range(1,6))
        assert saved['episodes_per_policy']==920 and saved['train_counts']==[368,368] and saved['test_counts']==[92,92]
        checks.append('exact 45-policy manifest with unchanged per-policy episodes and splits')
        saved['status']='completed';prepared.write_text(json.dumps(saved))
        state_root=states.prepare(prepared.parent,tmp/'state_comparison')
        sm=json.loads((state_root/'analysis_run.json').read_text())
        assert sm['n_policy_entries']==45 and sm['n_rounds']==20
        checks.append('discrete-state preparation accepts same 45-policy cohort')
    result=dict(status='passed',checks=checks,policies=60,representation_policies=45,
                step_penalty=.03,seeds=list(range(1,6)),source_hashes_verified_by_training_manifest=True)
    output=Path(manifest['infrastructure_root']).parent/('analysis_validation_step_penalty_0.03.json')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('manifest',type=Path)
    verify(parser.parse_args().manifest)
