"""Recover the three MLP runs using their own completed validation evidence.

The shared gate also ran RNN checks in a different compilation context; their
failure must not invalidate the MLP's exact state/batch/metric and eval checks.
Never overwrite started runs or rerun the nine completed RNN targets.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.learning_diagnostics import sha,write_json,targets,target_name,load_target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args()
    out=args.run_dir.resolve()
    submission=json.loads((out/'submission.json').read_text())
    bundle=Path(submission['code_bundle'])
    selected=(0,1,8)
    pilot=out/'pilots'/target_name(targets()[0])
    validation=json.loads((pilot/'resume_validation.json').read_text())
    assert validation['passed']
    assert validation['validator_sha256']==sha(bundle/'diagnostics/validate_learning_diagnostics.py')
    for key in ('comparisons','batch_checks','metric_checks'):
        assert validation[key] and all(c['exact'] for c in validation[key])
    evaluation=json.loads((pilot/'evaluation_status.json').read_text())
    assert evaluation['state']=='complete' and evaluation['completed_conditions']==6
    assert evaluation['panel_sha256']==sha(out/'evaluation_panel.json')
    fidelity=json.loads((pilot/'fidelity.json').read_text())
    assert fidelity['passed'] and fidelity['checked_updates']==32
    for index in selected:
        row,_=load_target(index)
        assert row['condition']=='mlp_diverse_influence'
        if (out/'runs'/target_name(row)/'status.json').exists():
            raise ValueError('A target has already started; refusing to overwrite '+target_name(row))
    old_array=submission.get('mlp_array_job_id')
    if old_array:
        state=subprocess.run(['squeue','-h','-j',str(old_array),'-o','%T'],capture_output=True,text=True,check=False).stdout.strip()
        if state:
            raise ValueError('An MLP array remains active: '+state)
    for filename in ('record_learning_diagnostics.py','evaluate_learning_diagnostics.py','run_learning_diagnostics.sh'):
        assert sha(bundle/'diagnostics'/filename)==sha(out/'source/diagnostics'/filename)
    commands=[]
    def submit(extra):
        cmd=['sbatch','--parsable',*extra]
        result=subprocess.run(cmd,check=True,capture_output=True,text=True)
        job=result.stdout.strip().split(';')[0]
        commands.append(dict(job_id=job,command=cmd))
        return job
    job=submit(['--account=nlp','--partition=sphinx,sc-loprio','--constraint=80G','--gres=gpu:a100:1',
        '--mem=64G','--cpus-per-task=8','--exclude=sphinx9','--time=1-00:00:00',
        '--chdir='+str(ROOT),'--array=0-1,8%2','--job-name=seed_diag_mlp_recover',
        '--output='+str(out/'slurm/mlp_recovery_%A_%a.out'),'--error='+str(out/'slurm/mlp_recovery_%A_%a.err'),
        str(bundle/'diagnostics/run_learning_diagnostics.sh'),str(out),'full',str(bundle)])
    write_json(out/'submission_history/mlp_cancelled_shared_gate.json',submission)
    submission.update(mlp_array_job_id=job,state='nine_rnn_complete_three_mlp_resubmitted',
        mlp_recovery=dict(validated_target=target_name(targets()[0]),validated_state_leaves=len(validation['comparisons']),
            validated_batch_fields=len(validation['batch_checks']),validated_metrics=len(validation['metric_checks']),
            validation_path=str(pilot/'resume_validation.json'),evaluation_path=str(pilot/'evaluation_status.json'),
            reason='Use the completed MLP-specific validation; remove unrelated RNN gate dependency.',
            previous_cancelled_array=old_array,targets=list(selected),concurrency=2,commands=commands,
            recovery_script_sha256=sha(__file__)))
    write_json(out/'submission.json',submission)
    report_cmd=shlex.join(['/nlp/scr/jshe/miniconda3/envs/emergent_partner_model/bin/python',
        str(bundle/'diagnostics/report_learning_diagnostics.py'),'--run-dir',str(out)])
    report=submit(['--account=nlp','--partition=sphinx,sc-loprio','--mem=2G','--cpus-per-task=1','--time=00:05:00',
        '--chdir='+str(ROOT),'--dependency=afterany:'+job,'--job-name=seed_diag_report',
        '--output='+str(out/'slurm/report_%j.out'),'--error='+str(out/'slurm/report_%j.err'),
        '--wrap','GRID_DIAGNOSTICS_REPO_ROOT='+shlex.quote(str(ROOT))+' '+report_cmd])
    submission['report_job_id']=report
    submission['mlp_recovery']['commands']=commands
    write_json(out/'submission.json',submission)
    print(json.dumps(dict(mlp_array=job,report_job=report,targets=list(selected),concurrency=2),indent=2))

if __name__=='__main__':main()
