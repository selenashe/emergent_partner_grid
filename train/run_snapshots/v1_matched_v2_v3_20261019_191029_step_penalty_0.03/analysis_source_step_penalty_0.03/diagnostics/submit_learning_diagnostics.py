"""Freeze diagnostic sources and queue the validated twelve-policy expansion."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.learning_diagnostics import targets,load_target,checkpoint_updates,sha,write_json
from repo_paths import artifact_root,prepare_artifact_path,link_artifact


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--pilot-job-id',required=True)
    p.add_argument('--retry-failed-pipeline',action='store_true')
    p.add_argument('--retry-reason',default='Retry the failed validation gate with corrected validation code.')
    a=p.parse_args()
    out=a.run_dir.resolve()
    submission=json.loads((out/'submission.json').read_text())
    retry = a.retry_failed_pipeline
    if submission.get('full_run_submitted') and not retry:
        raise ValueError('Full array already submitted')
    if retry:
        # Retrying validation reuses intact pilots. Do not reuse this path to
        # overwrite a full replay that has already started.
        if any((out/'runs').glob('*/status.json')):
            raise ValueError('A full replay already started; select individual recovery actions instead')
        for key in ('gate_job_id','full_array_job_id','report_job_id'):
            if submission.get(key):
                active=subprocess.run(['squeue','-h','-j',str(submission[key]),'-o','%T'],
                    capture_output=True,text=True,check=False).stdout.strip()
                if active:
                    raise ValueError(f'Previous {key} is still active: {active}')
        from diagnostics.learning_diagnostics import PILOTS,target_name
        for i in PILOTS:
            row,_=load_target(i)
            pilot=out/'pilots'/target_name(row)
            status=json.loads((pilot/'status.json').read_text())
            fidelity=json.loads((pilot/'fidelity.json').read_text())
            assert status['state']=='complete' and fidelity['passed'] and fidelity['checked_updates']==32
        history=out/'submission_history'
        history.mkdir(exist_ok=True)
        index=len(list(history.glob('attempt*.json')))+1
        write_json(history/f'attempt{index:02d}.json',submission)
        previous_bundle=Path(submission['code_bundle'])
        bundle=out/f'source_retry{index}'
    else:
        bundle=out/'source'
    for i,row in enumerate(targets()): load_target(i)
    # Verify write access and physical routing before using GPU allocations.
    logical=out/'storage_probe.bin'
    physical=prepare_artifact_path(logical)
    assert str(physical).startswith('/juice6/scr6/nlp/interp-models/emergent_partner_grid/')
    physical.write_bytes(b'diagnostic scratch routing verified\n')
    link_artifact(logical)
    assert logical.resolve()==physical
    logical.unlink();physical.unlink()
    bundle.mkdir(exist_ok=False)
    (bundle/'diagnostics').mkdir()
    (bundle/'diagnostics/__init__.py').write_text('')
    names=('learning_diagnostics.py','record_learning_diagnostics.py','evaluate_learning_diagnostics.py',
           'validate_learning_diagnostics.py','run_learning_diagnostics.sh','diagnostic_pipeline_gate.sh',
           'report_learning_diagnostics.py','submit_learning_diagnostics.py','smoke_learning_diagnostics.py',
           'learning_trace_schema.md','seed_debugging_plan.md')
    for name in names:
        # Preserve the tested recorder/evaluator byte for byte on a gate retry.
        base=previous_bundle if retry and name not in (
            'validate_learning_diagnostics.py','diagnostic_pipeline_gate.sh','submit_learning_diagnostics.py') else ROOT
        shutil.copyfile(base/'diagnostics'/name,bundle/'diagnostics'/name)
    # Freeze selected rows as well as code; subsequent cohort-table edits
    # must not change the identity of a queued array task.
    shutil.copyfile(ROOT/'train/seed_diagnostics/v1_matched_v2_v3/seed_selection/representative_seeds.csv',bundle/'diagnostics/representative_seeds.csv')
    common=bundle/'diagnostics/learning_diagnostics.py'
    text=common.read_text()
    text=text.replace("COHORT = ROOT / 'train/seed_diagnostics/v1_matched_v2_v3/seed_selection/representative_seeds.csv'","COHORT = Path(__file__).with_name('representative_seeds.csv')")
    common.write_text(text)
    routing=(ROOT/'repo_paths.py').read_text().replace(
        'ROOT = Path(__file__).resolve().parent',
        "ROOT = Path(os.environ.get('GRID_DIAGNOSTICS_REPO_ROOT', str(Path(__file__).resolve().parent)))")
    (bundle/'repo_paths.py').write_text(routing)
    write_json(out/'source_hashes.json',{str(f.relative_to(bundle)):sha(f) for f in bundle.rglob('*') if f.is_file()})
    write_json(out/'checkpoint_schedule.json',{str(i):dict(target=row,updates=checkpoint_updates(row)) for i,row in enumerate(targets())})
    env=dict(os.environ,GRID_DIAGNOSTICS_REPO_ROOT=str(ROOT),GRID_ARTIFACT_ROOT=str(artifact_root()))
    def submit(args):
        result=subprocess.run(['sbatch','--parsable',*args],check=True,capture_output=True,text=True,env=env)
        return result.stdout.strip().split(';')[0]
    base=['--account=nlp','--partition=sphinx,sc-loprio','--constraint=80G','--gres=gpu:a100:1',
          '--mem=64G','--cpus-per-task=8','--exclude=sphinx9','--chdir='+str(ROOT)]
    dependencies=[] if retry else ['--dependency=afterok:'+a.pilot_job_id]
    gate_args=[*base,'--time=01:00:00',*dependencies,'--kill-on-invalid-dep=yes',
        '--job-name=seed_diag_validate','--output='+str(out/'slurm/gate_%j.out'),'--error='+str(out/'slurm/gate_%j.err'),
        str(bundle/'diagnostics/diagnostic_pipeline_gate.sh'),str(out),str(bundle)]
    gate=submit(gate_args)
    submission.update(gate_job_id=gate,gate_command=['sbatch','--parsable',*gate_args],
        code_bundle=str(bundle),full_run_submitted=False)
    if retry:
        submission.update(retry_number=index,reused_pilots=True,
            retry_reason=a.retry_reason)
    write_json(out/'submission.json',submission)
    full_args=[*base,'--time=1-00:00:00','--array=0-11%2','--dependency=afterok:'+gate,'--kill-on-invalid-dep=yes',
        '--job-name=seed_diag_full','--output='+str(out/'slurm/full_%A_%a.out'),'--error='+str(out/'slurm/full_%A_%a.err'),
        str(bundle/'diagnostics/run_learning_diagnostics.sh'),str(out),'full',str(bundle)]
    full=submit(full_args)
    submission.update(full_array_job_id=full,full_command=['sbatch','--parsable',*full_args],
        full_run_submitted=True,state='queued_with_fidelity_gate',code_bundle=str(bundle),
        scratch_root=str(artifact_root()),recording_coverage='every training transition and every training/evaluation round',
        max_concurrent_gpus=2)
    write_json(out/'submission.json',submission)
    report_args=['--account=nlp','--partition=sphinx,sc-loprio','--mem=2G','--cpus-per-task=1','--time=00:05:00',
        '--chdir='+str(ROOT),'--dependency=afterany:'+full,'--job-name=seed_diag_report',
        '--output='+str(out/'slurm/report_%j.out'),'--error='+str(out/'slurm/report_%j.err'),
        '--wrap','GRID_DIAGNOSTICS_REPO_ROOT='+str(ROOT)+' /nlp/scr/jshe/miniconda3/envs/emergent_partner_model/bin/python '+str(bundle/'diagnostics/report_learning_diagnostics.py')+' --run-dir '+str(out)]
    # All paths above are fixed repository paths without spaces/metacharacters.
    report=submit(report_args)
    submission.update(report_job_id=report,report_command=['sbatch','--parsable',*report_args])
    write_json(out/'submission.json',submission)
    print(json.dumps(dict(pilots=a.pilot_job_id,gate=gate,full_array=full,report=report,run_dir=str(out)),indent=2))

if __name__=='__main__': main()
