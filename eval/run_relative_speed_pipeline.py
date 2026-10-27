"""Run and independently verify all specialist-only representation outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys


def run(root, cpus):
    root=Path(root).resolve()
    manifest=json.loads((root/'analysis_run.json').read_text())
    repo=Path(manifest['repo_root']);python=manifest['analysis_python']
    if len(cpus)!=3 or not set(cpus)<=os.sched_getaffinity(0):
        raise ValueError('Provide three allowed CPU assignments')
    for relative in ['eval/run_relative_speed_pipeline.py','eval/round_state_prediction_comparison.py',
        'eval/export_round_state_comparison_pngs.py','diagnostics/verify_relative_speed_results.py',
        'diagnostics/verify_round_state_comparison.py']:
        target=root/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if (repo/relative).resolve()!=target.resolve():shutil.copyfile(repo/relative,target)
        manifest['source_sha256'][relative]=hashlib.sha256(target.read_bytes()).hexdigest()
    (root/'analysis_run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    status=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),stages=[])
    def save():
        (root/'pipeline_status.json').write_text(json.dumps(status,indent=2)+'\n')
    env=dict(os.environ,GRID_PROJECT_ROOT=str(repo),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',MPLCONFIGDIR=str(root/'mpl_cache'),NUMBA_CACHE_DIR=str(root/'numba_cache'))
    def step(name,*arguments):
        status['status']=name;save();print('[stage]',name,flush=True)
        subprocess.run([python,'-u',*map(str,arguments)],env=env,cwd=repo,check=True)
        status['stages'].append(dict(name=name,completed_utc=datetime.now(timezone.utc).isoformat()));save()
    comparison=root/'prediction_fractions_state_comparison'
    try:
        step('fit_specialist_probes_and_umaps',root/'source/eval/run_relative_speed_representation.py','local',root/'analysis_run.json','--cpus',*cpus)
        step('audit_specialist_probes',root/'source/diagnostics/verify_relative_speed_results.py',root)
        if not comparison.exists():
            step('prepare_state_comparison',root/'source/eval/round_state_prediction_comparison.py','prepare',comparison,'--source-results',root)
        step('fit_discrete_state_probes',comparison/'source/round_state_prediction_comparison.py','run',comparison)
        step('export_comparison_pngs',root/'source/eval/export_round_state_comparison_pngs.py',comparison)
        step('audit_state_comparison',root/'source/diagnostics/verify_round_state_comparison.py',comparison)
        figures=[]
        for condition in manifest['conditions']:
            path=comparison/condition/'comparison_all_versions.png';data=path.read_bytes()
            assert struct.unpack('>II',data[16:24])==(1800,3960)
            assert len(list((comparison/condition).glob('comparison_*_seed*.png')))==15
            figures.append(dict(condition=condition,path=str(path),sha256=hashlib.sha256(data).hexdigest()))
        status.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat(),policies=45,
            episodes_per_policy=920,train_counts=[368,368],test_counts=[92,92],figures=figures)
        save()
        with (root/'README.md').open('a') as stream:
            stream.write('\nIndependent audits passed for all 45 base policies and all 45 state comparisons. '
                'All three combined comparison PNGs and 45 individual-seed PNGs were verified. '
                'Pipeline execution and figure hashes are in pipeline_status.json.\n')
        print('[completed] Specialist-only representation pipeline and both audits.',flush=True)
    except BaseException as error:
        status.update(status='failed',error=repr(error));save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--cpus',type=int,nargs=3,required=True)
    args=parser.parse_args();run(args.root,args.cpus)
