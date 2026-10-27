"""Five-seed v1_matched/v2/v3 replication changing only step penalty to 0.03."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import yaml

ROOT = Path(os.environ.get('GRID_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
INFRA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INFRA))
from bash import extend_counterbalanced_training as original
from bash.step_penalty_downstream import freeze
from repo_paths import artifact_path, resolve_path, resolve_record_paths
original.ROOT = ROOT
original.path_resolver.ROOT = ROOT

SUFFIX = '_step_penalty_0.03'
VERSIONS = ('v1_matched', 'v2', 'v3')
CONDITIONS = original.CONDITIONS
SEEDS = tuple(range(1, 6))
PYTHON = original.PYTHON
ANALYSIS_PYTHON = '/juice6/u/nlp/climblab/tmp/epg_repr_all10_20261008/bin/python'
sha, write = original.sha, original.write


def now(): return datetime.now(timezone.utc).isoformat()
def load(path): return json.loads(Path(path).read_text())
def read_record(path): return resolve_record_paths(load(path))
def tag(condition, seed): return f'{condition}_seed{seed}{SUFFIX}'


def prepare():
    paths = {
        'v1_matched': ROOT / 'train/manifests/sbatch_v1_matched_20261009_182845.json',
        'v2': ROOT / 'train/manifests/sbatch_counterbalanced1096_20261002_235609.json',
        'v3': ROOT / 'train/manifests/sbatch_counterbalanced1096_20261008_035848_v3.json',
    }
    bases = {v: read_record(p) for v, p in paths.items()}
    original.verify_frozen_base(bases['v2'])
    batch = 'v1_matched_v2_v3_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + SUFFIX
    work = ROOT / 'train/run_snapshots' / batch
    work.mkdir(parents=True, exist_ok=False)
    infrastructure = work / 'infrastructure'
    for relative in ['bash/submit_step_penalty_training.py', 'bash/step_penalty_downstream.py',
                     'bash/extend_counterbalanced_training.py', 'bash/run_counterbalanced_seed_extension.sh', 'repo_paths.py']:
        dst = infrastructure / relative; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dst)
    comparison = ROOT / 'eval/protocol_comparison' / batch
    curves = ROOT / 'train/training_curves' / batch
    representation = ROOT / 'eval/representation_results' / ('relative_speed_' + batch)
    analysis = work / ('analysis_source' + SUFFIX)
    adaptation = freeze(ROOT, analysis, batch, comparison, curves, representation)
    variants, jobs, versions = {}, [], {}
    for version in VERSIONS:
        base = bases[version]
        data = base['versions'][version]
        source_specs = {version: (Path(data['frozen_source_root']), data['source_sha256'])}
        if version == 'v1_matched':
            influence_source = ROOT / 'train/run_snapshots/v1_matched_20261009_182845/v1_matched/source'
            source_specs[version + '_influence'] = (influence_source,
                {rel: sha(influence_source / rel) for rel in data['source_sha256']})
        for variant, (old, hashes) in source_specs.items():
            dst = work / (variant + SUFFIX) / 'source'
            for relative, digest in hashes.items():
                if sha(old / relative) != digest: raise ValueError('Baseline source changed: ' + str(old / relative))
                target = dst / relative; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(old / relative, target)
            variants[variant] = dict(original_source_root=str(old), frozen_source_root=str(dst), source_sha256=hashes)
        checkpoints = ROOT / 'train/train_logs' / (version + SUFFIX) / batch
        evaluations = ROOT / 'eval/eval_out' / (version + SUFFIX) / batch
        for directory in [checkpoints, evaluations]:
            directory.mkdir(parents=True, exist_ok=False)
            artifact_path(directory).mkdir(parents=True, exist_ok=False)
        versions[version] = dict(frozen_source_root=variants[version]['frozen_source_root'],
            source_sha256=variants[version]['source_sha256'], allocation_protocol=data['allocation_protocol'],
            checkpoint_root=str(checkpoints), evaluation_root=str(evaluations), evaluation_jobs=[])
        for condition in CONDITIONS:
            for seed in SEEDS:
                baseline = Path(data['checkpoint_root']) / f'{condition}_seed{seed}_config.json'
                cfg = read_record(baseline)
                if cfg['SEED'] != seed or cfg['ENV_KWARGS']['step_penalty'] != .01: raise ValueError(str(baseline))
                new = copy.deepcopy(cfg)
                new['ENV_KWARGS']['step_penalty'] = .03
                new['SAVE_PARAMS_PATH'] = str(checkpoints / (tag(condition, seed) + '.safetensors'))
                config_path = work / 'configs' / (version + SUFFIX) / (tag(condition, seed) + '.yaml')
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(yaml.safe_dump(new, sort_keys=False))
                variant = version + '_influence' if version == 'v1_matched' and condition != 'rnn_diverse_noinfluence' else version
                jobs.append(dict(version=version, condition=condition, seed=seed, tag=tag(condition, seed),
                    source_variant=variant, config_path=str(config_path), config_sha256=sha(config_path),
                    baseline_config=str(baseline), baseline_config_sha256=sha(baseline)))
    manifest = dict(batch=batch, status='prepared', prepared_at_utc=now(), experiment='V1_matched',
        scientific_change='Only ENV_KWARGS.step_penalty: 0.01 to 0.03; five learner seeds (1-5).',
        artifact_suffix=SUFFIX, step_penalty=.03, success_reward=1., action_selection='categorical',
        evaluation_seeds=list(SEEDS), new_seeds=list(SEEDS),
        nominal_timesteps_per_policy=60000000, effective_timesteps_per_policy=59965440,
        baseline_manifests={v:dict(path=str(paths[v]),sha256=sha(paths[v])) for v in VERSIONS},
        frozen_layouts_dir=bases['v1_matched']['frozen_layouts_dir'],
        layout_files_sha256=bases['v1_matched']['layout_files_sha256'],
        schedule_paths=bases['v1_matched']['schedule_paths'],
        schedule_sha256={k:sha(p) for k,p in bases['v1_matched']['schedule_paths'].items()},
        sampling_protocol=bases['v1_matched']['sampling_protocol'],
        resources_per_training_job=bases['v1_matched']['resources_per_training_job'],
        evaluation=bases['v1_matched']['evaluation'], versions=versions, source_variants=variants,
        training_jobs=jobs, preflight=dict(passed=False, cpu_checks=[], production_checks=[]),
        downstream_jobs=[], log_dir=str(ROOT / 'train/slurm_logs' / batch),
        aggregate_root=str(comparison / ('all_5_seeds' + SUFFIX)), comparison_root=str(comparison),
        curves_root=str(curves), representation_root=str(representation),
        analysis_root=str(analysis), analysis_adaptations=adaptation,
        infrastructure_root=str(infrastructure),
        infrastructure_sha256={str(p.relative_to(infrastructure)):sha(p) for p in infrastructure.rglob('*') if p.is_file()},
        noinfluence_goal_selection='uniform_random_each_round', analysis_python=ANALYSIS_PYTHON)
    Path(manifest['log_dir']).mkdir(parents=True, exist_ok=True)
    path = ROOT / 'train/manifests' / ('sbatch_' + batch + '.json')
    write(path, manifest)
    verify(path)
    print(path, flush=True)
    return path


def verify(path):
    m = load(path)
    if m['evaluation_seeds'] != list(SEEDS) or len(m['training_jobs']) != 60: raise ValueError('Expected exactly sixty policies')
    identities = {(j['version'],j['condition'],j['seed']) for j in m['training_jobs']}
    if identities != {(v,c,s) for v in VERSIONS for c in CONDITIONS for s in SEEDS}: raise ValueError('Policy coverage')
    for data in m['baseline_manifests'].values():
        if sha(data['path']) != data['sha256']: raise ValueError('Baseline manifest changed')
    for data in m['source_variants'].values():
        for rel,digest in data['source_sha256'].items():
            if sha(Path(data['frozen_source_root'])/rel) != digest or sha(Path(data['original_source_root'])/rel) != digest:
                raise ValueError('Scientific source drift: ' + rel)
    for rel,digest in m['infrastructure_sha256'].items():
        if sha(Path(m['infrastructure_root'])/rel) != digest: raise ValueError('Launcher drift: '+rel)
    for rel,record in m['analysis_adaptations'].items():
        if sha(Path(m['analysis_root'])/rel) != record['frozen_sha256']: raise ValueError('Analysis source drift: '+rel)
    for regime,p in m['schedule_paths'].items():
        if sha(p) != m['schedule_sha256'][regime]: raise ValueError('Schedule drift')
    digest=hashlib.sha256()
    for p in sorted(Path(m['frozen_layouts_dir']).glob('*.json')):
        digest.update(p.name.encode());digest.update(p.read_bytes())
    if digest.hexdigest()!=m['layout_files_sha256']:raise ValueError('Layout drift')
    for job in m['training_jobs']:
        if sha(job['config_path'])!=job['config_sha256'] or sha(job['baseline_config'])!=job['baseline_config_sha256']:
            raise ValueError('Config bytes changed: '+job['tag'])
        baseline=read_record(job['baseline_config']); wanted=copy.deepcopy(baseline)
        wanted['ENV_KWARGS']['step_penalty']=.03
        wanted['SAVE_PARAMS_PATH']=str(Path(m['versions'][job['version']]['checkpoint_root'])/(job['tag']+'.safetensors'))
        cfg=yaml.safe_load(Path(job['config_path']).read_text())
        if cfg!=wanted:raise ValueError('Changes beyond step penalty and output path: '+job['tag'])
        if cfg['TOTAL_TIMESTEPS']!=60000000 or cfg['NUM_ENVS']*cfg['NUM_STEPS']*915!=59965440:
            raise ValueError('Training budget changed')
        if not Path(cfg['SAVE_PARAMS_PATH']).stem.endswith(SUFFIX):raise ValueError('Missing suffix')
    return m


def command(m, job, smoke=False, production=False):
    local=copy.deepcopy(m)
    local['versions'][job['version']]['frozen_source_root']=m['source_variants'][job['source_variant']]['frozen_source_root']
    cmd,source,target=original.training_command(local,job,smoke=smoke,production_smoke=production)
    cmd[0]=PYTHON
    return cmd,source,target


def publish(directory, prefix):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    for physical in artifact_path(directory).glob(prefix+'*'):
        logical=directory/physical.name
        if logical==physical:continue
        if logical.exists() and not logical.is_symlink():raise FileExistsError(logical)
        tmp=logical.with_name(logical.name+'.publishing');tmp.unlink(missing_ok=True)
        tmp.symlink_to(physical);tmp.replace(logical)


def evaluation_command(m,job,smoke_target=None):
    """Read each saved input where it exists; write new rollouts on scratch."""
    source=Path(m['source_variants'][job['source_variant']]['frozen_source_root'])
    data=m['versions'][job['version']]
    checkpoint=(smoke_target/'params') if smoke_target else Path(data['checkpoint_root'])/job['tag']
    config=resolve_path(str(checkpoint)+'_config.json')
    params=resolve_path(str(checkpoint)+'.safetensors')
    for input_path in (config,params):
        if not input_path.is_file():raise FileNotFoundError(input_path)
    prefix=(smoke_target/'standalone_eval') if smoke_target else artifact_path(data['evaluation_root'])/job['tag']
    cmd=[PYTHON,'-u',str(source/'analysis/evaluate_partner_modelling.py'),
        '--config',str(config),'--params',str(params),
        '--layouts_dir',m['frozen_layouts_dir'],'--n_episodes_per_capability','1' if smoke_target else '20',
        '--seed','12345','--out_prefix',str(prefix)]
    if job['condition'].startswith('rnn'):cmd.append('--save_hidden')
    return cmd,source,prefix


def evaluate_job(m,job,smoke_target=None):
    cmd,source,prefix=evaluation_command(m,job,smoke_target)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(cmd,cwd=source,env=original.environment(source,cpu=bool(smoke_target)),check=True)
    for ext in ['_summary.json','_train.h5','_test.h5']:
        if not Path(str(prefix)+ext).is_file():raise FileNotFoundError(str(prefix)+ext)
    if not smoke_target:publish(m['versions'][job['version']]['evaluation_root'],job['tag'])


def validate(path):
    m=verify(path)
    specs=[('v1_matched','rnn_diverse_influence'),('v1_matched','rnn_diverse_noinfluence'),
           ('v2','rnn_diverse_influence'),('v3','rnn_diverse_influence'),('v3','mlp_diverse_influence')]
    cores=','.join(map(str,sorted(os.sched_getaffinity(0))[:4]))
    for version,condition in specs:
        if any(r['version']==version and r['condition']==condition for r in m['preflight']['cpu_checks']):continue
        job=next(j for j in m['training_jobs'] if (j['version'],j['condition'],j['seed'])==(version,condition,1))
        cmd,source,target=command(m,job,smoke=True)
        print('[CPU preflight]',version,condition,flush=True)
        with (target/('training'+SUFFIX+'.log')).open('w') as log:
            subprocess.run(['taskset','-c',cores]+cmd,cwd=source,env=original.environment(source,cpu=True),stdout=log,stderr=subprocess.STDOUT,check=True)
        audit=load(target/'params_sampling_audit.json')
        if audit['collected_environment_steps']!=256:raise ValueError('Smoke rollout budget')
        cfg=load(target/'params_config.json')
        if cfg['ENV_KWARGS']['step_penalty']!=.03 or cfg['ENV_KWARGS']['success_reward']!=1:raise ValueError('Smoke reward config')
        # Standalone evaluator executes in a fresh process, isolated from launcher imports.
        with (target/('evaluation'+SUFFIX+'.log')).open('w') as log:
            env=original.environment(source,cpu=True);env['GRID_PROJECT_ROOT']=str(ROOT)
            subprocess.run(['taskset','-c',cores,PYTHON,str(Path(m['infrastructure_root'])/'bash/submit_step_penalty_training.py'),
                'smoke-eval',str(path),version,condition],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        m['preflight']['cpu_checks'].append(dict(version=version,condition=condition,passed=True,output=str(target)))
        write(path,m); print('[passed]',version,condition,flush=True)
    m['preflight'].update(passed=True,completed_at_utc=now(),config_only_step_penalty_and_output_path=True,scientific_sources_byte_identical=True)
    write(path,m)


def train(path,version,condition,seed,production=False):
    m=verify(path);job=next(j for j in m['training_jobs'] if (j['version'],j['condition'],j['seed'])==(version,condition,int(seed)))
    cmd,source,target=command(m,job,production=production)
    subprocess.run(cmd,cwd=source,env=original.environment(source),check=True)
    if production:
        if load(target/'params_sampling_audit.json')['collected_environment_steps']!=65536:raise ValueError('Production tensor smoke budget')
    else:publish(m['versions'][version]['checkpoint_root'],job['tag'])


def _view(m):
    """Explicit canonical-name aliases, private to this analysis cohort."""
    result=copy.deepcopy(m)
    result['training_jobs']=[];result['evaluation_inputs']={}
    analysis=Path(m['analysis_root']);batch=m['batch']
    for version in VERSIONS:
        aliases=analysis/('inputs'+SUFFIX)
        checkpoints=aliases/'checkpoints'/(version+SUFFIX)/batch
        evaluations=aliases/'evaluations'/(version+SUFFIX)/batch
        checkpoints.mkdir(parents=True,exist_ok=True);evaluations.mkdir(parents=True,exist_ok=True)
        data=m['versions'][version]
        for job in m['training_jobs']:
            if job['version']!=version:continue
            canonical=f"{job['condition']}_seed{job['seed']}"
            for root,src,exts in [(checkpoints,Path(data['checkpoint_root']),['.safetensors','_config.json','_sampling_audit.json']),
                                   (evaluations,Path(data['evaluation_root']),['_summary.json','_train.h5','_test.h5'])]:
                for ext in exts:
                    target=src/(job['tag']+ext)
                    if not target.is_file():raise FileNotFoundError(target)
                    link=root/(canonical+ext)
                    if not link.is_symlink():link.symlink_to(target)
                    if link.resolve()!=target.resolve():raise ValueError('Alias mismatch')
            j=copy.deepcopy(job);j['tag']=canonical;result['training_jobs'].append(j)
        result['versions'][version]['checkpoint_root']=str(checkpoints)
        result['versions'][version]['evaluation_root']=str(evaluations)
        result['evaluation_inputs'][version]={str(seed):str(evaluations) for seed in SEEDS}
    path=analysis/'train/manifests'/('sbatch_'+batch+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    write(path,result)
    for name in ['sbatch_v1_matched_20261009_182845.json','sbatch_counterbalanced1096_20261002_235609.json',
                 'sbatch_counterbalanced1096_20261008_035848_v3.json']:
        write(path.parent/name,result)
    return path,result


def view(m):
    lock=Path(m['analysis_root'])/'.analysis_inputs.lock'
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        return _view(m)


def downstream(path,stage):
    m=verify(path);analysis=Path(m['analysis_root'])
    view_path,v=view(m)
    output=Path(m['comparison_root'])/('pipeline'+SUFFIX);output.mkdir(parents=True,exist_ok=True)
    status_path=output/(stage+'_status'+SUFFIX+'.json')
    status=load(status_path) if status_path.exists() else dict(status='running',completed_steps=[])
    env=dict(os.environ,GRID_PROJECT_ROOT=str(analysis),PYTHONPATH=str(analysis),PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
        JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',XLA_FLAGS='')
    def step(name,relative,*args):
        if name in status['completed_steps']:return
        status.update(status='running',current_step=name);write(status_path,status)
        print('[downstream]',name,flush=True)
        with (output/(name+SUFFIX+'.log')).open('w') as log:
            subprocess.run([ANALYSIS_PYTHON,'-u',str(analysis/relative),*map(str,args)],cwd=analysis,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        status['completed_steps'].append(name);write(status_path,status)
    try:
        if stage=='behavior':
            # Compare only the three new protocols, all with penalty 0.03.
            for population in ['train','test']:
                code=("from functools import partial; from pathlib import Path; "
                    "from eval import compare_allocation_protocols as c; "
                    "from eval.plot_episode_return_dots import plot_episode_return_dots; import json; "
                    f"m=json.load(open({str(view_path)!r})); c.INPUTS=m['evaluation_inputs']; "
                    f"c.SEEDS=tuple(m['evaluation_seeds']); c.POPULATION={population!r}; "
                    "c.OUT=Path(m['aggregate_root'])/('familiar' if c.POPULATION=='train' else 'novel'); "
                    "c.VERSION_COLORS['v1_matched']='#9470bb'; c.VERSION_LABELS['v1_matched']='v1_matched'; "
                    "c.plot_episode_return_dots=partial(plot_episode_return_dots,version_order=('v1_matched','v2','v3')); c.main()")
                name='compare_'+population
                if name not in status['completed_steps']:
                    with (output/(name+SUFFIX+'.log')).open('w') as log:
                        subprocess.run([ANALYSIS_PYTHON,'-c',code],cwd=analysis,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    status['completed_steps'].append(name);write(status_path,status)
            step('profiles','eval/plot_profile_performance.py','--extension-manifest',view_path)
            step('round_returns','eval/plot_round_return_progression.py','--manifest',view_path)
            step('training_curves','train/plot_protocol_training_curves.py','--out-dir',m['curves_root'])
            step('final_training_returns','train/plot_final_training_return_dots.py')
            step('per_seed_training_returns','train/plot_per_seed_training_returns.py')
            step('best_seed_returns','eval/plot_overcooked_paper_match.py','--manifest',view_path)
            selection=Path(m['comparison_root'])/'overcooked_paper_match/best_of_5/selection.json'
            step('selected_training_returns','train/plot_selected_training_returns.py','--selection',selection)
            step('assignments','eval/analyze_assignment_behaviors.py','--manifest',view_path,'--populations','test','train')
            step('selected_assignments','eval/plot_best_seed_assignment_behaviors.py','--manifest',view_path,'--selection',selection)
            step('selected_assignments_three_conditions','eval/plot_best_seed_assignment_behaviors.py','--manifest',view_path,'--selection',selection,'--exclude-no-influence')
            step('selected_three_metric_dashboard','eval/plot_selected_assignment_dashboard.py','--manifest',view_path,'--selection',selection)
            for population in ['train','test']:
                step('oracle_'+population,'eval/optimal_training_return.py','--population',population)
                step('oracle_validation_'+population,'diagnostics/validate_optimal_training_return.py','--population',population)
                step('oracle_plot_'+population,'eval/plot_training_return_ceiling.py','--population',population)
        elif stage=='representation':
            policies=[]
            for job in m['training_jobs']:
                if not job['condition'].startswith('rnn'):continue
                data=m['versions'][job['version']];source=Path(m['source_variants'][job['source_variant']]['frozen_source_root'])
                cfg=Path(data['checkpoint_root'])/(job['tag']+'_config.json')
                checkpoint=Path(data['checkpoint_root'])/(job['tag']+'.safetensors')
                rollouts={p:str((Path(data['evaluation_root'])/(job['tag']+'_'+p+'.h5')).resolve()) for p in ['train','test']}
                policies.append(dict(version=job['version'],condition=job['condition'],seed=job['seed'],
                    source_root=str(source),evaluator_sha256=sha(source/'analysis/evaluate_partner_modelling.py'),
                    config=str(cfg.resolve()),config_sha256=sha(cfg),checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha(checkpoint),
                    layouts_dir=m['frozen_layouts_dir'],allocation_protocol=data['allocation_protocol'],original_rollouts=rollouts,
                    original_stats={p:dict(size=Path(q).stat().st_size,mtime_ns=Path(q).stat().st_mtime_ns) for p,q in rollouts.items()}))
            inputs=analysis/('relative_speed_inputs'+SUFFIX+'.json');write(inputs,dict(status='verified_inputs',policies=policies,training_manifest=str(path)))
            root=Path(m['representation_root'])
            if not (root/'analysis_run.json').exists():step('prepare_representation','eval/run_relative_speed_representation.py','prepare',root,'--input-manifest',inputs)
            cpus=sorted(os.sched_getaffinity(0))[:3]
            step('representation_and_audits','eval/run_relative_speed_pipeline.py',root,'--cpus',*cpus)
        else:raise ValueError(stage)
        status.update(status='completed',completed_at_utc=now());write(status_path,status)
    except BaseException as error:
        status.update(status='failed',error=repr(error));write(status_path,status);raise


def finish(path):
    m=verify(path)
    output=Path(m['comparison_root'])/('pipeline'+SUFFIX)
    for stage in ['behavior','representation']:
        if load(output/(stage+'_status'+SUFFIX+'.json'))['status']!='completed':raise ValueError('Incomplete downstream: '+stage)
    repr_status=load(Path(m['representation_root'])/'pipeline_status.json')
    if repr_status['status']!='completed' or repr_status['policies']!=45:raise ValueError('Representation coverage')
    for j in m['training_jobs']:
        cfg=load(Path(m['versions'][j['version']]['checkpoint_root'])/(j['tag']+'_config.json'))
        if cfg['ENV_KWARGS']['step_penalty']!=.03 or cfg['NUM_UPDATES']!=915:raise ValueError('Completed config differs')
    m.update(status='completed',completed_at_utc=now());write(path,m)
    print('All sixty trainings, evaluations, plots, and 45 representation analyses completed.',flush=True)


def submit(path):
    path=Path(path).resolve();m=verify(path)
    if m['status']=='completed':return
    if not m['preflight']['passed']:raise ValueError('CPU preflight required')
    for filename in ['analysis_validation_step_penalty_0.03.json','representation_method_invariance_step_penalty_0.03.json']:
        if load(Path(m['infrastructure_root']).parent/filename)['status']!='passed':raise ValueError('Analysis validation required')
    clean={k:v for k,v in os.environ.items() if not k.startswith('SLURM_') and k not in ['JAX_PLATFORMS','CUDA_VISIBLE_DEVICES','XLA_FLAGS']}
    runner=Path(m['infrastructure_root'])/'bash/submit_step_penalty_training.py'
    wrapper=Path(m['infrastructure_root'])/'bash/run_counterbalanced_seed_extension.sh'
    def launch(stage,args=(),dependencies=(),gpu=True,hours=8):
        name='_'.join([stage,*map(str,args)])+SUFFIX
        cmd=['sbatch','--parsable','--account=nlp','--partition=sphinx','--mem=32G','--cpus-per-task=4',
            '--exclude=sphinx9','--time='+str(hours)+':00:00','--job-name='+name,
            '--chdir='+str(ROOT),'--export=ALL,GRID_PROJECT_ROOT='+str(ROOT),
            '--output='+str(Path(m['log_dir'])/(stage+'_%j'+SUFFIX+'.out')),
            '--error='+str(Path(m['log_dir'])/(stage+'_%j'+SUFFIX+'.err'))]
        if gpu:cmd+=['--constraint=80G','--gres=gpu:1']
        if dependencies:cmd+=['--dependency=afterok:'+':'.join(dependencies),'--kill-on-invalid-dep=yes']
        cmd += [str(wrapper),str(runner),stage,str(path),*map(str,args)]
        job_id=subprocess.check_output(cmd,cwd=ROOT,env=clean,text=True).strip().split(';')[0]
        if not job_id.isdigit():raise RuntimeError(job_id)
        print(stage,*args,job_id,flush=True)
        return dict(job_id=job_id,command=cmd)
    m['status']='submitting';write(path,m)
    specs=[('v1_matched','rnn_diverse_influence'),('v1_matched','rnn_diverse_noinfluence'),
           ('v2','rnn_diverse_influence'),('v3','rnn_diverse_influence'),('v3','mlp_diverse_influence')]
    checks=m['preflight']['production_checks']
    for version,condition in specs:
        if any(j['version']==version and j['condition']==condition for j in checks):continue
        checks.append(dict(version=version,condition=condition,**launch('production-smoke',[version,condition,1],hours=1)));write(path,m)
    for j in m['training_jobs']:
        if 'job_id' in j:continue
        j.update(launch('train',[j['version'],j['condition'],j['seed']], [c['job_id'] for c in checks]))
        j['log_path']=str(Path(m['log_dir'])/('train_'+j['job_id']+SUFFIX+'.out'));write(path,m)
    for version in VERSIONS:
        evaluations=m['versions'][version]['evaluation_jobs']
        for condition in CONDITIONS:
            if any(j['condition']==condition for j in evaluations):continue
            deps=[j['job_id'] for j in m['training_jobs'] if j['version']==version and j['condition']==condition]
            evaluations.append(dict(condition=condition,**launch('eval',[version,condition],deps,hours=2)));write(path,m)
    eval_ids=[j['job_id'] for d in m['versions'].values() for j in d['evaluation_jobs']]
    for stage,gpu,hours in [('behavior',False,4),('representation',False,8),('finish',False,1)]:
        if any(j['stage']==stage for j in m['downstream_jobs']):continue
        deps=eval_ids if stage!='finish' else [j['job_id'] for j in m['downstream_jobs']]
        m['downstream_jobs'].append(dict(stage=stage,**launch(stage,dependencies=deps,gpu=gpu,hours=hours)));write(path,m)
    m.update(status='submitted',submitted_at_utc=now());write(path,m)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','verify','validate','submit','train','production-smoke','smoke-eval','eval','behavior','representation','finish'])
    p.add_argument('manifest',nargs='?',type=Path);p.add_argument('version',nargs='?');p.add_argument('condition',nargs='?');p.add_argument('seed',nargs='?',type=int)
    a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode in ['train','production-smoke']:train(a.manifest,a.version,a.condition,a.seed,production=a.mode=='production-smoke')
    elif a.mode=='smoke-eval':
        m=verify(a.manifest);j=next(j for j in m['training_jobs'] if (j['version'],j['condition'],j['seed'])==(a.version,a.condition,1))
        target=ROOT/'train/run_snapshots'/m['batch']/'preflight'/a.version/a.condition
        evaluate_job(m,j,target)
    elif a.mode=='eval':
        m=verify(a.manifest)
        for j in m['training_jobs']:
            if j['version']==a.version and j['condition']==a.condition:evaluate_job(m,j)
    elif a.mode in ['behavior','representation']:downstream(a.manifest,a.mode)
    else:globals()[a.mode](a.manifest)


if __name__=='__main__':main()
