"""Train v3 seeds 1–10 in four conditions, matched to executed v2 methods.

Stages freeze and verify scientific inputs, run CPU/GPU preflight, submit
training, evaluate each condition, and plot v1/v2/v3 using all ten seeds.
Submission is resumable: every Slurm ID is saved before submitting another job.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(os.environ.get('GRID_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bash import extend_counterbalanced_training as original
from train.v3_protocol import upgrade_environment

original.ROOT = ROOT
original.path_resolver.ROOT = ROOT
CONDITIONS = original.CONDITIONS
SEEDS = tuple(range(1, 11))
BASE_MANIFEST = ROOT / 'train/manifests' / f'sbatch_{original.BASE_BATCH}.json'
EXTENSION_MANIFEST = ROOT / 'train/manifests/sbatch_counterbalanced1096_20261007_001529_seeds6to10.json'
ENVIRONMENT_FILE = 'jaxmarl/environments/coordination_grid/coordination_grid.py'
sha, write, load, environment = original.sha, original.write, original.load, original.environment


def prepare():
    base, extension = load(BASE_MANIFEST), load(EXTENSION_MANIFEST)
    original.verify_frozen_base(base)
    if extension['status'] != 'completed' or extension['evaluation_seeds'] != list(SEEDS):
        raise ValueError('Expected completed categorical v1/v2 evaluations for seeds 1–10')
    batch = 'counterbalanced1096_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_v3'
    work = ROOT / 'train/run_snapshots' / batch
    work.mkdir(parents=True, exist_ok=False)
    infrastructure = work / 'infrastructure'
    for relative in ('bash/submit_v3_counterbalanced_training.py', 'bash/extend_counterbalanced_training.py',
                     'bash/run_counterbalanced_seed_extension.sh', 'train/v3_protocol.py', 'train/__init__.py',
                     'eval/__init__.py', 'eval/compare_allocation_protocols.py',
                     'eval/plot_profile_performance.py', 'repo_paths.py'):
        destination = infrastructure / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    source = work / 'v3_balanced_training/source'
    v2_source = Path(base['versions']['v2']['frozen_source_root'])
    # Copy exactly the manifest's source files, excluding bytecode and stray outputs.
    for relative in base['versions']['v2']['source_sha256']:
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(v2_source / relative, destination)
    (source / ENVIRONMENT_FILE).write_text(upgrade_environment((v2_source / ENVIRONMENT_FILE).read_text()))
    checkpoints = ROOT / 'train/train_logs/v3_balanced_training' / batch
    evaluations = ROOT / 'eval/eval_out/v3_balanced_training' / batch
    checkpoints.mkdir(parents=True, exist_ok=False)
    evaluations.mkdir(parents=True, exist_ok=False)
    jobs, templates = [], {}
    for condition in CONDITIONS:
        template, template_path = original.relocated_config(base, 'v2', condition)
        templates[str(template_path)] = sha(template_path)
        for seed in SEEDS:
            tag = f'{condition}_seed{seed}'
            cfg = dict(template, ALLOCATION_PROTOCOL='online_v3', SEED=seed,
                       SAVE_PARAMS_PATH=str(checkpoints / f'{tag}.safetensors'))
            path = work / 'configs/v3' / f'{tag}.yaml'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            jobs.append(dict(version='v3', condition=condition, seed=seed, tag=tag,
                             config_path=str(path), config_sha256=sha(path)))
    data = dict(frozen_source_root=str(source), allocation_protocol='online_v3',
                source_sha256={p: sha(source / p) for p in base['versions']['v2']['source_sha256']},
                checkpoint_root=str(checkpoints), evaluation_root=str(evaluations), evaluation_jobs=[])
    inputs = dict(extension['evaluation_inputs'])
    inputs['v3'] = {str(seed): str(evaluations) for seed in SEEDS}
    logs = ROOT / 'train/slurm_logs' / batch
    logs.mkdir(parents=True, exist_ok=True)
    m = dict(batch=batch, status='prepared', prepared_at_utc=datetime.now(timezone.utc).isoformat(),
             base_manifest=str(BASE_MANIFEST), base_manifest_sha256=sha(BASE_MANIFEST),
             comparison_manifest=str(EXTENSION_MANIFEST), comparison_manifest_sha256=sha(EXTENSION_MANIFEST),
             action_selection='categorical', sampling_protocol=base['sampling_protocol'],
             new_seeds=list(SEEDS), evaluation_seeds=list(SEEDS),
             frozen_layouts_dir=base['frozen_layouts_dir'], layout_files_sha256=base['layout_files_sha256'],
             schedule_paths=base['schedule_paths'], versions={'v3': data},
             training_jobs=jobs, evaluation_inputs=inputs, template_sha256=templates,
             infrastructure_root=str(infrastructure),
             infrastructure_sha256={str(p.relative_to(infrastructure)): sha(p)
                                    for p in infrastructure.rglob('*') if p.is_file()},
             log_dir=str(logs), aggregate_root=str(ROOT / 'eval/protocol_comparison' / batch / 'all_10_seeds'),
             nominal_timesteps_per_policy=60000000, effective_timesteps_per_policy=59965440,
             evaluation=base['evaluation'], resources_per_training_job=base['resources_per_training_job'],
             scientific_change='Influence-enabled t=0 allocation is chosen by ego; cooldown stays zero. '
                               'All t>=1 transitions and no-influence transitions retain v2 rules.',
             unchanged_from_v2=['trainer', 'architecture', 'PPO', 'evaluator', 'reward', 'layouts',
                                'counterbalanced schedules', 'training budget'],
             aggregation_jobs=[], preflight={})
    path = ROOT / 'train/manifests' / f'sbatch_{batch}.json'
    write(path, m)
    verify(path)
    print(path, flush=True)
    return path


def verify(path):
    m = load(path)
    for key in ('base_manifest', 'comparison_manifest'):
        if sha(m[key]) != m[key + '_sha256']:
            raise ValueError('Historical manifest changed: ' + key)
    base, extension = load(m['base_manifest']), load(m['comparison_manifest'])
    original.verify_frozen_base(base)
    if (m['action_selection'] != 'categorical' or m['evaluation_seeds'] != list(SEEDS)
            or m['schedule_paths'] != base['schedule_paths']
            or m['frozen_layouts_dir'] != base['frozen_layouts_dir']
            or m['nominal_timesteps_per_policy'] != 60000000
            or m['effective_timesteps_per_policy'] != 59965440):
        raise ValueError('V3 must match v2 sampling, layouts, schedules, seeds and budget')
    jobs = m['training_jobs']
    if len(jobs) != 40 or {(j['version'], j['condition'], j['seed']) for j in jobs} != {
            ('v3', c, s) for c in CONDITIONS for s in SEEDS}:
        raise ValueError('Expected 40 distinct v3 policies')
    data = m['versions']['v3']
    source, v2_source = Path(data['frozen_source_root']), Path(base['versions']['v2']['frozen_source_root'])
    if data['allocation_protocol'] != 'online_v3':
        raise ValueError('Expected online_v3')
    if set(data['source_sha256']) != set(base['versions']['v2']['source_sha256']):
        raise ValueError('V3 source inventory differs from v2')
    for relative, digest in data['source_sha256'].items():
        if sha(source / relative) != digest:
            raise ValueError('Frozen v3 source changed: ' + relative)
        if relative == ENVIRONMENT_FILE:
            if (source / relative).read_text() != upgrade_environment((v2_source / relative).read_text()):
                raise ValueError('V3 environment differs from the initialization-only transformation')
        elif digest != base['versions']['v2']['source_sha256'][relative]:
            raise ValueError('V3 must preserve v2 source bytes: ' + relative)
    expected_inputs = dict(extension['evaluation_inputs'])
    expected_inputs['v3'] = {str(seed): data['evaluation_root'] for seed in SEEDS}
    if m['evaluation_inputs'] != expected_inputs:
        raise ValueError('V1/v2/v3 evaluation mapping differs from the matched ten seeds')
    for relative, digest in m['infrastructure_sha256'].items():
        if sha(Path(m['infrastructure_root']) / relative) != digest:
            raise ValueError('Frozen infrastructure changed: ' + relative)
    for template, digest in m['template_sha256'].items():
        if sha(template) != digest:
            raise ValueError('V2 config changed: ' + template)
    for job in jobs:
        if sha(job['config_path']) != job['config_sha256']:
            raise ValueError('Frozen config changed: ' + job['tag'])
        template, _ = original.relocated_config(base, 'v2', job['condition'])
        wanted = dict(template, ALLOCATION_PROTOCOL='online_v3', SEED=job['seed'],
                      SAVE_PARAMS_PATH=str(Path(data['checkpoint_root']) / f"{job['tag']}.safetensors"))
        if yaml.safe_load(Path(job['config_path']).read_text()) != wanted:
            raise ValueError('V3 scientific config drift: ' + job['tag'])
    return m


def validate(path):
    m = verify(path)
    records = m['preflight'].get('smoke_runs', [])
    m['preflight'].update(passed=False, smoke_runs=records)
    write(path, m)
    cores = ','.join(map(str, sorted(os.sched_getaffinity(0))[:4]))
    for condition in CONDITIONS:
        if any(r['condition'] == condition and r['passed'] for r in records):
            continue
        job = next(j for j in m['training_jobs'] if j['condition'] == condition and j['seed'] == 1)
        command, source, target = original.training_command(m, job, smoke=True)
        log = target / 'smoke.log'
        print('[CPU preflight]', condition, flush=True)
        with log.open('w') as handle:
            subprocess.run(['taskset', '-c', cores] + command, cwd=source, env=environment(source, cpu=True),
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        audit = json.loads((target / 'params_sampling_audit.json').read_text())
        if audit['collected_environment_steps'] != 256 or audit['allocated_episode_count_difference'] > 1:
            raise ValueError('V3 queue smoke audit failed')
        if not (target / 'params_eval.json').exists():
            raise ValueError('Built-in evaluation missing')
        # Exercise the unchanged standalone v2 evaluator under the v3 environment.
        prefix = target / 'standalone_eval'
        command = [sys.executable, str(source / 'analysis/evaluate_partner_modelling.py'),
                   '--config', str(target / 'params_config.json'), '--params', str(target / 'params.safetensors'),
                   '--layouts_dir', m['frozen_layouts_dir'], '--n_episodes_per_capability', '1',
                   '--seed', '12345', '--out_prefix', str(prefix)]
        if condition.startswith('rnn'):
            command.append('--save_hidden')
        with (target / 'standalone_evaluator.log').open('w') as handle:
            subprocess.run(['taskset', '-c', cores] + command, cwd=source, env=environment(source, cpu=True),
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        summaries = json.loads(Path(str(prefix) + '_summary.json').read_text())
        if any(s['allocation_protocol'] != 'online_v3' for s in summaries.values()):
            raise ValueError('Wrong standalone evaluation protocol')
        records.append(dict(condition=condition, seed=1, passed=True, log=str(log),
                            collected_environment_steps=256, standalone_evaluation_passed=True))
        write(path, m)
        print('[CPU preflight passed]', condition, flush=True)
    m['preflight'].update(passed=True, completed_at_utc=datetime.now(timezone.utc).isoformat())
    write(path, m)


def run_training(path, tag, production_smoke=False):
    m = verify(path)
    job = next(j for j in m['training_jobs'] if j['tag'] == tag)
    command, source, target = original.training_command(m, job, production_smoke=production_smoke)
    subprocess.run(command, cwd=source, env=environment(source), check=True)
    if production_smoke:
        audit = json.loads((target / 'params_sampling_audit.json').read_text())
        if audit['collected_environment_steps'] != 65536 or not (target / 'params_eval.json').exists():
            raise ValueError('GPU preflight did not complete the production tensor dimensions')
        print('[GPU production preflight passed]', tag, flush=True)


def run_evaluation(path, condition):
    m = verify(path)
    data = m['versions']['v3']
    source = Path(data['frozen_source_root'])
    for job in m['training_jobs']:
        if job['condition'] != condition:
            continue
        checkpoint = Path(data['checkpoint_root']) / job['tag']
        prefix = Path(data['evaluation_root']) / job['tag']
        command = [sys.executable, '-u', str(source / 'analysis/evaluate_partner_modelling.py'),
                   '--config', str(checkpoint) + '_config.json', '--params', str(checkpoint) + '.safetensors',
                   '--layouts_dir', m['frozen_layouts_dir'], '--n_episodes_per_capability', '20',
                   '--seed', '12345', '--out_prefix', str(prefix)]
        if condition.startswith('rnn'):
            command.append('--save_hidden')
        print('[evaluation]', job['tag'], flush=True)
        subprocess.run(command, cwd=source, env=environment(source), check=True)
        for suffix in ('_summary.json', '_train.h5', '_test.h5'):
            if not Path(str(prefix) + suffix).is_file():
                raise ValueError('Missing evaluation artifact: ' + str(prefix) + suffix)


def run_aggregation(path):
    m = verify(path)
    infrastructure = Path(m['infrastructure_root'])
    for population in ('train', 'test'):
        subprocess.run([sys.executable, str(infrastructure / 'eval/compare_allocation_protocols.py'),
                        '--extension-manifest', str(path), '--population', population],
                       env=environment(infrastructure), check=True)
    subprocess.run([sys.executable, str(infrastructure / 'eval/plot_profile_performance.py'),
                    '--extension-manifest', str(path)], env=environment(infrastructure), check=True)
    m = load(path)
    m.update(status='completed', completed_at_utc=datetime.now(timezone.utc).isoformat())
    write(path, m)


def submit(path):
    m = verify(path)
    if m['status'] == 'completed':
        print('V3 already completed; no jobs submitted.')
        return
    if not m['preflight'].get('passed'):
        raise ValueError('Passed CPU preflight is required')
    env = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_')
           and k not in ('JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES', 'XLA_FLAGS')}
    infrastructure = Path(m['infrastructure_root'])
    wrapper = str(infrastructure / 'bash/run_counterbalanced_seed_extension.sh')
    runner = str(infrastructure / 'bash/submit_v3_counterbalanced_training.py')
    common = ['sbatch', '--parsable', '--account=nlp', '--partition=sphinx', '--constraint=80G',
              '--gres=gpu:1', '--mem=32G', '--cpus-per-task=4', '--exclude=sphinx9',
              f'--chdir={ROOT}', '--export=ALL,GRID_PROJECT_ROOT=' + str(ROOT)]
    def launch(stage, name, time, arguments, dependencies=(), gpu=True):
        command = (common if gpu else [x for x in common if x != '--gres=gpu:1']) + [
            '--time=' + time, '--job-name=' + name,
            f"--output={m['log_dir']}/{stage}_%j.out", f"--error={m['log_dir']}/{stage}_%j.err"]
        if dependencies:
            command += ['--dependency=afterok:' + ':'.join(dependencies), '--kill-on-invalid-dep=yes']
        command += [wrapper, runner, stage, str(path), *arguments]
        result = subprocess.check_output(command, cwd=ROOT, env=env, text=True).strip().split(';')[0]
        if not result.isdigit():
            raise RuntimeError('Unexpected sbatch result: ' + result)
        print(stage, name, result, flush=True)
        return dict(job_id=result, command=command)
    m['status'] = 'submitting'
    write(path, m)
    checks = m['preflight'].setdefault('slurm_launcher_checks', [])
    for condition in ('rnn_diverse_influence', 'mlp_diverse_influence'):
        if any(j['condition'] == condition for j in checks):
            continue
        checks.append(dict(launch('production-smoke', 'cb_v3_check_' + condition, '00:30:00',
                                  [condition + '_seed1']), condition=condition))
        write(path, m)
    for job in m['training_jobs']:
        if 'job_id' in job:
            continue
        job.update(launch('train', 'cb_v3_' + job['tag'], '08:00:00', [job['tag']],
                          [j['job_id'] for j in checks]))
        write(path, m)
    evaluations = m['versions']['v3']['evaluation_jobs']
    for condition in CONDITIONS:
        if any(j['condition'] == condition for j in evaluations):
            continue
        dependencies = [j['job_id'] for j in m['training_jobs'] if j['condition'] == condition]
        evaluations.append(dict(launch('eval', 'cb_v3_eval_' + condition, '02:00:00', [condition], dependencies),
                                condition=condition))
        write(path, m)
    if not m['aggregation_jobs']:
        m['aggregation_jobs'].append(launch('aggregate', 'cb_v1_v2_v3_plots', '01:00:00', [],
                                            [j['job_id'] for j in evaluations], gpu=False))
        write(path, m)
    m.update(status='submitted', submitted_at_utc=datetime.now(timezone.utc).isoformat())
    write(path, m)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'verify', 'validate', 'submit', 'train',
                                       'production-smoke', 'eval', 'aggregate'))
    parser.add_argument('manifest', nargs='?', type=Path)
    parser.add_argument('target', nargs='?')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.manifest is None:
        parser.error('Manifest path required')
    elif args.mode in ('train', 'production-smoke'):
        run_training(args.manifest, args.target, production_smoke=args.mode == 'production-smoke')
    elif args.mode == 'eval':
        run_evaluation(args.manifest, args.target)
    elif args.mode == 'aggregate':
        run_aggregation(args.manifest)
    else:
        globals()[args.mode](args.manifest)
