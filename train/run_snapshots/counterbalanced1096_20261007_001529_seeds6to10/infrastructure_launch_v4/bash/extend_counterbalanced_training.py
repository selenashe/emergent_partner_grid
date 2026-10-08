"""Extend the original categorical counterbalanced batch using its frozen methods.

Preparation creates reviewable configs but submits nothing. Validation executes
short CPU runs. Submission launches seeds 6–10, dependent evaluations, then an
all-ten-seed comparison. Historical artifacts remain untouched.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(os.environ.get('GRID_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repo_paths as path_resolver
# A frozen runner lives below the repository. Resolve historical locations
# against the actual shared workspace, using its own frozen path-resolution code.
path_resolver.ROOT = ROOT
from repo_paths import resolve_record_paths

BASE_BATCH = 'counterbalanced1096_20261002_235609'
CONDITIONS = ('rnn_diverse_influence', 'mlp_diverse_influence',
              'rnn_single_influence', 'rnn_diverse_noinfluence')
NEW_SEEDS = tuple(range(6, 11))
PYTHON = '/nlp/scr/jshe/miniconda3/envs/emergent_partner_model/bin/python'


def sha(path):
    """Fingerprint file bytes; unchanged hashes establish source identity."""
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, record):
    """Replace a manifest atomically so each submitted job ID is durable."""
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2) + '\n')
    tmp.replace(path)


def load(path):
    return resolve_record_paths(json.loads(Path(path).read_text()))


def relocated_config(base, version, condition):
    """Reuse a completed policy's settings, resolving obsolete directory names.

    Source code and scientific settings come from the original experiment.
    Changing a filename's location does not change what the model learns.
    """
    path = Path(base['versions'][version]['checkpoint_root']) / f'{condition}_seed1_config.json'
    return load(path), path


def verify_frozen_base(base):
    """Check every original source file, both schedules and all layout bytes.

    The original batch contains no greedy action-selection patch. Its exact
    sources preserve even protocol-specific historical behavior in v1 and v2.
    """
    if base.get('action_selection', 'categorical') != 'categorical':
        raise ValueError('Cannot extend a greedy batch')
    if base['batch'] != BASE_BATCH or base['status'] != 'completed':
        raise ValueError('Expected the completed original counterbalanced batch')
    digest = hashlib.sha256()
    layouts = sorted(Path(base['frozen_layouts_dir']).glob('*.json'))
    if len(layouts) != 1096:
        raise ValueError('Expected 1096 original layouts')
    for path in layouts:
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    if digest.hexdigest() != base['layout_files_sha256']:
        raise ValueError('Original layouts changed')
    for regime, path in base['schedule_paths'].items():
        if sha(path) != base['schedules'][regime]['sha256']:
            raise ValueError('Original schedule changed: ' + regime)
    for data in base['versions'].values():
        source = Path(data['frozen_source_root'])
        for relative, expected in data['source_sha256'].items():
            if sha(source / relative) != expected:
                raise ValueError('Original frozen source changed: ' + relative)
        trainer = (source / 'baselines/IPPO/ippo_rnn_coordination_grid.py').read_text()
        if 'select_action(' in trainer or 'pi.sample(seed=_rng)' not in trainer:
            raise ValueError('Original categorical action sampling must be preserved')


def prepare():
    """Create 40 configs, reusing each condition's exact original hyperparameters.

    Only learner seed and output locations change. Paths to the same frozen
    layouts and schedules are repaired in memory after repository relocation.
    Old and new evaluation directories are explicitly mapped to seeds 1–10.
    """
    base_path = ROOT / 'train/manifests' / f'sbatch_{BASE_BATCH}.json'
    base = load(base_path)
    verify_frozen_base(base)
    batch = 'counterbalanced1096_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_seeds6to10'
    work = ROOT / 'train/run_snapshots' / batch
    work.mkdir(parents=True, exist_ok=False)
    infrastructure = work / 'infrastructure'
    for relative in ('bash/extend_counterbalanced_training.py', 'bash/run_counterbalanced_seed_extension.sh',
                     'eval/compare_allocation_protocols.py', 'repo_paths.py'):
        destination = infrastructure / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    infrastructure_hashes = {str(p.relative_to(infrastructure)): sha(p)
                             for p in infrastructure.rglob('*') if p.is_file()}
    jobs, versions, inputs, templates = [], {}, {}, {}
    for version in ('v1', 'v2'):
        original = base['versions'][version]
        checkpoints = ROOT / 'train/train_logs' / f'{version}_balanced_training' / batch
        evaluations = ROOT / 'eval/eval_out' / f'{version}_balanced_training' / batch
        checkpoints.mkdir(parents=True, exist_ok=False)
        evaluations.mkdir(parents=True, exist_ok=False)
        versions[version] = dict(original, checkpoint_root=str(checkpoints),
                                 evaluation_root=str(evaluations), evaluation_jobs=[])
        inputs[version] = {str(seed): original['evaluation_root'] if seed <= 5 else str(evaluations)
                           for seed in range(1, 11)}
        for condition in CONDITIONS:
            template, template_path = relocated_config(base, version, condition)
            if template.get('ACTION_SELECTION', 'categorical') != 'categorical':
                raise ValueError('Unexpected action selection in completed config')
            templates[str(template_path)] = sha(template_path)
            # Every original seed must have the same scientific settings; differing
            # learner seeds and checkpoint filenames are the two intended differences.
            reference = {k: v for k, v in template.items() if k not in ('SEED', 'SAVE_PARAMS_PATH')}
            for seed in range(1, 6):
                old_path = template_path.with_name(f'{condition}_seed{seed}_config.json')
                old = load(old_path)
                if {k: v for k, v in old.items() if k not in ('SEED', 'SAVE_PARAMS_PATH')} != reference:
                    raise ValueError(f'Original seed settings disagree: {old_path}')
                templates[str(old_path)] = sha(old_path)
            for seed in NEW_SEEDS:
                tag = f'{condition}_seed{seed}'
                cfg = dict(template, SEED=seed, SAVE_PARAMS_PATH=str(checkpoints / f'{tag}.safetensors'))
                cfg_path = work / 'configs' / version / f'{tag}.yaml'
                cfg_path.parent.mkdir(parents=True, exist_ok=True)
                cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                jobs.append(dict(version=version, condition=condition, seed=seed, tag=tag,
                                 config_path=str(cfg_path), config_sha256=sha(cfg_path)))
    log_dir = ROOT / 'train/slurm_logs' / batch
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(batch=batch, status='prepared', prepared_at_utc=datetime.now(timezone.utc).isoformat(),
                    base_manifest=str(base_path), base_manifest_sha256=sha(base_path),
                    action_selection='categorical', new_seeds=list(NEW_SEEDS), evaluation_seeds=list(range(1, 11)),
                    frozen_layouts_dir=base['frozen_layouts_dir'], schedule_paths=base['schedule_paths'],
                    versions=versions, training_jobs=jobs, evaluation_inputs=inputs,
                    template_sha256=templates, infrastructure_root=str(infrastructure),
                    infrastructure_sha256=infrastructure_hashes, log_dir=str(log_dir),
                    aggregate_root=str(ROOT / 'eval/protocol_comparison' / batch / 'all_10_seeds'),
                    nominal_timesteps_per_policy=60000000, effective_timesteps_per_policy=59965440,
                    evaluation=base['evaluation'], resources_per_training_job=base['resources_per_training_job'],
                    aggregation_jobs=[], preflight={})
    path = ROOT / 'train/manifests' / f'sbatch_{batch}.json'
    write(path, manifest)
    verify(path)
    print(path, flush=True)
    return path


def verify(path):
    """Reject changed inputs, duplicate/missing runs, or scientific config drift."""
    m = load(path)
    if sha(m['base_manifest']) != m['base_manifest_sha256']:
        raise ValueError('Original manifest changed')
    base = load(m['base_manifest'])
    verify_frozen_base(base)
    if m['action_selection'] != 'categorical' or m['evaluation_seeds'] != list(range(1, 11)):
        raise ValueError('All-ten evaluation must use original categorical methods')
    expected = {(v, c, s) for v in ('v1', 'v2') for c in CONDITIONS for s in NEW_SEEDS}
    actual = [(j['version'], j['condition'], j['seed']) for j in m['training_jobs']]
    if len(actual) != 40 or set(actual) != expected:
        raise ValueError('Expected exactly 40 distinct new policies')
    for version, data in m['versions'].items():
        original = base['versions'][version]
        for key in ('frozen_source_root', 'source_sha256', 'allocation_protocol'):
            if data[key] != original[key]:
                raise ValueError('Extension must use the original frozen protocol source')
        wanted_inputs = {str(seed): original['evaluation_root'] if seed <= 5 else data['evaluation_root']
                         for seed in range(1, 11)}
        if m['evaluation_inputs'][version] != wanted_inputs:
            raise ValueError('Evaluation seed mapping differs from original plus extension')
    for relative, expected_sha in m['infrastructure_sha256'].items():
        if sha(Path(m['infrastructure_root']) / relative) != expected_sha:
            raise ValueError('Extension infrastructure changed: ' + relative)
    for template_path, expected_sha in m['template_sha256'].items():
        if sha(template_path) != expected_sha:
            raise ValueError('Original resolved config changed')
    for j in m['training_jobs']:
        if sha(j['config_path']) != j['config_sha256']:
            raise ValueError('Extension config changed')
        cfg = yaml.safe_load(Path(j['config_path']).read_text())
        original, _ = relocated_config(base, j['version'], j['condition'])
        wanted = dict(original, SEED=j['seed'], SAVE_PARAMS_PATH=str(
            Path(m['versions'][j['version']]['checkpoint_root']) / f"{j['tag']}.safetensors"))
        if cfg != wanted:
            raise ValueError('Scientific settings differ from the original batch')
    return m


def environment(source, cpu=False):
    """Isolate imports to the original snapshot; select CPU only for preflight."""
    env = os.environ.copy()
    env.update(PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1', HYDRA_FULL_ERROR='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if cpu:
        env.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', XLA_FLAGS='')
    return env


def training_command(m, job, smoke=False, gpu_smoke=False, production_smoke=False):
    """Invoke the original Hydra trainer with the extension's resolved config.

    Absolute config directories avoid editing the historical source snapshot.
    Smoke runs shorten only compute budget and episode horizons; production
    runs receive the original 60M-step settings without those overrides.
    """
    source = Path(m['versions'][job['version']]['frozen_source_root'])
    cfg_path = Path(job['config_path'])
    target = ROOT / 'train/hydra_outputs' / m['batch'] / job['version'] / job['tag']
    command = [sys.executable, '-u', str(source / 'baselines/IPPO/ippo_rnn_coordination_grid.py'),
               '--config-path', str(cfg_path.parent), '--config-name', cfg_path.stem,
               f'hydra.run.dir={target}']
    if smoke:
        target = ROOT / 'train/run_snapshots' / m['batch'] / ('slurm_preflight' if gpu_smoke else 'preflight') / job['version'] / job['condition']
        target.mkdir(parents=True, exist_ok=True)
        command += ['NUM_ENVS=4', 'NUM_STEPS=64', 'TOTAL_TIMESTEPS=256', 'UPDATE_EPOCHS=1',
                    'NUM_MINIBATCHES=4', 'ENV_KWARGS.max_steps=3', 'EVAL_EPISODES_PER_CAPABILITY=1',
                    f'SAVE_PARAMS_PATH={target / "params.safetensors"}', 'STDOUT_SUMMARY=false',
                    f'hydra.run.dir={target / "hydra"}']
    if production_smoke:
        # Exercise the actual convolution/recurrent/PPO tensor dimensions. Tiny
        # CPU-shaped checks can select different cuDNN kernels on GPU and cannot
        # certify the production batch. Only the duration and eval repetitions
        # are shortened; rollout dimensions, epochs and minibatches are original.
        cfg = yaml.safe_load(cfg_path.read_text())
        budget = int(cfg['NUM_ENVS']) * int(cfg['NUM_STEPS'])
        target = ROOT / 'train/run_snapshots' / m['batch'] / 'slurm_production_preflight' / job['version'] / job['condition']
        target.mkdir(parents=True, exist_ok=True)
        command += [f'TOTAL_TIMESTEPS={budget}', 'EVAL_EPISODES_PER_CAPABILITY=1',
                    f'SAVE_PARAMS_PATH={target / "params.safetensors"}', 'STDOUT_SUMMARY=false',
                    f'hydra.run.dir={target / "hydra"}']
    return command, source, target


def validate(path):
    """Train/evaluate one new seed in every protocol/condition before submission.

    Eight short runs exercise paths, categorical rollout sampling, PPO updates,
    checkpoint writing, queue audits and built-in familiar/novel evaluations.
    Successful base preflight is evidence about the original standalone evaluator.
    """
    m = verify(path)
    records = m['preflight'].get('smoke_runs', [])
    m['preflight'] = dict(passed=False, smoke_runs=records)
    write(path, m)
    cores = ','.join(map(str, sorted(os.sched_getaffinity(0))[:4]))
    for job in m['training_jobs']:
        if job['seed'] != 6 or any(r['version'] == job['version'] and r['condition'] == job['condition'] for r in records):
            continue
        command, source, target = training_command(m, job, smoke=True)
        log = target / 'smoke.log'
        print('[preflight]', job['version'], job['condition'], flush=True)
        with log.open('w') as handle:
            result = subprocess.run(['taskset', '-c', cores] + command, cwd=source,
                                    env=environment(source, cpu=True), stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(log.read_text()[-12000:])
        audit = json.loads((target / 'params_sampling_audit.json').read_text())
        if audit['collected_environment_steps'] != 256 or audit['allocated_episode_count_difference'] > 1:
            raise ValueError('Smoke queue audit failed')
        if not (target / 'params_eval.json').exists():
            raise ValueError('Built-in evaluation missing')
        records.append(dict(version=job['version'], condition=job['condition'], seed=6,
                            passed=True, log=str(log), collected_environment_steps=256))
        write(path, m)
        print('[preflight passed]', job['version'], job['condition'], flush=True)
    verify(path)
    if len(records) != 8:
        raise ValueError('Eight smoke runs are required')
    m['preflight'].update(passed=True, original_source_hashes_verified=True,
                          exact_original_hyperparameters_verified=True,
                          completed_at_utc=datetime.now(timezone.utc).isoformat())
    write(path, m)


def run_training(path, tag, version, smoke=False, production_smoke=False):
    """Run one full new policy without modifying the original frozen trainer."""
    m = verify(path)
    job = next(j for j in m['training_jobs'] if j['tag'] == tag and j['version'] == version)
    command, source, target = training_command(m, job, smoke=smoke, gpu_smoke=smoke, production_smoke=production_smoke)
    subprocess.run(command, cwd=source, env=environment(source), check=True)
    if smoke or production_smoke:
        audit = json.loads((target / "params_sampling_audit.json").read_text())
        expected_steps = 65536 if production_smoke else 256
        if audit["collected_environment_steps"] != expected_steps or not (target / "params_eval.json").exists():
            raise ValueError("Slurm GPU preflight did not complete training and evaluation")
        print("[Slurm GPU preflight passed]", version, tag, flush=True)


def run_evaluation(path, version):
    """Evaluate every new checkpoint using the original categorical evaluator.

    Fixed evaluation seed 12345 and 20 episodes per capability match the
    completed first five policies. Export RNN hidden states as before. Both
    familiar (24 profiles) and novel (22 profiles) populations are evaluated.
    """
    m = verify(path)
    data = m['versions'][version]
    source = Path(data['frozen_source_root'])
    jobs = [j for j in m['training_jobs'] if j['version'] == version]
    for job in jobs:
        prefix = Path(data['evaluation_root']) / job['tag']
        checkpoint = Path(data['checkpoint_root']) / job['tag']
        command = [sys.executable, '-u', str(source / 'analysis/evaluate_partner_modelling.py'),
                   '--config', str(checkpoint) + '_config.json', '--params', str(checkpoint) + '.safetensors',
                   '--layouts_dir', m['frozen_layouts_dir'], '--n_episodes_per_capability', '20',
                   '--seed', '12345', '--out_prefix', str(prefix)]
        if job['condition'].startswith('rnn'):
            command.append('--save_hidden')
        print('[evaluation]', version, job['tag'], flush=True)
        subprocess.run(command, cwd=source, env=environment(source), check=True)
    # Summaries alone do not establish that complete rollout files exist.
    for job in jobs:
        prefix = Path(data['evaluation_root']) / job['tag']
        for suffix in ('_summary.json', '_train.h5', '_test.h5'):
            if not Path(str(prefix) + suffix).is_file():
                raise ValueError('Missing evaluation artifact: ' + str(prefix) + suffix)


def run_aggregation(path):
    """Average all ten independent learner seeds equally, with sample SD.

    Older seeds read their original evaluation files. Newer seeds read this
    extension's files. No best-seed selection enters either population average.
    The comparison validates all 80 policy results per population before saving.
    """
    m = verify(path)
    comparison = Path(m['infrastructure_root']) / 'eval/compare_allocation_protocols.py'
    for population in ('train', 'test'):
        subprocess.run([sys.executable, str(comparison), '--extension-manifest', str(path),
                        '--population', population], check=True, env=environment(Path(m['infrastructure_root'])))
    m = load(path)
    m.update(status='completed', completed_at_utc=datetime.now(timezone.utc).isoformat())
    write(path, m)


def submit(path):
    """Submit 40 policies and queue evaluation/aggregation only after success.

    Save each Slurm ID immediately. A repeated submission skips IDs already
    recorded, preventing accidental duplicate training. Each evaluation waits
    for its 20 policies, and aggregation waits for both evaluations.
    """
    m = verify(path)
    if m['status'] == 'completed':
        print('Extension already completed; no jobs submitted.')
        return
    if not m['preflight'].get('passed'):
        raise ValueError('Passed preflight is required before submission')
    env = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_')
           and k not in ('JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES', 'XLA_FLAGS')}
    infrastructure = Path(m['infrastructure_root'])
    wrapper = infrastructure / 'bash/run_counterbalanced_seed_extension.sh'
    runner = infrastructure / 'bash/extend_counterbalanced_training.py'
    common = ['sbatch', '--parsable', '--account=nlp', '--partition=sphinx', '--constraint=80G',
              '--gres=gpu:1', '--mem=32G', '--cpus-per-task=4', '--exclude=sphinx9',
              f'--chdir={ROOT}', '--export=ALL,GRID_PROJECT_ROOT=' + str(ROOT)]
    m['status'] = 'submitting'; write(path, m)
    def launch(command):
        result = subprocess.check_output(command, cwd=ROOT, env=env, text=True).strip().split(';')[0]
        if not result.isdigit():
            raise RuntimeError('Unexpected sbatch result: ' + result)
        return result
    for job in m['training_jobs']:
        if 'job_id' in job:
            continue
        # A repair can queue production while its compute-node launcher checks
        # wait for resources. Slurm must report every check successful first.
        checks = m['preflight'].get('slurm_launcher_checks', [])
        guard = []
        if checks:
            dependency = 'afterok:' + ':'.join(check['job_id'] for check in checks)
            guard = ['--dependency=' + dependency, '--kill-on-invalid-dep=yes']
            job['launcher_check_dependency'] = dependency
        command = common + guard + ['--time=08:00:00', f"--job-name=cb_{job['version']}_{job['tag']}",
                            f"--output={m['log_dir']}/train_%j.out", f"--error={m['log_dir']}/train_%j.err",
                            str(wrapper), str(runner), 'train', str(path), job['version'], job['tag']]
        job.update(job_id=launch(command), command=command)
        write(path, m)
        print(job['version'], job['tag'], job['job_id'], flush=True)
    for version, data in m['versions'].items():
        if data['evaluation_jobs']:
            continue
        dependency = 'afterok:' + ':'.join(j['job_id'] for j in m['training_jobs'] if j['version'] == version)
        command = common + ['--time=02:00:00', f'--job-name=cb_{version}_eval_seeds6to10',
                            '--dependency=' + dependency, '--kill-on-invalid-dep=yes',
                            f"--output={m['log_dir']}/eval_%j.out", f"--error={m['log_dir']}/eval_%j.err",
                            str(wrapper), str(runner), 'eval', str(path), version]
        data['evaluation_jobs'].append(dict(job_id=launch(command), dependency=dependency, command=command))
        write(path, m)
        print(version, 'evaluation', data['evaluation_jobs'][0]['job_id'], flush=True)
    if not m['aggregation_jobs']:
        dependency = 'afterok:' + ':'.join(v['evaluation_jobs'][0]['job_id'] for v in m['versions'].values())
        # Aggregation requires no GPU; retain the same partition and CPU resources.
        command = [x for x in common if x != '--gres=gpu:1'] + [
            '--time=01:00:00', '--job-name=cb_all10_aggregate', '--dependency=' + dependency,
            '--kill-on-invalid-dep=yes', f"--output={m['log_dir']}/aggregate_%j.out",
            f"--error={m['log_dir']}/aggregate_%j.err", str(wrapper), str(runner), 'aggregate', str(path)]
        m['aggregation_jobs'].append(dict(job_id=launch(command), dependency=dependency, command=command))
        write(path, m)
        print('all-ten aggregation', m['aggregation_jobs'][0]['job_id'], flush=True)
    m.update(status='submitted', submitted_at_utc=datetime.now(timezone.utc).isoformat())
    write(path, m)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'validate', 'verify', 'submit', 'train', 'eval', 'aggregate', 'gpu-smoke', 'production-smoke'))
    parser.add_argument('manifest', nargs='?', type=Path)
    parser.add_argument('version', nargs='?', choices=('v1', 'v2'))
    parser.add_argument('tag', nargs='?')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.manifest is None:
        parser.error('Manifest path required')
    elif args.mode in ('train', 'gpu-smoke', 'production-smoke'):
        run_training(args.manifest, args.tag, args.version, smoke=args.mode == 'gpu-smoke', production_smoke=args.mode == 'production-smoke')
    elif args.mode == 'eval':
        run_evaluation(args.manifest, args.version)
    elif args.mode == 'aggregate':
        run_aggregation(args.manifest)
    else:
        globals()[args.mode](args.manifest)
