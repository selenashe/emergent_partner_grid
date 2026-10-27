"""Run isolated behavior and representation analyses of the completed five-seed cohort."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(os.environ.get('GRID_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
INFRA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INFRA))
from bash import submit_t0fix_direct_input_training as training
from bash.step_penalty_downstream import freeze

SUFFIX = training.SUFFIX
SEEDS = training.SEEDS
VERSIONS = training.VERSIONS
PYTHON = '/juice6/u/nlp/climblab/tmp/epg_repr_all10_20261008/bin/python'
sha, write = training.sha, training.write


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def prepare(path):
    path = Path(path).resolve()
    base = training.verify(path)
    if base['status'] != 'completed':
        raise ValueError('Completed training and evaluations are required')
    batch = 'analyses_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + SUFFIX
    work = ROOT / 'train/run_snapshots' / batch
    work.mkdir(parents=True, exist_ok=False)
    infrastructure = work / 'infrastructure'
    for rel in ('bash/run_t0fix_direct_input_analyses.py', 'bash/submit_t0fix_direct_input_training.py',
                'bash/step_penalty_downstream.py', 'bash/run_counterbalanced_seed_extension.sh',
                'repo_paths.py', 'train/__init__.py', 'train/experiment.py',
                'train/experiment_runtime.py', 'train/cohort_integrity.py',
                'train/prepare_experiment.py', 'train/observation_pipeline.py',
                'train/ippo_rnn_coordination_grid.py', 'diagnostics/validate_t0fix_direct_input_analyses.py'):
        dst = infrastructure / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
    analysis = work / ('analysis_source' + SUFFIX)
    behavior = Path(base['comparison_root']) / ('behavioral_analysis' + SUFFIX)
    representations = ROOT / 'eval/representation_results' / ('relative_speed_' + base['batch'])
    adaptations = freeze(ROOT, analysis, base['batch'], behavior, Path(base['curves_root']), representations,
                         step_penalty=.01, artifact_suffix=SUFFIX)
    # Existing readers use canonical policy basenames. Keep compatibility aliases
    # entirely inside this new analysis copy, pointing to the suffixed checkpoints.
    view = copy.deepcopy(base)
    view['experiment'] = 'V1_matched'
    view['evaluation_seeds'] = list(SEEDS)
    view['new_seeds'] = list(SEEDS)
    view['evaluation_inputs'] = {}
    for version in VERSIONS:
        checkpoint_aliases = analysis / ('inputs' + SUFFIX) / 'checkpoints' / (version + SUFFIX)
        evaluation_aliases = analysis / ('inputs' + SUFFIX) / 'evaluations' / (version + SUFFIX)
        checkpoint_aliases.mkdir(parents=True); evaluation_aliases.mkdir(parents=True)
        for job in view['training_jobs']:
            if job['version'] != version: continue
            canonical = f"{job['condition']}_seed{job['seed']}"
            data = base['versions'][version]
            for aliases, root, extensions in (
                (checkpoint_aliases, data['checkpoint_root'], ('.safetensors', '_config.json', '_sampling_audit.json')),
                (evaluation_aliases, data['evaluation_root'], ('_summary.json', '_train.h5', '_test.h5')),
            ):
                for ext in extensions:
                    target = (Path(root) / (job['tag'] + ext)).resolve()
                    if not target.is_file(): raise FileNotFoundError(target)
                    (aliases / (canonical + ext)).symlink_to(target)
            job['tag'] = canonical
            job['command'] = ['--output=' + job['log_path']]
        view['versions'][version]['checkpoint_root'] = str(checkpoint_aliases)
        view['versions'][version]['evaluation_root'] = str(evaluation_aliases)
        view['evaluation_inputs'][version] = {str(s): str(evaluation_aliases) for s in SEEDS}
    view_path = analysis / 'train/manifests' / ('sbatch_' + base['batch'] + '.json')
    view_path.parent.mkdir(parents=True, exist_ok=True)
    write(view_path, view)
    policies = []
    for job in base['training_jobs']:
        if not job['condition'].startswith('rnn'): continue
        data = base['versions'][job['version']]
        source = Path(base['source_variants'][job['source_variant']]['frozen_source_root'])
        prefix = Path(data['checkpoint_root']) / job['tag']
        config, checkpoint = Path(str(prefix) + '_config.json').resolve(), Path(str(prefix) + '.safetensors').resolve()
        cfg = load(config)
        if cfg['GRID_ENCODER'] != 'flat' or cfg['INCLUDE_IS_T0'] is not True or cfg['RUN_UPDATES'] != 915:
            raise ValueError('Wrong trained observation pipeline')
        rollouts = {p: str((Path(data['evaluation_root']) / (job['tag'] + '_' + p + '.h5')).resolve()) for p in ('train', 'test')}
        policies.append(dict(version=job['version'], condition=job['condition'], seed=job['seed'],
            source_root=str(source), evaluator_sha256=sha(source / 'analysis/evaluate_partner_modelling.py'),
            config=str(config), config_sha256=sha(config), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
            layouts_dir=base['frozen_layouts_dir'], allocation_protocol=data['allocation_protocol'],
            original_rollouts=rollouts, original_stats={p: dict(size=Path(q).stat().st_size,
                mtime_ns=Path(q).stat().st_mtime_ns) for p, q in rollouts.items()}))
    if len(policies) != 45: raise ValueError('Expected all 45 RNN policies')
    inputs_path = work / ('representation_inputs' + SUFFIX + '.json')
    write(inputs_path, dict(status='verified_inputs', policies=policies, training_manifest=str(path)))
    logs = ROOT / 'eval/slurm_logs' / batch
    logs.mkdir(parents=True, exist_ok=True)
    result = dict(batch=batch, status='prepared', prepared_at_utc=now(), artifact_suffix=SUFFIX,
        training_manifest=str(path), training_manifest_sha256=sha(path),
        analysis_root=str(analysis), analysis_adaptations=adaptations, view_manifest=str(view_path),
        view_manifest_sha256=sha(view_path), representation_inputs=str(inputs_path),
        representation_inputs_sha256=sha(inputs_path), behavior_root=str(behavior),
        representation_root=str(representations), pipeline_root=str(behavior / ('pipeline' + SUFFIX)),
        seeds=list(SEEDS), policies=60, rnn_policies=45, jobs={}, log_dir=str(logs), validation=dict(passed=False),
        infrastructure_root=str(infrastructure), infrastructure_sha256={str(p.relative_to(infrastructure)): sha(p)
            for p in infrastructure.rglob('*') if p.is_file()})
    manifest = ROOT / 'train/manifests' / ('sbatch_' + batch + '.json')
    write(manifest, result)
    verify(manifest)
    print(manifest, flush=True)
    return manifest


def verify(path):
    m = load(path)
    for filename, digest in [(m['training_manifest'], m['training_manifest_sha256']),
                             (m['view_manifest'], m['view_manifest_sha256']),
                             (m['representation_inputs'], m['representation_inputs_sha256'])]:
        if sha(filename) != digest: raise ValueError('Input manifest changed: ' + filename)
    for rel, data in m['analysis_adaptations'].items():
        if sha(Path(m['analysis_root']) / rel) != data['frozen_sha256']:
            raise ValueError('Analysis source drift: ' + rel)
    for rel, digest in m['infrastructure_sha256'].items():
        if sha(Path(m['infrastructure_root']) / rel) != digest: raise ValueError('Launcher drift: ' + rel)
    view = load(m['view_manifest'])
    if view['evaluation_seeds'] != list(SEEDS) or len(view['training_jobs']) != 60:
        raise ValueError('Wrong behavioral cohort')
    for job in view['training_jobs']:
        prefix = Path(view['versions'][job['version']]['checkpoint_root']) / job['tag']
        cfg = load(str(prefix) + '_config.json')
        if cfg['SEED'] != job['seed'] or cfg['GRID_ENCODER'] != 'flat' or cfg['INCLUDE_IS_T0'] is not True:
            raise ValueError('Alias points to the wrong policy')
    return m


def environment(m):
    analysis = m['analysis_root']
    return dict(os.environ, GRID_PROJECT_ROOT=analysis, PYTHONPATH=analysis, PYTHONUNBUFFERED='1',
                PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                NUMBA_NUM_THREADS='1', JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', XLA_FLAGS='')


def validate(path):
    m = verify(path)
    original_root = Path(m['training_manifest']).parents[2]
    # The neural probe training, feature extraction, split, label definitions and
    # assignment measurements must remain byte-for-byte identical as ASTs.
    functions = {
        'eval/relative_speed_representation.py': ('relative_classes', 'masks', 'train_probes', 'load_data', 'analyze'),
        'eval/representation_analysis.py': ('make_probe_split', 'load_checkpoint_rollouts'),
        'eval/round_state_prediction_comparison.py': ('round_end_indices', 'prediction_fractions', 'extract_states', 'fit_policy'),
        'eval/analyze_assignment_behaviors.py': ('extract_arrays', 'read_policy', 'round_metrics'),
    }
    preserved = []
    for rel, names in functions.items():
        before = ast.parse((original_root / rel).read_text())
        after = ast.parse((Path(m['analysis_root']) / rel).read_text())
        for name in names:
            left = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == name)
            right = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == name)
            if ast.dump(left) != ast.dump(right): raise ValueError('Scientific method changed: ' + rel + ':' + name)
            preserved.append(rel + ':' + name)
    script = Path(m['infrastructure_root']) / 'diagnostics/validate_t0fix_direct_input_analyses.py'
    subprocess.run([PYTHON, '-u', str(script), str(path)], env=environment(m), check=True)
    m['validation'] = dict(passed=True, validated_at_utc=now(), scientific_functions_preserved=preserved,
                           step_penalty=.01, seeds=list(SEEDS), policies=60, representation_policies=45)
    write(path, m)


def run_stage(path, stage):
    m = verify(path); analysis = Path(m['analysis_root'])
    output = Path(m['pipeline_root']); output.mkdir(parents=True, exist_ok=True)
    status_path = output / (stage + '_status' + SUFFIX + '.json')
    status = load(status_path) if status_path.exists() else dict(status='running', completed_steps=[])
    def step(name, rel, *args):
        if name in status['completed_steps']: return
        status.update(status='running', current_step=name); write(status_path, status)
        print('[stage]', stage, name, flush=True)
        with (output / (name + SUFFIX + '.log')).open('w') as log:
            subprocess.run([PYTHON, '-u', str(analysis / rel), *map(str, args)], cwd=analysis,
                           env=environment(m), stdout=log, stderr=subprocess.STDOUT, check=True)
        status['completed_steps'].append(name); write(status_path, status)
    try:
        if stage == 'behavior':
            view = m['view_manifest']
            returns = Path(m['behavior_root']) / 'overcooked_paper_match'
            selection = returns / 'best_of_5/selection.json'
            step('best_seed_return_selection', 'eval/plot_overcooked_paper_match.py', '--manifest', view, '--out-dir', returns)
            existing = Path(load(m['training_manifest'])['comparison_root']) / ('dot_plots' + SUFFIX) / ('best_seed_selection' + SUFFIX + '.csv')
            if existing.is_file():
                import csv
                prior = {(r['version'], r['condition']): int(r['selected_seed']) for r in csv.DictReader(existing.open())
                         if r['cohort'] == 't0fix_direct_input'}
                current = {(r['version'], r['condition']): r['seed'] for r in load(selection)['results']}
                if current != prior: raise ValueError('Best-seed selection differs from the comparison plots')
            step('all_seed_assignment_metrics', 'eval/analyze_assignment_behaviors.py', '--manifest', view,
                 '--out-dir', Path(m['behavior_root']) / ('assignment_behaviors_all_seeds' + SUFFIX), '--populations', 'test', 'train')
            selected_out = Path(m['behavior_root']) / ('assignment_behaviors_best_training_seeds' + SUFFIX) / 'novel'
            step('best_seed_assignment_metrics', 'eval/plot_best_seed_assignment_behaviors.py', '--manifest', view,
                 '--selection', selection, '--out-dir', selected_out)
            step('best_seed_three_conditions', 'eval/plot_best_seed_assignment_behaviors.py', '--manifest', view,
                 '--selection', selection, '--exclude-no-influence', '--out-dir', selected_out / 'three_conditions')
            step('selected_three_metric_dashboard', 'eval/plot_selected_assignment_dashboard.py', '--manifest', view,
                 '--selection', selection, '--out-dir', selected_out)
        elif stage == 'representation':
            root = Path(m['representation_root'])
            if not (root / 'analysis_run.json').exists():
                step('prepare_representation', 'eval/run_relative_speed_representation.py', 'prepare', root,
                     '--input-manifest', m['representation_inputs'])
            cpus = sorted(os.sched_getaffinity(0))[:3]
            step('representation_and_independent_audits', 'eval/run_relative_speed_pipeline.py', root, '--cpus', *cpus)
        else: raise ValueError(stage)
        status.update(status='completed', completed_at_utc=now()); write(status_path, status)
    except BaseException as error:
        status.update(status='failed', error=repr(error)); write(status_path, status); raise


def finish(path):
    m = verify(path)
    for stage in ('behavior', 'representation'):
        status = load(Path(m['pipeline_root']) / (stage + '_status' + SUFFIX + '.json'))
        if status['status'] != 'completed': raise ValueError('Incomplete stage: ' + stage)
    root = Path(m['representation_root'])
    status = load(root / 'pipeline_status.json')
    if status['status'] != 'completed' or status['policies'] != 45: raise ValueError('Representation coverage')
    for audit in (root / 'independent_validation.json', root / 'prediction_fractions_state_comparison/independent_validation.json'):
        if load(audit)['status'] != 'passed': raise ValueError('Independent analysis audit failed')
    m.update(status='completed', completed_at_utc=now()); write(path, m)
    print('Completed all-seed and selected-seed behaviors, 45 representation analyses and both independent audits.', flush=True)


def submit(path):
    path = Path(path).resolve(); m = verify(path)
    if not m['validation']['passed']: raise ValueError('Analysis validation required')
    clean = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_') and
             k not in ('JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES', 'XLA_FLAGS')}
    for stage, hours in (('behavior', 4), ('representation', 8), ('finish', 1)):
        if stage in m['jobs']: continue
        cmd = ['sbatch', '--parsable', '--account=nlp', '--partition=sphinx', '--cpus-per-task=4', '--mem=32G',
               '--exclude=sphinx9', f'--time={hours}:00:00', '--job-name=' + stage + SUFFIX,
               '--chdir=' + str(ROOT), '--export=ALL,GRID_PROJECT_ROOT=' + str(ROOT), '--kill-on-invalid-dep=yes',
               '--output=' + str(Path(m['log_dir']) / (stage + '_%j' + SUFFIX + '.out')),
               '--error=' + str(Path(m['log_dir']) / (stage + '_%j' + SUFFIX + '.err'))]
        if stage == 'finish':
            cmd.append('--dependency=afterok:' + ':'.join(m['jobs'][s]['job_id'] for s in ('behavior', 'representation')))
        cmd += [str(Path(m['infrastructure_root']) / 'bash/run_counterbalanced_seed_extension.sh'),
                str(Path(m['infrastructure_root']) / 'bash/run_t0fix_direct_input_analyses.py'), stage, str(path)]
        job_id = subprocess.check_output(cmd, cwd=ROOT, env=clean, text=True).strip().split(';')[0]
        if not job_id.isdigit(): raise RuntimeError(job_id)
        m['jobs'][stage] = dict(job_id=job_id, command=cmd); write(path, m)
        print(stage, job_id, flush=True)
    m.update(status='submitted', submitted_at_utc=now()); write(path, m)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'verify', 'validate', 'submit', 'behavior', 'representation', 'finish'))
    p.add_argument('manifest', type=Path)
    a = p.parse_args()
    if a.mode in ('behavior', 'representation'): run_stage(a.manifest, a.mode)
    else: globals()[a.mode](a.manifest)


if __name__ == '__main__': main()
