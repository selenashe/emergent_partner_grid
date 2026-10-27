"""Regenerate dependent reports and compare random versus archived parity goals."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(os.environ.get('GRID_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from repo_paths import artifact_path

CONDITION = 'rnn_diverse_noinfluence'
VERSION = 'v1_matched'
CANONICAL = ROOT / 'train/manifests/sbatch_v1_matched_20261009_182845.json'
REPRESENTATION = ROOT / '/juice6/u/jshe/emergent_partner_grid/eval/representation_results/relative_speed_v1_matched_v2_v3_20261019_191029_step_penalty_0.03'
DOWNSTREAM_SOURCES = (
    'eval/compare_v1_matched.py', 'eval/compare_allocation_protocols.py', 'eval/plot_episode_return_dots.py',
    'eval/plot_profile_performance.py', 'eval/plot_round_return_progression.py', 'eval/plot_overcooked_paper_match.py',
    'eval/analyze_assignment_behaviors.py', 'eval/plot_best_seed_assignment_behaviors.py',
    'eval/plot_training_return_ceiling.py', 'eval/optimal_training_return.py',
    'eval/run_relative_speed_representation.py', 'eval/run_relative_speed_pipeline.py',
    'eval/relative_speed_representation.py', 'eval/representation_analysis.py',
    'eval/round_state_prediction_comparison.py', 'eval/export_round_state_comparison_pngs.py',
    'train/plot_protocol_training_curves.py', 'train/training_log_format.py',
    'train/plot_final_training_return_dots.py', 'train/plot_per_seed_training_returns.py',
    'train/plot_selected_training_returns.py', 'diagnostics/validate_optimal_training_return.py',
    'diagnostics/diagnose_seed_variability.py', 'diagnostics/verify_relative_speed_results.py',
    'diagnostics/verify_round_state_comparison.py', 'repo_paths.py',
    'jaxmarl/environments/coordination_grid/capability_populations.py',
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(stage, manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = read(manifest_path)
    for relative, digest in manifest.get('downstream_source_sha256', {}).items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Downstream source changed since submission: ' + relative)
    output = ROOT / 'train/run_snapshots' / manifest['batch'] / 'downstream'
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / (stage + '_status.json')
    status = read(status_path) if status_path.exists() else dict(status='running', completed_steps=[])
    status['status'] = 'running'
    write(status_path, status)
    env = dict(os.environ, GRID_PROJECT_ROOT=str(ROOT), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMBA_NUM_THREADS='1', JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', XLA_FLAGS='')
    env['PYTHONPATH'] = str(ROOT)
    python = manifest.get('analysis_python', '/juice6/u/nlp/climblab/tmp/epg_repr_all10_20261008/bin/python')

    def step(name, script, *arguments, interpreter=None):
        if name in status['completed_steps']:
            return
        status['current_step'] = name
        write(status_path, status)
        print('[downstream]', name, flush=True)
        command = [interpreter or python, '-u', str(ROOT / script), *map(str, arguments)]
        with (output / (name + '.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        status['completed_steps'].append(name)
        write(status_path, status)

    try:
        if stage == 'behavior':
            audit_random_assignments(manifest, output)
            for population in ('train', 'test'):
                step('compare_' + population, 'eval/compare_v1_matched.py', CANONICAL, '--population', population)
            step('profile_performance', 'eval/plot_profile_performance.py', '--extension-manifest', CANONICAL)
            step('round_returns', 'eval/plot_round_return_progression.py', '--manifest', CANONICAL)
            step('training_curves', 'train/plot_protocol_training_curves.py')
            step('final_training_dots', 'train/plot_final_training_return_dots.py')
            step('per_seed_training_returns', 'train/plot_per_seed_training_returns.py')
            step('best_seed_returns', 'eval/plot_overcooked_paper_match.py', '--manifest', CANONICAL)
            comparison_root = Path(manifest['aggregate_root']).parent
            selection = comparison_root / 'overcooked_paper_match/best_of_5/selection.json'
            step('selected_training_returns', 'train/plot_selected_training_returns.py', '--selection', selection)
            step('assignment_behaviors', 'eval/analyze_assignment_behaviors.py', '--manifest', CANONICAL)
            step('selected_assignment_behaviors', 'eval/plot_best_seed_assignment_behaviors.py',
                 '--manifest', CANONICAL, '--selection', selection)
            step('selected_assignment_three_conditions', 'eval/plot_best_seed_assignment_behaviors.py',
                 '--manifest', CANONICAL, '--selection', selection, '--exclude-no-influence')
            # These are policy-independent exact corpus solutions. Uniform parity
            # and a fair random draw have the same mean over complete episodes.
            # Reuse that solution and rerun environment validation and observations.
            reuse = []
            for directory, population in [('training_return_ceiling', 'train'), ('heldout_return_ceiling', 'test')]:
                active = ROOT / 'eval' / directory / 'v1_matched_v2_v3'
                archived = active.with_name(active.name + '_archived')
                active.mkdir(parents=True, exist_ok=True)
                for name in ('summary.json', 'per_layout_profile.csv', 'ceiling_summary.csv', 'initial_information_bounds.csv'):
                    shutil.copy2(archived / name, active / name)
                step('validate_ceiling_' + population, 'diagnostics/validate_optimal_training_return.py',
                     '--population', population)
                step('plot_ceiling_' + population, 'eval/plot_training_return_ceiling.py', '--population', population)
                reuse.append(dict(population=population, archived_solution=str(archived),
                                  reason='Unchanged geometry, delays, fixed-goal transitions, and 50/50 assignment expectation; '
                                         'environment validation and observed-policy plots were rerun.'))
            write(output / 'oracle_solution_reuse.json', reuse)
        elif stage == 'representation':
            from eval.run_relative_speed_representation import prepare
            inputs_path = ROOT / 'eval/relative_speed_inputs.json'
            if not inputs_path.exists():
                inputs = read(ROOT / 'eval/relative_speed_inputs_archived.json')
                data = manifest['versions'][VERSION]
                for record in inputs['policies']:
                    if record['version'] != VERSION or record['condition'] != CONDITION:
                        continue
                    tag = CONDITION + '_seed' + str(record['seed'])
                    record['source_root'] = data['frozen_source_root']
                    record['evaluator_sha256'] = sha(Path(data['frozen_source_root']) / 'analysis/evaluate_partner_modelling.py')
                    record['checkpoint'] = str(Path(data['checkpoint_root']) / (tag + '.safetensors'))
                    record['config'] = str(Path(data['checkpoint_root']) / (tag + '_config.json'))
                    for key in ('checkpoint', 'config'):
                        record[key + '_sha256'] = sha(record[key])
                    for population in ('train', 'test'):
                        path = artifact_path(data['evaluation_root']) / (tag + '_' + population + '.h5')
                        record['original_rollouts'][population] = str(path)
                        stat = path.stat()
                        record['original_stats'][population] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
                inputs.update(status='verified_inputs', replacement_run_manifest=str(manifest_path))
                write(inputs_path, inputs)
            if not (REPRESENTATION / 'analysis_run.json').exists():
                prepare(REPRESENTATION, inputs_path)
            cpus = sorted(os.sched_getaffinity(0))[:3]
            if len(cpus) != 3:
                raise ValueError('Representation analysis needs three CPUs')
            step('representations_and_independent_audits', 'eval/run_relative_speed_pipeline.py',
                 REPRESENTATION, '--cpus', *cpus)
        elif stage == 'finish':
            for dependency in ('behavior', 'representation'):
                if read(output / (dependency + '_status.json'))['status'] != 'completed':
                    raise ValueError('Incomplete downstream dependency: ' + dependency)
            step('seed_diagnostics', 'diagnostics/diagnose_seed_variability.py', '--rollouts')
            comparison(manifest, output)
            for path in (manifest_path, CANONICAL):
                latest = read(path)
                latest.update(status='completed', completed_at_utc=datetime.now(timezone.utc).isoformat(),
                              random_vs_archived_comparison=str(output / 'random_vs_archived.json'))
                write(path, latest)
        else:
            raise ValueError(stage)
        status.update(status='completed', completed_at_utc=datetime.now(timezone.utc).isoformat())
        write(status_path, status)
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        write(status_path, status)
        raise


def comparison(manifest, output):
    """Compare identical seed identities; retain all old files for human review."""
    data = manifest['versions'][VERSION]
    rows = []
    for seed in range(1, 6):
        directory = Path(data['evaluation_root'])
        active = read(directory / (CONDITION + '_seed' + str(seed) + '_summary.json'))
        archived = read(directory / (CONDITION + '_archived_seed' + str(seed) + '_summary.json'))
        for population in ('train', 'test'):
            for metric in ('mean_ep_return', 'round_success_rate', 'mean_successful_completion_time'):
                a, b = active[population][metric], archived[population][metric]
                rows.append(dict(seed=seed, population=population, metric=metric,
                                 random=float(a), archived_alternating=float(b), difference=float(a-b)))
    aggregates = []
    rng = np.random.default_rng(20261019)
    for population in ('train', 'test'):
        for metric in ('mean_ep_return', 'round_success_rate', 'mean_successful_completion_time'):
            group = [r for r in rows if r['population'] == population and r['metric'] == metric]
            differences = np.array([r['difference'] for r in group])
            bootstrap = differences[rng.integers(0, 5, size=(10000, 5))].mean(axis=1)
            aggregates.append(dict(population=population, metric=metric,
                random_mean=float(np.mean([r['random'] for r in group])),
                archived_alternating_mean=float(np.mean([r['archived_alternating'] for r in group])),
                mean_difference=float(differences.mean()),
                difference_ci95=np.quantile(bootstrap, [.025, .975]).tolist()))
    probe_comparison = []
    for root, label in [(REPRESENTATION, 'random'),
                        (REPRESENTATION.with_name(REPRESENTATION.name + '_archived'), 'archived_alternating')]:
        with (root / 'probe_per_seed.csv').open() as stream:
            for row in csv.DictReader(stream):
                if (row['version'] == VERSION and row['condition'] == CONDITION and row['axis'] == 'round'
                        and row['cutoff'] == '20' and row['kind'] == 'primary' and row['population'] == 'all'):
                    probe_comparison.append(dict(variant=label, seed=int(row['training_seed']),
                                                 balanced_accuracy=float(row['balanced_accuracy'])))
    if len(probe_comparison) != 20:
        raise ValueError('Missing random/archived round-20 probe results')
    write(output / 'random_vs_archived.json', dict(status='ready_for_review', n_seeds=5,
        training_manifest=str(manifest['canonical_manifest']), archive_manifest=manifest['archive_plan'],
        behavior=aggregates, per_seed=rows, round20_probes=probe_comparison,
        uncertainty='Percentile bootstrap of ten paired learner-seed differences; 10000 resamples.',
        archive_deleted=False, deletion_requires='Explicit user approval after reviewing both versions.'))
    with (output / 'random_vs_archived_per_seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, population in zip(axes, ('train', 'test')):
        group = [r for r in rows if r['population'] == population and r['metric'] == 'mean_ep_return']
        for row in group:
            axis.plot([0, 1], [row['archived_alternating'], row['random']], color='#999999', alpha=.5)
        axis.scatter(np.zeros(10), [r['archived_alternating'] for r in group], label='Archived alternating')
        axis.scatter(np.ones(10), [r['random'] for r in group], label='Random')
        axis.set_xticks([0, 1], ['Alternating (archived)', 'Random'])
        axis.set_title('Familiar partners' if population == 'train' else 'Novel partners')
        axis.set_ylabel('Mean episode return')
    fig.tight_layout()
    fig.savefig(output / 'random_vs_archived.png', dpi=180)
    plt.close(fig)


def audit_random_assignments(manifest, output):
    """Check every completed evaluation round, excluding post-done padding."""
    import h5py
    checks = []
    directory = Path(manifest['versions'][VERSION]['evaluation_root'])
    for seed in range(1, 6):
        for population, expected in [('train', 480), ('test', 440)]:
            path = directory / (CONDITION + '_seed' + str(seed) + '_' + population + '.h5')
            with h5py.File(path) as file:
                dones = np.asarray(file['dones'], dtype=bool)
                initialized = np.asarray(file['is_t0'], dtype=bool)
                assignments = np.asarray(file['partner_assignment'])
                round_idx = np.asarray(file['round_idx'])
                goals = np.asarray(file['partner_goal'])
            if len(dones) != expected or not dones.any(axis=1).all():
                raise ValueError('Incomplete evaluation episodes: ' + str(path))
            initial = []
            for episode in range(expected):
                length = int(dones[episode].argmax()) + 1
                starts = np.flatnonzero(initialized[episode, :length])
                if len(starts) != 20:
                    raise ValueError('Expected twenty round initializations')
                chosen = assignments[episode, starts]
                if not np.isin(chosen, [1, 2]).all():
                    raise ValueError('Invalid initial assignment')
                initial.append(chosen)
                for index, start in enumerate(starts):
                    stop = int(starts[index+1]) if index+1 < 20 else length
                    if not np.all(round_idx[episode, start:stop] == index):
                        raise ValueError('Inconsistent round indices')
                    if not np.all(assignments[episode, start:stop] == chosen[index]):
                        raise ValueError('No-influence goal changed within a round')
                    if goals[episode, start] != 0 or not np.all(goals[episode, start+1:stop] == 3-chosen[index]):
                        raise ValueError('V1_matched goal commitment changed')
            initial = np.stack(initial)
            red_fraction = float(np.mean(initial == 1))
            if not .45 < red_fraction < .55 or not np.any(initial[:, 1:] == initial[:, :-1]):
                raise ValueError('Assignments do not match fresh fair random draws')
            checks.append(dict(seed=seed, population=population, episodes=expected,
                               rounds=expected*20, allocation_red_fraction=red_fraction,
                               repeated_adjacent_fraction=float(np.mean(initial[:, 1:] == initial[:, :-1]))))
    write(output / 'random_assignment_audit.json', dict(passed=True, checks=checks,
          frozen_source_sha256=manifest['versions'][VERSION]['source_sha256'][
              'jaxmarl/environments/coordination_grid/coordination_grid.py']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('behavior', 'representation', 'finish'))
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    run(args.stage, args.manifest)
