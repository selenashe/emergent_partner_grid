"""Isolated five-seed, four-condition direct-input replications of three protocols.

Prepare from the retained direct-input scientific configs and protocol sources.
All scientific settings are verified against the independently sealed contract.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml

ROOT = Path(os.environ.get('GRID_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
INFRA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INFRA))
from train import experiment_runtime as shared
from train.experiment_runtime import evaluate_job, publish
from train.experiment import CONDITIONS, TRAINING_MANIFEST
from repo_paths import artifact_path, resolve_path, resolve_record_paths
from train.observation_pipeline import PIPELINE_VERSION, TRAINER_FILE, upgrade_trainer

SUFFIX = '_t0fix_direct_input'
VERSIONS = ('v1_matched', 'v2', 'v3')
SEEDS = tuple(range(1, 6))
SETTINGS = dict(GRID_ENCODER='flat', INCLUDE_IS_T0=True)
sha, write, PYTHON = shared.sha, shared.write, shared.PYTHON


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def record(path):
    return resolve_record_paths(load(path))


def expected_config(baseline, output):
    config = copy.deepcopy(baseline)
    config.update(SETTINGS)
    config['SAVE_PARAMS_PATH'] = str(output)
    return config


def prepare():
    from train.prepare_experiment import prepare as prepare_current
    return prepare_current(ROOT, TRAINING_MANIFEST)


def verify(path):
    # The independently sealed contract preserves the original, already-passed
    # source transformation and scientific config checks without retired files.
    from train.cohort_integrity import verify as verify_current
    return verify_current(path)


def training_command(m, job, smoke=None):
    source = Path(m['source_variants'][job['source_variant']]['frozen_source_root'])
    cfg = Path(job['config_path'])
    target = artifact_path(ROOT / 'train/hydra_outputs' / m['batch'] / (job['version'] + SUFFIX) / job['tag'])
    cmd = [PYTHON, '-u', str(source / TRAINER_FILE), '--config-path', str(cfg.parent), '--config-name', cfg.stem,
           f'hydra.run.dir={target}']
    if smoke:
        target = ROOT / 'train/run_snapshots' / m['batch'] / (smoke + '_preflight' + SUFFIX) / (job['version'] + SUFFIX) / job['tag']
        target.mkdir(parents=True, exist_ok=True)
        cmd += ['+PILOT_UPDATES=1', 'EVAL_EPISODES_PER_CAPABILITY=1',
                f'SAVE_PARAMS_PATH={target / ("params" + SUFFIX + ".safetensors")}',
                'STDOUT_SUMMARY=false', f'hydra.run.dir={target / "hydra"}']
        if smoke == 'cpu':
            cmd += ['NUM_ENVS=4', 'NUM_STEPS=64', 'TOTAL_TIMESTEPS=234240', 'UPDATE_EPOCHS=1',
                    'NUM_MINIBATCHES=4', 'ENV_KWARGS.max_steps=3']
    return cmd, source, target


def check_saved(m, job, smoke=None):
    _, _, target = training_command(m, job, smoke)
    prefix = target / ('params' + SUFFIX) if smoke else Path(m['versions'][job['version']]['checkpoint_root']) / job['tag']
    cfg = record(resolve_path(str(prefix) + '_config.json'))
    audit_path = resolve_path(str(prefix) + '_sampling_audit.json')
    audit = load(audit_path)
    steps = 256 if smoke == 'cpu' else 65536 if smoke else 59965440
    if (cfg['GRID_ENCODER'] != 'flat' or cfg['INCLUDE_IS_T0'] is not True or
            cfg['RUN_UPDATES'] != (1 if smoke else 915) or cfg['NUM_UPDATES'] != 915 or
            audit['collected_environment_steps'] != steps):
        raise ValueError('Saved input settings or training audit differs')
    from safetensors.numpy import load_file
    import numpy as np
    weights = load_file(str(resolve_path(str(prefix) + '.safetensors')))
    if not weights or not all(np.isfinite(x).all() for x in weights.values()):
        raise ValueError('Invalid checkpoint')
    return prefix


def run_training(path, index, smoke=None):
    m = verify(path); job = m['training_jobs'][index]
    cmd, source, _ = training_command(m, job, smoke)
    env = shared.environment(source, cpu=smoke == 'cpu')
    if smoke == 'cpu':
        cmd = ['taskset', '-c', ','.join(map(str, sorted(os.sched_getaffinity(0))[:4]))] + cmd
    subprocess.run(cmd, cwd=source, env=env, check=True)
    check_saved(m, job, smoke)
    if not smoke:
        publish(m['versions'][job['version']]['checkpoint_root'], job['tag'])


def validate(path):
    m = verify(path)
    for index, job in enumerate(m['training_jobs']):
        if job['seed'] != 1 or any(x['version'] == job['version'] and x['condition'] == job['condition']
                                   for x in m['preflight']['cpu_checks']):
            continue
        _, _, target = training_command(m, job, 'cpu')
        log = target / ('validation' + SUFFIX + '.log')
        print('[CPU preflight]', job['version'], job['condition'], flush=True)
        runner = Path(m['infrastructure_root']) / 'bash/submit_t0fix_direct_input_training.py'
        env = dict(os.environ, GRID_PROJECT_ROOT=str(ROOT))
        with log.open('w') as stream:
            result = subprocess.run([PYTHON, '-u', str(runner), 'cpu-smoke', str(path), str(index)],
                                    env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(str(log) + '\n' + log.read_text()[-10000:])
        prefix = check_saved(m, job, 'cpu')
        # The evaluator expects the normal params basename; supply an isolated
        # manifest view pointing directly to the smoke checkpoint's real name.
        view = copy.deepcopy(m)
        view['versions'][job['version']]['checkpoint_root'] = str(prefix.parent)
        smoke_job = dict(job, tag=prefix.name)
        with (target / ('standalone_evaluation' + SUFFIX + '.log')).open('w') as stream:
            from contextlib import redirect_stdout, redirect_stderr
            # Invoke evaluation in another Python process to preserve snapshot imports.
            with redirect_stdout(stream), redirect_stderr(stream):
                evaluate_smoke(view, smoke_job, prefix, stream)
        m['preflight']['cpu_checks'].append(dict(version=job['version'], condition=job['condition'], passed=True,
                                                output=str(target), log=str(log)))
        write(path, m)
        print('[passed]', job['version'], job['condition'], flush=True)
    m['preflight'].update(passed=True, completed_at_utc=now())
    write(path, m)


def evaluate_smoke(m, job, prefix, stream):
    source = Path(m['source_variants'][job['source_variant']]['frozen_source_root'])
    output = prefix.parent / ('standalone_eval' + SUFFIX)
    cmd = [PYTHON, '-u', str(source / 'analysis/evaluate_partner_modelling.py'), '--config', str(prefix) + '_config.json',
           '--params', str(prefix) + '.safetensors', '--layouts_dir', m['frozen_layouts_dir'],
           '--n_episodes_per_capability', '1', '--seed', '12345', '--out_prefix', str(output)]
    cores = ','.join(map(str, sorted(os.sched_getaffinity(0))[:4]))
    subprocess.run(['taskset', '-c', cores] + cmd, cwd=source, env=shared.environment(source, cpu=True),
                   stdout=stream, stderr=subprocess.STDOUT, check=True)
    for ext in ('_summary.json', '_train.h5', '_test.h5'):
        if not Path(str(output) + ext).is_file():
            raise FileNotFoundError(str(output) + ext)


def submit(path):
    path = Path(path).resolve(); m = verify(path)
    if not m['preflight']['passed']:
        raise ValueError('CPU preflight must pass before submission')
    runner = Path(m['infrastructure_root']) / 'bash/submit_t0fix_direct_input_training.py'
    wrapper = Path(m['infrastructure_root']) / 'bash/run_counterbalanced_seed_extension.sh'
    clean = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_') and
             k not in ('JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES', 'XLA_FLAGS')}
    def launch(stage, indices=None, limit=None, dependency=None, gpu=True, hours=8):
        if stage in m['jobs']:
            return m['jobs'][stage]['job_id']
        cmd = ['sbatch', '--parsable', '--account=nlp', '--partition=sphinx', '--mem=32G', '--cpus-per-task=4',
               '--exclude=sphinx9', f'--time={hours}:00:00', '--job-name=' + stage + SUFFIX,
               '--chdir=' + str(ROOT), '--export=ALL,GRID_PROJECT_ROOT=' + str(ROOT),
               '--output=' + str(Path(m['log_dir']) / (stage + '_%A_%a' + SUFFIX + '.out')),
               '--error=' + str(Path(m['log_dir']) / (stage + '_%A_%a' + SUFFIX + '.err')),
               '--kill-on-invalid-dep=yes']
        if gpu: cmd += ['--constraint=80G', '--gres=gpu:1']
        if indices is not None: cmd.append('--array=' + indices + '%' + str(limit))
        if dependency: cmd.append('--dependency=' + dependency)
        cmd += [str(wrapper), str(runner), stage, str(path)]
        job_id = subprocess.check_output(cmd, cwd=ROOT, env=clean, text=True).strip().split(';')[0]
        if not job_id.isdigit(): raise RuntimeError(job_id)
        m['jobs'][stage] = dict(job_id=job_id, command=cmd)
        write(path, m)
        print(stage, job_id, flush=True)
        return job_id
    m['status'] = 'submitting'; write(path, m)
    smoke_indices = ','.join(str(i) for i, j in enumerate(m['training_jobs']) if j['seed'] == 1)
    smoke = launch('gpu-smoke', smoke_indices, 4, hours=1)
    train = launch('train', '0-59', 12, 'afterok:' + smoke)
    evaluation = launch('eval', '0-59', 6, 'aftercorr:' + train, hours=2)
    launch('report', dependency='afterok:' + evaluation, gpu=False, hours=2)
    for i, j in enumerate(m['training_jobs']):
        j['job_id'] = train + '_' + str(i)
        j['log_path'] = str(Path(m['log_dir']) / ('train_' + j['job_id'] + SUFFIX + '.out'))
    m.update(status='submitted', submitted_at_utc=now())
    write(path, m)


def report(path):
    """Export the five-seed direct-input evaluations and training histories."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from train.training_log_format import FIELDS, PATTERN
    m = verify(path)
    output, curves = Path(m['comparison_root']), Path(m['curves_root'])
    output.mkdir(parents=True, exist_ok=True); curves.mkdir(parents=True, exist_ok=True)
    rows, histories, provenance = [], [], []
    for j in m['training_jobs']:
        check_saved(m, j)
        new_prefix = Path(m['versions'][j['version']]['evaluation_root']) / j['tag']
        for cohort, prefix, log in [('t0fix_direct_input', new_prefix, Path(j['log_path']))]:
            summary_path = resolve_path(str(prefix) + '_summary.json')
            summary = load(summary_path)
            for population, counts in [('train', 480), ('test', 440)]:
                s = summary[population]
                if s['n_episodes'] != counts or s['n_rounds'] != counts * 20:
                    raise ValueError('Incomplete full evaluation: ' + str(summary_path))
                rows.append(dict(version=j['version'], condition=j['condition'], seed=j['seed'],
                    cohort=cohort, population=population, mean_ep_return=s['mean_ep_return'],
                    round_success_rate=s['round_success_rate']))
            content = resolve_path(log).read_text()
            matches = [PATTERN.fullmatch(line) for line in content.splitlines() if line.startswith('[u ')]
            if len(matches) != 915 or any(x is None for x in matches):
                raise ValueError('Incomplete training curve: ' + str(log))
            records = [dict(zip(FIELDS, [float(x) if x is not None else np.nan for x in match.groups()])) for match in matches]
            if [int(r['update']) for r in records] != list(range(1, 916)):
                raise ValueError('Unordered training curve')
            for r in records:
                if r['env_steps'] != r['update'] * 65536:
                    raise ValueError('Training step mismatch')
                if r['episode_steps'] <= 0: r['episode_return'] = np.nan
                histories.append(dict(version=j['version'], condition=j['condition'], seed=j['seed'], cohort=cohort, **r))
            provenance.append(dict(version=j['version'], condition=j['condition'], seed=j['seed'], cohort=cohort,
                summary=str(summary_path), summary_sha256=sha(summary_path), log=str(log), log_sha256=sha(resolve_path(log))))
    def save_csv(p, data):
        with p.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    save_csv(output / ('evaluation_per_seed' + SUFFIX + '.csv'), rows)
    save_csv(curves / ('updates' + SUFFIX + '.csv'), histories)
    lines = ['# Five-seed direct-input training and evaluation', '',
        'Learner seeds 1–5; 59,965,440 transitions per policy; 20 evaluation episodes per capability.', '',
        '| Experiment | Condition | Population | Return | Round success |',
        '|---|---|---|---:|---:|']
    for v in VERSIONS:
        for c in CONDITIONS:
            for p in ('train', 'test'):
                group = [r for r in rows if (r['version'], r['condition'], r['population']) == (v, c, p)]
                mean = np.mean([r['mean_ep_return'] for r in group])
                success = np.mean([r['round_success_rate'] for r in group])
                lines.append(f'| {v} | {c} | {p} | {mean:.3f} | {success:.2%} |')
    for metric in ('episode_return', 'round_success'):
        fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharex=True, sharey=True)
        for row, c in enumerate(CONDITIONS):
            for col, v in enumerate(VERSIONS):
                ax = axes[row, col]
                for cohort, color, label in [('t0fix_direct_input', '#009E73', 't0 fix + direct input')]:
                    ys = []
                    for seed in SEEDS:
                        data = [r for r in histories if (r['version'], r['condition'], r['seed'], r['cohort']) == (v, c, seed, cohort)]
                        vals = np.array([r[metric] for r in data])
                        valid = np.isfinite(vals)
                        sums = np.r_[0, np.cumsum(np.where(valid, vals, 0))]
                        n = np.r_[0, np.cumsum(valid)]
                        stop = np.arange(1, 916); start = np.maximum(0, stop - 16)
                        smooth = np.divide(sums[stop]-sums[start], n[stop]-n[start], out=np.full(915, np.nan), where=(n[stop]-n[start]) > 0)
                        ys.append(smooth)
                        ax.plot(stop * 65536 / 1e6, smooth, color=color, alpha=.15, lw=.6)
                    ax.plot(stop * 65536 / 1e6, np.nanmean(ys, axis=0), color=color, label=label, lw=2)
                ax.set_title(v + ' · ' + c.replace('_', ' '), fontsize=9)
                ax.grid(alpha=.2)
                if row == 3: ax.set_xlabel('Training steps (millions)')
                if col == 0: ax.set_ylabel(metric.replace('_', ' '))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2)
        fig.suptitle('Seeds 1–5 · faint: individual seeds · bold: seed mean · trailing 16 updates', y=.97)
        fig.tight_layout(rect=(0, 0, 1, .95))
        for ext in ('png', 'pdf'): fig.savefig(curves / (metric + SUFFIX + '.' + ext), dpi=160)
        plt.close(fig)
    (output / ('comparison' + SUFFIX + '.md')).write_text('\n'.join(lines) + '\n')
    write(output / ('provenance' + SUFFIX + '.json'), dict(inputs=provenance, seeds=list(SEEDS)))
    m.update(status='completed', completed_at_utc=now())
    write(path, m)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'verify', 'validate', 'submit', 'cpu-smoke', 'gpu-smoke', 'train', 'eval', 'report'])
    p.add_argument('manifest', nargs='?', type=Path)
    p.add_argument('index', nargs='?', type=int)
    a = p.parse_args()
    if a.mode == 'prepare': prepare(); return
    if a.mode in ('verify', 'validate', 'submit', 'report'): globals()[a.mode](a.manifest); return
    index = a.index if a.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    if a.mode == 'eval':
        m = verify(a.manifest); j = m['training_jobs'][index]
        check_saved(m, j); evaluate_job(m, j)
    else:
        run_training(a.manifest, index, {'cpu-smoke': 'cpu', 'gpu-smoke': 'gpu'}.get(a.mode))


if __name__ == '__main__':
    main()
