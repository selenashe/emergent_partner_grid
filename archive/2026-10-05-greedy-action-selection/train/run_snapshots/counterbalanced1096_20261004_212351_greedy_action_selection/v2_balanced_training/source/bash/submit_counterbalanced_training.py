"""Prepare/test frozen v1/v2 sources, then submit 40 paired balanced runs.

Preparation never submits jobs. Submission resumes from its durable manifest.
The v1 source is recovered from the original experiment's Git revision;
both versions receive the same sampler/action-selection patch and 1096-layout corpus.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))
sys.path.insert(0, str(ROOT))
from counterbalanced_scheduler import build_schedule, save_schedule, SAMPLING_PROTOCOL
from repo_paths import resolve_record_paths

V1_REVISION = '541452613cdc531c98dfc5e0b8f00efe34fae3de'
CONDITIONS = ('rnn_diverse_influence', 'mlp_diverse_influence',
              'rnn_single_influence', 'rnn_diverse_noinfluence')
V1_FILES = {'jaxmarl/environments/coordination_grid/coordination_grid.py': 'jaxmarl/environments/coordination_grid/coordination_grid.py',
            'jaxmarl/environments/coordination_grid/__init__.py': 'jaxmarl/environments/coordination_grid/__init__.py',
            'baselines/IPPO/ippo_rnn_coordination_grid.py': 'train/ippo_rnn_coordination_grid.py',
            'baselines/IPPO/config/ippo_coordination_grid.yaml': 'train/config/ippo_coordination_grid.yaml',
            'analysis/evaluate_partner_modelling.py': 'eval/evaluate_partner_modelling.py'}


def relocate_v1_imports(text):
    """Relocate imports only; retain the original v1 protocol and PPO functions."""
    return (text.replace('from sweep_scheduler import', 'from episode_scheduler import')
            .replace('_REPO_ROOT / "baselines" / "IPPO"', '_REPO_ROOT / "train"')
            .replace('_REPO_ROOT / "dev"', '_REPO_ROOT / "data_prep"'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Expected exactly one patch anchor: ' + old[:80])
    return text.replace(old, new)


def patch_action_selection(text):
    """Apply the requested collector/evaluator change to the recovered v1 trainer."""
    if 'from action_selection import select_action' not in text:
        anchor = 'from episode_scheduler import build_schedule, initial_episode_cursor, summarize_schedule'
        text = replace_once(text, anchor, anchor + '\nfrom action_selection import select_action')
    matches = re.findall(r'pi\.sample\(seed=([a-z_]+)\)', text)
    if matches:
        if matches != ['_rng', 'ka']:
            raise ValueError('Unexpected trainer policy sample sites: ' + str(matches))
        text = re.sub(r'pi\.sample\(seed=([a-z_]+)\)',
            lambda m: f'select_action(pi, {m[1]}, config.get("ACTION_SELECTION", "categorical"))', text)
    if text.count('select_action(pi, ') != 2:
        raise ValueError('Expected both training and in-process evaluation to use the shared selector')
    return text


def patch_evaluator_action_selection(text):
    old = 'pi.sample(seed=ka)'
    new = 'trainer_mod.select_action(pi, ka, config.get("ACTION_SELECTION", "categorical"))'
    if old in text:
        text = replace_once(text, old, new)
    elif text.count(new) != 1:
        raise ValueError('Standalone evaluator must use the shared selector')
    return text


def patch_trainer(text):
    """Patch sampling and audit bookkeeping only; preserve each PPO implementation."""
    text = replace_once(text, 'from episode_scheduler import build_schedule, initial_episode_cursor, summarize_schedule',
        'from episode_scheduler import build_schedule, initial_episode_cursor, summarize_schedule\n'
        'from counterbalanced_scheduler import (load_schedule, initial_queue, dispatch, lookup,\n'
        '    accumulate_audit, write_audit, SAMPLING_PROTOCOL)')
    start = text.index('    # --- Build the (capability, layout) sample schedule')
    end = text.index('    def train(rng):', start)
    text = text[:start] + '''    # Preallocated packets, shared across profiles; no independent worker cursors.
    if config.get("SAMPLING_PROTOCOL") != SAMPLING_PROTOCOL:
        raise ValueError("This frozen experiment requires the counterbalanced sampler")
    schedule_path = config["COUNTERBALANCED_SCHEDULE_PATHS"][config["PARTNER_REGIME"]]
    schedule = load_schedule(schedule_path)
    if (schedule.n_layouts != env.n_layouts or schedule.rounds != env.rounds_per_episode
            or not np.array_equal(schedule.capability_pairs, np.asarray(_cap_pairs_py))):
        raise ValueError("Schedule corpus, episode length, or capability pool mismatch")
    # Each completed partner episode consumes at least R environment steps.
    required_capacity = config["TOTAL_TIMESTEPS"] // schedule.rounds + config["NUM_ENVS"] + 1
    if schedule.n_episodes < required_capacity:
        raise ValueError("Preallocated queue cannot cover the complete training budget")
    config["COUNTERBALANCED_SCHEDULE_PATH"] = schedule_path
    config["N_EPS_TOTAL"] = schedule.n_episodes
    print("[counterbalanced schedule]", schedule.summary(), flush=True)
    _CB_LAYOUTS = jnp.asarray(schedule.layouts, dtype=jnp.int32)
    _CB_ORDER = jnp.asarray(schedule.profile_order, dtype=jnp.int32)
    _CB_CAP = jnp.asarray(schedule.capability_pairs, dtype=jnp.int32)

''' + text[end:]
    pattern = r'        cursor0 = jnp.asarray\([\s\S]*?        layouts0 = [^\n]*\n'
    text, count = re.subn(pattern, '''        cursor0 = initial_queue(config["NUM_ENVS"], len(_cap_pairs_py), env.n_layouts)
        cap0, layouts0 = lookup(cursor0.slot_episode, _CB_LAYOUTS, _CB_ORDER, _CB_CAP)
''', text)
    if count != 1:
        raise ValueError('Initial queue patch did not match once')
    pattern = r'                new_cursor = jnp.where\([\s\S]*?                next_layouts = [^\n]*\n'
    text, count = re.subn(pattern, '''                new_cursor = dispatch(episode_cursor, done_all)
                next_cap, next_layouts = lookup(new_cursor.slot_episode, _CB_LAYOUTS, _CB_ORDER, _CB_CAP)
''', text)
    if count != 1:
        raise ValueError('Reset queue patch did not match once')
    text = replace_once(text, '            # --------- metrics ---------',
        '            episode_cursor = accumulate_audit(episode_cursor, traj_batch.info,\n'
        '                traj_batch.pre_step_time, traj_batch.done, _CB_CAP)\n\n'
        '            # --------- metrics ---------')
    text = replace_once(text, '        print(f"[main] saved resolved config -> {config_out_path}", flush=True)',
        '        print(f"[main] saved resolved config -> {config_out_path}", flush=True)\n'
        '        write_audit(out["runner_state"][7], out["runner_state"][1],\n'
        '                    config["COUNTERBALANCED_SCHEDULE_PATH"], config, save_path)')
    return text


def prepare(action_selection='greedy_random_ties'):
    if action_selection not in ('categorical', 'greedy_random_ties'):
        raise ValueError('Unknown action selection mode')
    now = datetime.now(timezone.utc)
    batch = 'counterbalanced1096_' + now.strftime('%Y%m%d_%H%M%S')
    if action_selection == 'greedy_random_ties':
        batch += '_greedy_action_selection'
    batch_root = ROOT / 'train/run_snapshots' / batch
    batch_root.mkdir(parents=True, exist_ok=False)
    corpus = ROOT / 'data_prep/grids_capability_selected_balanced_1096'
    layouts = sorted((corpus / 'layouts/train').glob('*.json'))
    if len(layouts) != 1096:
        raise ValueError('Expected the approved 1096-layout corpus')
    red, blue = Counter(), Counter()
    for path in layouts:
        meta = json.loads(path.read_text())['metadata']
        red[meta['ego_to_red']] += 1
        blue[meta['ego_to_blue']] += 1
    if red != blue or red != Counter({i: 137 for i in range(1, 9)}):
        raise ValueError('Corpus distance marginals changed')
    digest = hashlib.sha256()
    for path in layouts:
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    # Copy the corpus once, shared immutably across both versions.
    frozen_corpus = batch_root / 'corpus'
    shutil.copytree(corpus / 'layouts', frozen_corpus / 'layouts')
    for name in ['manifest.json', 'train_layouts.npz']:
        shutil.copyfile(corpus / name, frozen_corpus / name)
    (frozen_corpus / 'layout_ids.json').write_text(json.dumps([p.stem for p in layouts], indent=2) + '\n')
    # Read the authoritative population without importing JAX/environment modules.
    population = {}
    exec(compile((ROOT / 'jaxmarl/environments/coordination_grid/capability_populations.py').read_text(), 'capability_populations.py', 'exec'), population)
    pairs = population['TRAIN_CAPABILITY_PAIRS']
    minimum = 60000000 // 20 + 256 + 1
    schedules, schedule_paths = {}, {}
    for regime, caps in [('diverse', pairs), ('single', [(1, 4)])]:
        path = batch_root / 'schedules' / f'{regime}.npz'
        schedules[regime] = save_schedule(build_schedule(caps, 1096, 20, minimum, 2026), path)
        schedule_paths[regime] = str(path)

    source_files = []
    for directory in ['jaxmarl', 'train/config', 'eval', 'tests/coordination_grid']:
        source_files.extend(p for p in sorted((ROOT / directory).rglob('*'))
                            if p.is_file() and p.suffix in ['.py', '.yaml', '.yml'])
    source_files.extend(sorted((ROOT / 'train').glob('*.py')))
    source_files.append(ROOT / 'repo_paths.py')
    source_files.extend(ROOT / p for p in ['data_prep/capability_selection.py', 'bash/train_final_experiment.sh',
                       'bash/eval_all_checkpoints.sh', 'bash/submit_counterbalanced_training.py'])
    jobs, versions = [], {}
    manifest_path = ROOT / 'train/manifests' / f'sbatch_{batch}.json'
    for version in ['v1', 'v2']:
        label = version + '_balanced_training'
        snapshot = batch_root / label / 'source'
        for path in source_files:
            target = snapshot / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        if version == 'v1':
            for relative, target in V1_FILES.items():
                content = subprocess.check_output(['git', 'show', f'{V1_REVISION}:{relative}'], cwd=ROOT)
                (snapshot / target).write_text(relocate_v1_imports(content.decode()))
        trainer = snapshot / 'train/ippo_rnn_coordination_grid.py'
        baseline_trainer_sha = sha(trainer)
        trainer.write_text(patch_trainer(patch_action_selection(trainer.read_text())))
        evaluator = snapshot / 'eval/evaluate_partner_modelling.py'
        evaluator.write_text(patch_evaluator_action_selection(evaluator.read_text()))
        cfg_path = snapshot / 'train/config/ippo_coordination_grid.yaml'
        cfg = yaml.safe_load(cfg_path.read_text())
        protocol = 'fixed_v1' if version == 'v1' else 'online_v2'
        cfg.update({'ALLOCATION_PROTOCOL': protocol, 'EXPERIMENT_VERSION': label,
                    'SAMPLING_PROTOCOL': SAMPLING_PROTOCOL,
                    'ACTION_SELECTION': action_selection,
                    'COUNTERBALANCED_SCHEDULE_PATHS': schedule_paths})
        cfg['ENV_KWARGS']['layouts_dir'] = str(frozen_corpus / 'layouts/train')
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        launcher = snapshot / 'bash/train_final_experiment.sh'
        if launcher.read_text().count('    ALLOCATION_PROTOCOL="${ALLOCATION_PROTOCOL:-online_v2}"') != 1:
            raise ValueError('Launcher must honor the frozen allocation protocol')
        checkpoints = ROOT / 'train/train_logs' / label / batch
        evaluations = ROOT / 'eval/eval_out' / label / batch
        checkpoints.mkdir(parents=True, exist_ok=False)
        evaluations.mkdir(parents=True, exist_ok=False)
        hashes = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*') if p.is_file()}
        versions[version] = {'experiment_version': label, 'allocation_protocol': protocol,
                             'frozen_source_root': str(snapshot), 'source_sha256': hashes,
                             'baseline_trainer_sha256': baseline_trainer_sha,
                             'checkpoint_root': str(checkpoints), 'evaluation_root': str(evaluations),
                             'evaluation_jobs': []}
        common = {'REPO_ROOT': str(snapshot), 'CHECKPOINT_DIR': str(checkpoints),
                  'HYDRA_OUTPUT_DIR': str(ROOT / 'train/hydra_outputs' / batch),
                  'LAYOUTS_DIR': str(frozen_corpus / 'layouts/train'), 'ALLOCATION_PROTOCOL': protocol,
                  'NUM_SEEDS': '1', 'NUM_ENVS': '256', 'NUM_STEPS': '256',
                  'ACTION_SELECTION': action_selection,
                  'UPDATE_EPOCHS': '4', 'NUM_MINIBATCHES': '64', 'TOTAL_TIMESTEPS': '60000000',
                  'LR': '5e-4', 'MAX_STEPS': '100', 'STEP_PENALTY': '0.01',
                  'ROUNDS_PER_EPISODE': '20', 'HIDE_PARTNER_UNTIL_TIME': '0',
                  'SCHEDULE_SEED': '2026', 'N_EPS_TOTAL': str(schedules['diverse']['preallocated_episodes']),
                  'EVAL_EPISODES_PER_CAPABILITY': '20', 'WANDB_MODE': 'disabled',
                  'XLA_FLAGS': '', 'JAX_PLATFORMS': '', 'PYTHONDONTWRITEBYTECODE': '1'}
        for condition in CONDITIONS:
            for seed in range(1, 6):
                tag = f'{condition}_seed{seed}'
                exports = common | {'CONDITION': condition, 'SEED': str(seed), 'TAG': tag}
                jobs.append({'version': version, 'condition': condition, 'seed': seed, 'tag': tag,
                             'command': ['sbatch', '--parsable', f'--chdir={snapshot}',
                                         f'--job-name=cg_{version}_cb_{"greedy_" if action_selection == "greedy_random_ties" else ""}{tag}',
                                         '--export=ALL,' + ','.join(f'{k}={v}' for k, v in exports.items()), str(launcher)]})
    manifest = {'batch': batch, 'status': 'prepared', 'prepared_at_utc': now.isoformat(),
                'v1_baseline_revision': V1_REVISION, 'sampling_protocol': SAMPLING_PROTOCOL,
                'action_selection': action_selection,
                'action_selection_note': 'Unique probability maxima are deterministic; only exact equal maxima use seeded uniform tie-breaking. Training and evaluation use the same rule. PPO objective and entropy coefficient remain unchanged.',
                'source_corpus': str(corpus), 'frozen_layouts_dir': str(frozen_corpus / 'layouts/train'),
                'layout_files_sha256': digest.hexdigest(), 'n_layouts': 1096,
                'corpus_note': 'Both versions use 1096 layouts; original v1 used 1000.',
                'nominal_timesteps_per_policy': 60000000, 'effective_timesteps_per_policy': 59965440,
                'schedule_paths': schedule_paths, 'schedules': schedules,
                'versions': versions, 'training_jobs': jobs,
                'resources_per_training_job': {'account': 'nlp', 'partition': 'sphinx', 'gpus': 1,
                    'constraint': '80G', 'memory': '32G', 'cpus': 4, 'time_limit': '08:00:00', 'exclude': 'sphinx9'},
                'evaluation': {'episodes_per_capability': 20, 'seed': 12345, 'familiar_profiles': 24,
                               'novel_profiles': 22, 'time_limit': '02:00:00',
                               'action_selection': action_selection},
                'preflight': {}}
    write(manifest_path, manifest)
    print(manifest_path, flush=True)
    return manifest_path


def verify_manifest(manifest):
    manifest = resolve_record_paths(manifest)
    digest = hashlib.sha256()
    for p in sorted(Path(manifest['frozen_layouts_dir']).glob('*.json')):
        digest.update(p.name.encode()); digest.update(p.read_bytes())
    if digest.hexdigest() != manifest['layout_files_sha256']:
        raise ValueError('Frozen corpus changed')
    for regime, p in manifest['schedule_paths'].items():
        if sha(p) != manifest['schedules'][regime]['sha256']:
            raise ValueError('Frozen schedule changed: ' + regime)
    for v in manifest['versions'].values():
        for p, expected in v['source_sha256'].items():
            if sha(Path(v['frozen_source_root']) / p) != expected:
                raise ValueError('Frozen source changed: ' + p)


def submit(path):
    m = json.loads(path.read_text())
    verify_manifest(m)
    if m.get('status') == 'completed':
        print('Batch is already completed; no jobs submitted.', flush=True)
        return
    m = resolve_record_paths(m)
    if m['preflight'].get('passed') is not True:
        raise ValueError('Validated preflight results are required before submission')
    env = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_')
           and k not in ['JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES']}
    m['status'] = 'submitting'; write(path, m)
    for job in m['training_jobs']:
        if 'job_id' not in job:
            result = subprocess.check_output(job['command'], cwd=ROOT, env=env, text=True).strip().split(';')[0]
            if not result.isdigit():
                raise RuntimeError('Unexpected sbatch response: ' + result)
            job['job_id'] = result; write(path, m)
            print(job['version'], job['condition'], job['seed'], result, flush=True)
    for version, data in m['versions'].items():
        if data['evaluation_jobs']:
            continue
        dependencies = 'afterok:' + ':'.join(j['job_id'] for j in m['training_jobs'] if j['version'] == version)
        exports = {'REPO_ROOT': data['frozen_source_root'], 'CHECKPOINT_DIR': data['checkpoint_root'],
                   'EVAL_OUT_DIR': data['evaluation_root'], 'LAYOUTS_DIR': m['frozen_layouts_dir'],
                   'EXPECTED_CHECKPOINTS': '20', 'XLA_FLAGS': '', 'JAX_PLATFORMS': '', 'PYTHONDONTWRITEBYTECODE': '1'}
        command = ['sbatch', '--parsable', f"--chdir={data['frozen_source_root']}",
                   f'--job-name=cg_{version}_cb_eval', f'--dependency={dependencies}', '--kill-on-invalid-dep=yes',
                   '--export=ALL,' + ','.join(f'{k}={v}' for k, v in exports.items()),
                   str(Path(data['frozen_source_root']) / 'bash/eval_all_checkpoints.sh')]
        result = subprocess.check_output(command, cwd=ROOT, env=env, text=True).strip().split(';')[0]
        if not result.isdigit():
            raise RuntimeError('Unexpected sbatch response: ' + result)
        data['evaluation_jobs'].append({'job_id': result, 'dependency': dependencies, 'command': command})
        write(path, m); print(version, 'evaluation', result, flush=True)
    m['status'] = 'submitted'; m['submitted_at_utc'] = datetime.now(timezone.utc).isoformat(); write(path, m)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--submit-manifest', type=Path)
    parser.add_argument('--action-selection', choices=('greedy_random_ties', 'categorical'),
                        default='greedy_random_ties', help='Used when preparing a new frozen batch')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.action_selection)
    else:
        submit(args.submit_manifest)
