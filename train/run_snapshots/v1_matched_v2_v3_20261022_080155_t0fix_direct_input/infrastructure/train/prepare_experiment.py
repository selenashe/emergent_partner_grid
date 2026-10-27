"""Prepare a new run from the independently verified direct-input cohort."""
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil

import yaml

from repo_paths import artifact_path
from train.cohort_integrity import verify
from train.experiment_runtime import sha, write

INFRASTRUCTURE = (
    'bash/submit_t0fix_direct_input_training.py', 'bash/run_counterbalanced_seed_extension.sh',
    'repo_paths.py', 'train/__init__.py', 'train/experiment.py', 'train/experiment_runtime.py',
    'train/cohort_integrity.py', 'train/prepare_experiment.py', 'train/observation_pipeline.py',
    'train/ippo_rnn_coordination_grid.py', 'train/training_log_format.py',
)


def prepare(root, template_path):
    root = Path(root)
    template = verify(template_path)
    suffix = '_t0fix_direct_input'
    batch = 'v1_matched_v2_v3_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + suffix
    work = root / 'train/run_snapshots' / batch
    work.mkdir(parents=True, exist_ok=False)
    infrastructure = work / 'infrastructure'
    for rel in INFRASTRUCTURE:
        target = infrastructure / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, target)
    contract = work / 'direct_input_integrity.json'
    shutil.copyfile(template['integrity_contract'], contract)
    variants = {}
    for name, data in template['source_variants'].items():
        source = work / (name + suffix) / 'source'
        for rel in data['source_sha256']:
            target = source / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(data['frozen_source_root']) / rel, target)
        variants[name] = dict(frozen_source_root=str(source), source_sha256=dict(data['source_sha256']))
    versions, jobs = {}, []
    for version, data in template['versions'].items():
        outputs = {kind: str(root / rel / (version + suffix) / batch)
                   for kind, rel in [('checkpoint_root', 'train/train_logs'), ('evaluation_root', 'eval/eval_out')]}
        for directory in outputs.values():
            Path(directory).mkdir(parents=True, exist_ok=False)
            artifact_path(directory).mkdir(parents=True, exist_ok=False)
        versions[version] = dict(outputs, allocation_protocol=data['allocation_protocol'],
                                frozen_source_root=variants[version]['frozen_source_root'],
                                source_sha256=variants[version]['source_sha256'])
        for previous in template['training_jobs']:
            if previous['version'] != version:
                continue
            job = {k: previous[k] for k in ('version', 'condition', 'seed', 'tag', 'source_variant')}
            config = yaml.safe_load(Path(previous['config_path']).read_text())
            config['SAVE_PARAMS_PATH'] = str(Path(outputs['checkpoint_root']) / (job['tag'] + '.safetensors'))
            path = work / 'configs' / (version + suffix) / (job['tag'] + '.yaml')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            job.update(config_path=str(path), config_sha256=sha(path))
            jobs.append(job)
    manifest = {k: copy.deepcopy(template[k]) for k in (
        'observation_pipeline', 'observation_settings', 'conditions', 'action_selection',
        'nominal_timesteps_per_policy', 'effective_timesteps_per_policy', 'frozen_layouts_dir',
        'layout_files_sha256', 'schedule_paths', 'schedule_sha256', 'schedules', 'resources', 'evaluation')}
    logs = root / 'train/slurm_logs' / batch
    logs.mkdir(parents=True, exist_ok=False)
    manifest.update(batch=batch, status='prepared', artifact_suffix=suffix, seeds=[1, 2, 3, 4, 5],
        prepared_at_utc=datetime.now(timezone.utc).isoformat(), source_variants=variants,
        versions=versions, training_jobs=jobs, infrastructure_root=str(infrastructure),
        infrastructure_sha256={str(p.relative_to(infrastructure)): sha(p) for p in infrastructure.rglob('*') if p.is_file()},
        integrity_contract=str(contract), integrity_contract_sha256=sha(contract), log_dir=str(logs),
        comparison_root=str(root / 'eval/protocol_comparison' / batch),
        curves_root=str(root / 'train/training_curves' / batch),
        preflight=dict(passed=False, cpu_checks=[]), jobs={})
    path = root / 'train/manifests' / ('sbatch_' + batch + '.json')
    write(path, manifest)
    verify(path)
    print(path, flush=True)
    return path
