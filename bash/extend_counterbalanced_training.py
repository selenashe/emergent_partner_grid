"""Compatibility exports for generic frozen-run IO; the seed-extension CLI is retired.

The maintained direct-input launcher uses train.experiment_runtime directly.
"""
import os
from pathlib import Path
import sys
import yaml
import repo_paths as path_resolver
from repo_paths import resolve_record_paths, artifact_path, artifact_root
from train.experiment import CONDITIONS
from train.experiment_runtime import PYTHON, sha, write, environment
ROOT = Path(os.environ.get('GRID_PROJECT_ROOT', Path(__file__).resolve().parents[1]))

def load(path):
    return resolve_record_paths(json.loads(Path(path).read_text()))

def relocated_config(base, version, condition):
    """Reuse a completed policy's settings, resolving obsolete directory names.

    Source code and scientific settings come from the original experiment.
    Changing a filename's location does not change what the model learns.
    """
    path = Path(base['versions'][version]['checkpoint_root']) / f'{condition}_seed1_config.json'
    return load(path), path

def training_command(m, job, smoke=False, gpu_smoke=False, production_smoke=False):
    """Invoke the original Hydra trainer with the extension's resolved config.

    Absolute config directories avoid editing the historical source snapshot.
    Smoke runs shorten only compute budget and episode horizons; production
    runs receive the original 60M-step settings without those overrides.
    """
    source = Path(job.get('frozen_source_root', m['versions'][job['version']]['frozen_source_root']))
    cfg_path = Path(job['config_path'])
    target = artifact_path(ROOT / 'train/hydra_outputs' / m['batch'] / job['version'] / job['tag'])
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
