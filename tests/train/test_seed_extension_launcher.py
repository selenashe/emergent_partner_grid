"""Slurm's relocated script must use the explicitly supplied frozen runner."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bash import extend_counterbalanced_training as extension


@pytest.fixture
def spooled_launcher(tmp_path):
    # Slurm copies the shell script without its neighboring Python files.
    # Stub conda activation so this shell-path regression runs without the
    # cluster's environment installation; retain the actual launcher logic.
    spool = tmp_path / 'slurm spool'
    spool.mkdir()
    shell = spool / 'slurm_script'
    original = Path(__file__).resolve().parents[2] / 'bash/run_counterbalanced_seed_extension.sh'
    shutil.copyfile(original, shell)
    stub = tmp_path / 'bash_env'
    stub.write_text('source() { :; }\nconda() { :; }\n')
    runner = tmp_path / 'shared source' / 'runner.py'
    runner.parent.mkdir()
    runner.write_text('import json, sys\nprint(json.dumps({"file": __file__, "args": sys.argv[1:]}))\n')
    env = dict(os.environ, BASH_ENV=str(stub))
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    return shell, runner, env


@pytest.mark.parametrize('stage', ['verify', 'train', 'eval', 'aggregate', 'gpu-smoke', 'production-smoke'])
def test_copied_script_finds_absolute_runner_for_every_stage(spooled_launcher, stage):
    shell, runner, env = spooled_launcher
    args = [stage, '/shared/manifest.json', 'v2', 'rnn_diverse_influence_seed6']
    result = subprocess.run(['bash', str(shell), str(runner), *args], env=env,
                            text=True, capture_output=True, check=True)
    report = json.loads(result.stdout)
    assert report == {'file': str(runner), 'args': args}


def test_copied_script_rejects_relative_runner(spooled_launcher):
    shell, runner, env = spooled_launcher
    result = subprocess.run(['bash', str(shell), runner.name, 'verify'], env=env,
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert 'Missing absolute runner file' in result.stderr


@pytest.mark.parametrize("guarded", [False, True])
def test_every_submission_passes_absolute_runner_and_resume_skips_recorded_ids(tmp_path, monkeypatch, guarded):
    infrastructure = tmp_path / 'infrastructure'
    manifest = dict(status='prepared', preflight={'passed': True}, infrastructure_root=str(infrastructure),
                    log_dir=str(tmp_path / 'logs'), versions={v: {'evaluation_jobs': []} for v in ('v1', 'v2')},
                    training_jobs=[dict(version=v, condition=c, seed=s, tag=f'{c}_seed{s}')
                                   for v in ('v1', 'v2') for c in extension.CONDITIONS for s in extension.NEW_SEEDS],
                    aggregation_jobs=[])
    if guarded:
        manifest['preflight']['slurm_launcher_checks'] = [dict(job_id=str(i)) for i in (901, 902)]
    commands = []
    def capture(command, **kwargs):
        commands.append(command)
        return str(10000 + len(commands)) + '\n'
    monkeypatch.setattr(extension, 'verify', lambda _: manifest)
    monkeypatch.setattr(extension, 'write', lambda *args: None)
    monkeypatch.setattr(extension.subprocess, 'check_output', capture)
    path = tmp_path / 'manifest.json'
    extension.submit(path)
    assert len(commands) == 43
    wrapper = str(infrastructure / 'bash/run_counterbalanced_seed_extension.sh')
    runner = str(infrastructure / 'bash/extend_counterbalanced_training.py')
    for command in commands:
        index = command.index(wrapper)
        assert command[index + 1] == runner
        assert Path(command[index + 1]).is_absolute()
        assert command[index + 2] in ('train', 'eval', 'aggregate')
        if command[index + 2] == 'train':
            if guarded:
                assert '--dependency=afterok:901:902' in command
                assert '--kill-on-invalid-dep=yes' in command
            else:
                assert not any(arg.startswith('--dependency=') for arg in command)
    extension.submit(path)
    assert len(commands) == 43


@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_production_gpu_check_retains_original_tensor_dimensions(tmp_path, monkeypatch, version):
    import yaml
    monkeypatch.setattr(extension, 'ROOT', tmp_path)
    config = tmp_path / 'configs' / version / 'rnn_diverse_influence_seed6.yaml'
    config.parent.mkdir(parents=True)
    cfg = dict(NUM_ENVS=256, NUM_STEPS=256, UPDATE_EPOCHS=4, NUM_MINIBATCHES=64,
               TOTAL_TIMESTEPS=60000000, EVAL_EPISODES_PER_CAPABILITY=20)
    config.write_text(yaml.safe_dump(cfg))
    source = tmp_path / 'original_frozen_source'
    manifest = dict(batch='extension', versions={version: {'frozen_source_root': str(source)}})
    job = dict(version=version, tag='rnn_diverse_influence_seed6',
               condition='rnn_diverse_influence', config_path=str(config))
    command, directory, target = extension.training_command(manifest, job, production_smoke=True)
    assert directory == source
    assert str(source / 'baselines/IPPO/ippo_rnn_coordination_grid.py') in command
    assert 'TOTAL_TIMESTEPS=65536' in command
    assert 'EVAL_EPISODES_PER_CAPABILITY=1' in command
    for key in ('NUM_ENVS', 'NUM_STEPS', 'UPDATE_EPOCHS', 'NUM_MINIBATCHES', 'ENV_KWARGS.max_steps'):
        assert not any(arg.startswith(key + '=') for arg in command)
    assert 'slurm_production_preflight' in target.parts
    assert yaml.safe_load(config.read_text()) == cfg
