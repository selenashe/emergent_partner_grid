"""Protect the full cohort's scientific scope and Slurm dependency chain."""
import copy
from pathlib import Path

import pytest

from bash import submit_t0fix_direct_input_training as launcher


def test_configs_keep_rewards_optimizer_and_baseline_immutable():
    baseline = dict(SEED=3, TOTAL_TIMESTEPS=60000000, NUM_ENVS=256, NUM_STEPS=256,
                    LR=.0005, LR_WARMUP=.05, MODEL_TYPE='mlp', SINGLE_PARTNER=[1, 4],
                    ENV_KWARGS=dict(step_penalty=.01, rounds_per_episode=20, influence=False),
                    SAVE_PARAMS_PATH='/original/params.safetensors')
    before = copy.deepcopy(baseline)
    result = launcher.expected_config(baseline, Path('/new/params_t0fix_direct_input.safetensors'))
    assert baseline == before
    assert result['GRID_ENCODER'] == 'flat' and result['INCLUDE_IS_T0'] is True
    assert {k: v for k, v in result.items() if k not in (*launcher.SETTINGS, 'SAVE_PARAMS_PATH')} == {
        k: v for k, v in before.items() if k != 'SAVE_PARAMS_PATH'}
    result['ENV_KWARGS']['step_penalty'] = 9
    assert baseline['ENV_KWARGS']['step_penalty'] == .01


@pytest.mark.parametrize('smoke', [None, 'cpu', 'gpu'])
def test_training_commands_keep_full_schedule_and_explicit_config(tmp_path, monkeypatch, smoke):
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    monkeypatch.setattr(launcher, 'artifact_path', lambda x: x)
    job = dict(version='v2', source_variant='v2', tag='model' + launcher.SUFFIX,
               config_path=str(tmp_path / 'config.yaml'))
    m = dict(batch='new' + launcher.SUFFIX, source_variants={'v2': {'frozen_source_root': str(tmp_path / 'source')}})
    command, source, _ = launcher.training_command(m, job, smoke)
    assert str(source / launcher.TRAINER_FILE) in command
    assert command[command.index('--config-path') + 1] == str(tmp_path)
    assert ('+PILOT_UPDATES=1' in command) == bool(smoke)
    if smoke == 'cpu':
        assert 'TOTAL_TIMESTEPS=234240' in command  # 915 * 4 * 64
    else:
        assert not any(x.startswith(('TOTAL_TIMESTEPS=', 'NUM_ENVS=', 'NUM_STEPS=', 'UPDATE_EPOCHS=', 'NUM_MINIBATCHES='))
                       for x in command)


def test_arrays_cover_sixty_policies_and_evaluate_matching_checkpoints(tmp_path, monkeypatch):
    jobs = [dict(version=v, condition=c, seed=s, tag=f'{c}_seed{s}' + launcher.SUFFIX)
            for v in launcher.VERSIONS for c in launcher.CONDITIONS for s in launcher.SEEDS]
    m = dict(training_jobs=jobs, infrastructure_root=str(tmp_path / 'bundle'), log_dir=str(tmp_path / 'logs'),
             preflight={'passed': True}, jobs={})
    monkeypatch.setattr(launcher, 'verify', lambda _: m)
    saved = []
    monkeypatch.setattr(launcher, 'write', lambda _, data: saved.append(copy.deepcopy(data)))
    calls = []
    def sbatch(command, **kwargs):
        calls.append(command)
        return str(100 + len(calls)) + '\n'
    monkeypatch.setattr(launcher.subprocess, 'check_output', sbatch)
    launcher.submit(tmp_path / 'manifest.json')
    assert len(calls) == 4
    assert '--array=0,5,10,15,20,25,30,35,40,45,50,55%4' in calls[0]
    assert '--array=0-59%12' in calls[1] and '--dependency=afterok:101' in calls[1]
    assert '--array=0-59%6' in calls[2] and '--dependency=aftercorr:102' in calls[2]
    assert '--dependency=afterok:103' in calls[3] and '--gres=gpu:1' not in calls[3]
    assert all('--kill-on-invalid-dep=yes' in call for call in calls)
    assert all(launcher.SUFFIX in j['log_path'] for j in jobs)
    assert [j['job_id'] for j in jobs] == [f'102_{i}' for i in range(60)]
    assert [len(x['jobs']) for x in saved[1:5]] == [1, 2, 3, 4]
    launcher.submit(tmp_path / 'manifest.json')
    assert len(calls) == 4


def test_cannot_submit_without_passing_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'verify', lambda _: {'preflight': {'passed': False}})
    with pytest.raises(ValueError, match='CPU preflight'):
        launcher.submit(tmp_path / 'manifest.json')
