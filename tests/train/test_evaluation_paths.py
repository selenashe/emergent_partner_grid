"""Evaluation must find frozen trainers' local configs and checkpoint files."""
from pathlib import Path

import pytest

import repo_paths
from train import experiment_runtime as launcher


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    scratch = tmp_path / 'scratch'
    monkeypatch.setattr(repo_paths, 'ROOT', project)
    monkeypatch.setattr(repo_paths, 'ARCHIVE_RELOCATIONS', {})
    monkeypatch.setenv('GRID_ARTIFACT_ROOT', str(scratch))
    manifest = {
        'source_variants': {'v2': {'frozen_source_root': str(project / 'source')}},
        'versions': {'v2': {
            'checkpoint_root': str(project / 'train/train_logs/v2/batch'),
            'evaluation_root': str(project / 'eval/eval_out/v2/batch'),
        }},
        'frozen_layouts_dir': str(project / 'layouts'),
    }
    job = dict(version='v2', source_variant='v2',
               condition='rnn_diverse_influence',
               tag='rnn_diverse_influence_seed1_t0fix_direct_input')
    return project, scratch, manifest, job


def save_inputs(manifest, job, placement):
    checkpoint = Path(manifest['versions']['v2']['checkpoint_root']) / job['tag']
    inputs = []
    for extension in ['_config.json', '.safetensors']:
        logical = Path(str(checkpoint) + extension)
        physical = repo_paths.artifact_path(logical)
        path = physical if placement == 'scratch' or (
            placement == 'split' and extension == '.safetensors') else logical
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'saved training artifact')
        inputs.append(path)
    return inputs


@pytest.mark.parametrize('placement', ['local', 'scratch', 'split'])
def test_resolves_each_input_and_publishes_rollouts(evaluation, monkeypatch, placement):
    project, scratch, manifest, job = evaluation
    config, params = save_inputs(manifest, job, placement)
    expected_prefix = scratch / 'eval/eval_out/v2/batch' / job['tag']
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert Path(command[command.index('--config') + 1]) == config
        assert Path(command[command.index('--params') + 1]) == params
        assert command[command.index('--n_episodes_per_capability') + 1] == '20'
        assert command[command.index('--seed') + 1] == '12345'
        assert '--save_hidden' in command
        assert Path(command[command.index('--out_prefix') + 1]) == expected_prefix
        for extension in ['_summary.json', '_train.h5', '_test.h5']:
            Path(str(expected_prefix) + extension).write_bytes(b'evaluation result')

    monkeypatch.setattr(launcher.subprocess, 'run', run)
    monkeypatch.setattr(launcher, 'environment', lambda *args, **kwargs: {})
    launcher.evaluate_job(manifest, job)
    assert len(calls) == 1
    for extension in ['_summary.json', '_train.h5', '_test.h5']:
        logical = project / 'eval/eval_out/v2/batch' / (job['tag'] + extension)
        assert logical.is_symlink()
        assert logical.resolve() == Path(str(expected_prefix) + extension)
    assert config.read_bytes() == params.read_bytes() == b'saved training artifact'


@pytest.mark.parametrize('missing', [0, 1])
def test_missing_input_fails_before_starting_evaluation(evaluation, monkeypatch, missing):
    _, scratch, manifest, job = evaluation
    inputs = save_inputs(manifest, job, 'local')
    inputs[missing].unlink()

    def unexpected_run(*args, **kwargs):
        pytest.fail('Evaluator started with a missing input')

    monkeypatch.setattr(launcher.subprocess, 'run', unexpected_run)
    with pytest.raises(FileNotFoundError) as error:
        launcher.evaluate_job(manifest, job)
    assert error.value.args[0] == inputs[missing]
    assert not scratch.exists()
