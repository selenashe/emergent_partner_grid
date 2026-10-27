"""Binary relocation preserves historical readers and local summary records."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import repo_paths


@pytest.fixture
def storage(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    project.mkdir()
    scratch = tmp_path / 'scratch'
    monkeypatch.setattr(repo_paths, 'ROOT', project)
    monkeypatch.setattr(repo_paths, 'ARCHIVE_RELOCATIONS', {})
    monkeypatch.setenv('GRID_ARTIFACT_ROOT', str(scratch))
    return project, scratch


def test_historical_binary_and_summary_resolve_without_changing_record(storage):
    project, scratch = storage
    relative = Path('train/train_logs/batch/policy.safetensors')
    target = scratch / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b'checkpoint')
    legacy = repo_paths.ORIGINAL_ROOT / 'dev/train_logs/batch/policy.safetensors'
    assert repo_paths.resolve_path(legacy) == target
    logical = project / relative
    repo_paths.link_artifact(logical)
    assert logical.is_symlink() and logical.read_bytes() == b'checkpoint'
    assert repo_paths.resolve_path(logical) == target
    summary = logical.with_name('policy_config.json')
    summary.write_text('{"seed": 1}')
    record = {'weights': str(legacy), 'config': str(summary)}
    before = json.dumps(record)
    resolved = repo_paths.resolve_record_paths(record)
    assert resolved == {'weights': str(target), 'config': str(summary)}
    assert json.dumps(record) == before


def test_numpy_round_trip_and_replacement_leave_json_local(storage):
    project, scratch = storage
    logical = project / 'eval/results/features/policy.npz'
    repo_paths.save_numpy_archive(logical, compressed=True, hidden=np.arange(6))
    assert logical.is_symlink()
    assert logical.resolve() == scratch / logical.relative_to(project)
    with np.load(logical) as arrays:
        np.testing.assert_array_equal(arrays['hidden'], np.arange(6))
    summary = logical.with_suffix('.json')
    summary.write_text('{"episodes": 6}')
    repo_paths.save_numpy_archive(logical, compressed=True, hidden=np.arange(3))
    with np.load(logical) as arrays:
        np.testing.assert_array_equal(arrays['hidden'], np.arange(3))
    assert summary.read_text() == '{"episodes": 6}'
    assert not (scratch / summary.relative_to(project)).exists()


def test_explicit_external_outputs_are_preserved(storage, tmp_path):
    external = tmp_path / 'custom/policy.npz'
    repo_paths.save_numpy_archive(external, values=np.arange(2))
    assert repo_paths.artifact_path(external) == external
    assert external.is_file() and not external.is_symlink()


def test_safetensors_round_trip(storage):
    from jaxmarl.wrappers.baselines import save_params, load_params
    import jax.numpy as jnp
    project, scratch = storage
    logical = project / 'train/train_logs/test/model.safetensors'
    save_params({'policy': {'weight': jnp.arange(4, dtype=jnp.float32)}}, logical)
    assert logical.is_symlink()
    assert logical.resolve() == scratch / logical.relative_to(project)
    np.testing.assert_array_equal(load_params(logical)['policy']['weight'], np.arange(4))


def test_hdf5_writer_uses_scratch_and_preserves_protocol(storage):
    from eval.evaluate_partner_modelling import _save_hdf5
    import h5py
    project, scratch = storage
    logical = project / 'eval/eval_out/test/policy_train.h5'
    _save_hdf5(str(logical), {'actions': np.arange(3)}, 'online_v2')
    assert logical.is_symlink()
    assert logical.resolve() == scratch / logical.relative_to(project)
    with h5py.File(logical) as stream:
        assert stream.attrs['allocation_protocol'] == 'online_v2'
        np.testing.assert_array_equal(stream['actions'][:], np.arange(3))


def test_schedule_binary_moves_but_parameters_and_checksum_stay_local(storage):
    from train.counterbalanced_scheduler import build_schedule, save_schedule, load_schedule
    project, scratch = storage
    logical = project / 'train/run_snapshots/test/schedules/diverse.npz'
    schedule = build_schedule([(1, 4), (4, 1)], 4, 2, 8, 2026)
    summary = save_schedule(schedule, logical)
    assert logical.is_symlink()
    assert logical.resolve() == scratch / logical.relative_to(project)
    assert json.loads(logical.with_suffix('.json').read_text()) == summary
    assert hashlib.sha256(logical.read_bytes()).hexdigest() == summary['sha256']
    np.testing.assert_array_equal(load_schedule(logical).layouts, schedule.layouts)


def test_artifact_root_requires_absolute_path(monkeypatch):
    monkeypatch.setenv('GRID_ARTIFACT_ROOT', 'relative/path')
    with pytest.raises(ValueError, match='absolute'):
        repo_paths.artifact_root()


def test_default_checkpoint_config_honors_storage_override(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    monkeypatch.setenv('GRID_ARTIFACT_ROOT', str(tmp_path / 'scratch'))
    config_file = Path(__file__).resolve().parents[2] / 'train/config/ippo_coordination_grid.yaml'
    config = OmegaConf.to_container(OmegaConf.load(config_file), resolve=True)
    assert config['SAVE_PARAMS_PATH'] == str(
        tmp_path / 'scratch/train/train_logs/counterbalanced/capability_params.safetensors')


def test_frozen_training_command_places_hydra_logs_on_scratch(storage, monkeypatch):
    from bash import extend_counterbalanced_training as launcher
    project, scratch = storage
    # Importing the launcher establishes its shared workspace root. Select the
    # temporary project for this test after that initialization.
    monkeypatch.setattr(repo_paths, 'ROOT', project)
    monkeypatch.setattr(launcher, 'ROOT', project)
    manifest = {'batch': 'test_batch', 'versions': {
        'v1_matched': {'frozen_source_root': str(project / 'train/run_snapshots/test/source')}}}
    job = {'version': 'v1_matched', 'tag': 'rnn_seed1',
           'config_path': str(project / 'train/run_snapshots/test/configs/rnn_seed1.yaml')}
    command, source, target = launcher.training_command(manifest, job)
    assert target == scratch / 'train/hydra_outputs/test_batch/v1_matched/rnn_seed1'
    assert f'hydra.run.dir={target}' in command
    assert source == Path(manifest['versions']['v1_matched']['frozen_source_root'])
