"""A fresh five-seed batch must prepare without any original CNN experiments."""
import json
from pathlib import Path
import shutil

from train import cohort_integrity, prepare_experiment
from train.experiment import TRAINING_MANIFEST
from train.experiment_runtime import sha
import repo_paths


def test_prepare_uses_only_retained_configs_and_protocol_sources(tmp_path, monkeypatch):
    original = json.loads(TRAINING_MANIFEST.read_text())
    schedules = {str(Path(p)): original['schedule_sha256'][k] for k, p in original['schedule_paths'].items()}
    monkeypatch.setattr(cohort_integrity, 'sha', lambda p: schedules.get(str(Path(p))) or sha(p))
    project = tmp_path / 'project'
    actual = Path(__file__).resolve().parents[2]
    for rel in prepare_experiment.INFRASTRUCTURE:
        target = project / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(actual / rel, target)
    monkeypatch.setattr(repo_paths, 'ROOT', project)
    monkeypatch.setenv('GRID_ARTIFACT_ROOT', str(tmp_path / 'scratch'))
    path = prepare_experiment.prepare(project, TRAINING_MANIFEST)
    record = cohort_integrity.verify(path)
    assert len(record['training_jobs']) == 60 and record['seeds'] == [1, 2, 3, 4, 5]
    assert len(record['source_variants']) == 4
    assert all(Path(v['frozen_source_root']).is_relative_to(project) for v in record['source_variants'].values())
    assert 'baseline_manifests' not in record
    assert all(not any(k.startswith('baseline_') for k in j) for j in record['training_jobs'])
    assert all('original_source_root' not in v for v in record['source_variants'].values())
    assert not (project / 'train/manifests/sbatch_counterbalanced1096_20261002_235609.json').exists()
