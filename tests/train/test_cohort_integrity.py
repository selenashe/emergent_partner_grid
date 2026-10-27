"""Cleanup must retain independent scientific and frozen-source integrity checks."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from train import cohort_integrity
from train.experiment_runtime import sha

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    path = ROOT / 'train/manifests/sbatch_v1_matched_v2_v3_20261022_080155_t0fix_direct_input.json'
    record = json.loads(path.read_text())
    # The original checksum migration separately reads both full schedule arrays.
    # Unit tests avoid repeatedly reading GB-sized arrays unrelated to tampering.
    schedule_hashes = {str(Path(p)): record['schedule_sha256'][k]
                       for k, p in record['schedule_paths'].items()}
    monkeypatch.setattr(cohort_integrity, 'sha', lambda p: schedule_hashes.get(str(Path(p))) or sha(p))
    target = tmp_path / 'manifest.json'
    target.write_text(json.dumps(record))
    return target, record


def test_sealed_cohort_verifies_without_baseline_pointers(manifest):
    path, record = manifest
    record.pop('baseline_manifests', None)
    for variant in record['source_variants'].values():
        variant.pop('original_source_root', None)
        variant.pop('original_source_sha256', None)
    for job in record['training_jobs']:
        for key in list(job):
            if key.startswith('baseline_'):
                del job[key]
    path.write_text(json.dumps(record))
    assert len(cohort_integrity.verify(path)['training_jobs']) == 60


def test_changed_config_cannot_be_hidden_by_updating_manifest_hash(manifest, tmp_path):
    path, record = manifest
    job = record['training_jobs'][0]
    config = yaml.safe_load(Path(job['config_path']).read_text())
    config['LR'] *= 2
    target = tmp_path / 'changed.yaml'
    target.write_text(yaml.safe_dump(config))
    job.update(config_path=str(target), config_sha256=sha(target))
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Scientific config changed'):
        cohort_integrity.verify(path)


def test_source_inventory_cannot_be_resealed_in_manifest(manifest):
    path, record = manifest
    variant = next(iter(record['source_variants'].values()))
    variant['source_sha256'][next(iter(variant['source_sha256']))] = '0' * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Scientific source contract changed'):
        cohort_integrity.verify(path)


def test_changed_contract_is_rejected(manifest, tmp_path):
    path, record = manifest
    contract = tmp_path / 'contract.json'
    contract.write_text('{}')
    record['integrity_contract'] = str(contract)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='integrity contract changed'):
        cohort_integrity.verify(path)


def test_influence_policy_cannot_use_random_control_source(manifest):
    path, record = manifest
    job = next(j for j in record['training_jobs'] if j['version'] == 'v1_matched'
               and j['condition'] == 'rnn_diverse_influence')
    job['source_variant'] = 'v1_matched'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='condition-specific protocol source'):
        cohort_integrity.verify(path)
