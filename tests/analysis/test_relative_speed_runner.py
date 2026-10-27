"""Specialist-only preparation must work without any previous result folder."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from eval import run_relative_speed_representation as runner


@pytest.fixture
def inputs(tmp_path,monkeypatch):
    repo=tmp_path/'repo';repo.mkdir();monkeypatch.setattr(runner,'ROOT',repo)
    for relative in ['eval/run_relative_speed_representation.py','eval/relative_speed_representation.py',
        'eval/representation_analysis.py','repo_paths.py','jaxmarl/environments/coordination_grid/capability_populations.py']:
        path=repo/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# current producer\n')
    raw=repo/'inputs';raw.mkdir()
    for name in ['checkpoint','config','train.h5','test.h5']:(raw/name).write_bytes(name.encode())
    evaluator=raw/'analysis/evaluate_partner_modelling.py';evaluator.parent.mkdir();evaluator.write_text('# frozen evaluator\n')
    records=[]
    for version in runner.VERSIONS:
        for seed in range(1,6):
            for condition in runner.CONDITIONS:
                records.append(dict(version=version,condition=condition,seed=seed,source_root=str(raw),
                    evaluator_sha256=runner.sha(evaluator),checkpoint=str(raw/'checkpoint'),checkpoint_sha256=runner.sha(raw/'checkpoint'),
                    config=str(raw/'config'),config_sha256=runner.sha(raw/'config'),layouts_dir=str(raw),allocation_protocol='fixed_allocation',
                    original_rollouts={k:str(raw/f'{k}.h5') for k in ['train','test']},
                    original_stats={k:dict(size=(raw/f'{k}.h5').stat().st_size,mtime_ns=(raw/f'{k}.h5').stat().st_mtime_ns) for k in ['train','test']}))
    manifest=repo/'inputs.json';runner.write(manifest,dict(status='verified_inputs',policies=records))
    monkeypatch.setattr(runner,'INPUT_MANIFEST',manifest)
    return repo,manifest


def test_prepare_needs_only_original_input_manifest(inputs):
    repo,manifest=inputs;out=repo/'new_results'
    saved=json.loads(runner.prepare(out).read_text())
    assert saved['schema_version']==2 and len(saved['policies'])==45
    assert (saved['episodes_per_policy'],saved['train_episodes_per_policy'],saved['test_episodes_per_policy'])==(920,736,184)
    assert saved['train_counts']==[368,368] and saved['test_counts']==[92,92]
    assert len(saved['classes'])==2
    assert saved['input_manifest_sha256']==runner.sha(manifest)
    for record in saved['policies']:
        assert set(record['original_rollouts'])=={'train','test'}
        assert Path(record['analysis_dir']).is_relative_to(out)
    runner.verify(saved)
    for stage in ['analyze','aggregate']:runner.script(saved,stage)
    with pytest.raises(ValueError,match='Only analysis'):runner.script(saved,'evaluate')


@pytest.mark.parametrize('changed',['checkpoint','train.h5','config'])
def test_changed_inputs_are_rejected_before_creating_results(inputs,changed):
    repo,_=inputs;(repo/'inputs'/changed).write_bytes(b'changed input')
    with pytest.raises(ValueError,match='changed'):runner.prepare(repo/'new_results')
    assert not (repo/'new_results').exists()


def test_duplicate_policy_mapping_is_rejected(inputs):
    repo,path=inputs;manifest=json.loads(path.read_text());manifest['policies'][-1]=manifest['policies'][0];runner.write(path,manifest)
    with pytest.raises(ValueError,match='45 distinct'):runner.prepare(repo/'new_results')
    assert not (repo/'new_results').exists()
