"""All-ten aggregation must weight old/new learner seeds equally and fail on gaps."""
import json
from pathlib import Path
import sys
import shutil

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval import compare_allocation_protocols as comparison


@pytest.fixture
def mixed_evaluations(tmp_path, monkeypatch):
    inputs = {}
    for version in ('v1', 'v2'):
        inputs[version] = {}
        for seed in range(1, 11):
            directory = tmp_path / version / ('original' if seed <= 5 else 'extension')
            directory.mkdir(parents=True, exist_ok=True)
            inputs[version][str(seed)] = str(directory)
            for condition in comparison.CONDITIONS:
                # Deliberately different old/new values expose accidental five-seed
                # aggregation or weighting based on file locations.
                success = seed / 10
                summaries = {}
                for population, count in (('train', 480), ('test', 440)):
                    summaries[population] = dict(n_episodes=count, n_rounds=count * 20,
                        allocation_protocol='fixed_v1' if version == 'v1' else 'online_v2',
                        round_success_rate=success, mean_ep_return=1.01 * 20 * success - .2,
                        mean_successful_completion_time=seed,
                        per_round_success_rate=[success] * 20)
                    rollout = directory / f'{condition}_seed{seed}_{population}.h5'
                    with h5py.File(rollout, 'w') as f:
                        dones = np.zeros((count, 23), bool)
                        dones[:, 19:] = True
                        f['dones'] = dones
                        # Padding intentionally also reports round_done; it must be
                        # excluded when checking exactly 20 completed rounds.
                        f['round_done'] = np.ones((count, 23), bool)
                (directory / f'{condition}_seed{seed}_summary.json').write_text(json.dumps(summaries))
    monkeypatch.setattr(comparison, 'INPUTS', inputs)
    monkeypatch.setattr(comparison, 'SEEDS', tuple(range(1, 11)))
    monkeypatch.setattr(comparison, 'COUNTERBALANCED', True)
    monkeypatch.setattr(comparison, 'OUT', tmp_path / 'comparison')
    return inputs


@pytest.mark.parametrize('population', ['train', 'test'])
def test_ten_seed_means_and_sample_sd_include_both_directories(mixed_evaluations, monkeypatch, population):
    monkeypatch.setattr(comparison, 'POPULATION', population)
    comparison.main()
    report = json.loads((comparison.OUT / 'comparison.json').read_text())
    assert report['seeds'] == list(range(1, 11))
    assert report['population'] == population
    assert len(report['aggregates']) == 8
    for aggregate in report['aggregates']:
        assert aggregate['n_seeds'] == 10
        assert aggregate['success_mean'] == pytest.approx(.55)
        assert aggregate['success_sd'] == pytest.approx(np.std(np.arange(1, 11) / 10, ddof=1))
        assert aggregate['episode_steps_mean'] == 20
    assert len(report['sources']) == 160
    assert sum('/original/' in p for p in report['sources']) == 80
    assert sum('/extension/' in p for p in report['sources']) == 80


def test_missing_new_seed_fails_instead_of_silently_averaging_five(mixed_evaluations, monkeypatch):
    monkeypatch.setattr(comparison, 'POPULATION', 'test')
    missing = Path(mixed_evaluations['v2']['10']) / 'rnn_diverse_noinfluence_seed10_summary.json'
    missing.unlink()
    with pytest.raises(FileNotFoundError):
        comparison.main()
    assert not (comparison.OUT / 'comparison.json').exists()


def test_v3_is_included_in_all_ten_seed_aggregates_and_plots(mixed_evaluations, monkeypatch, tmp_path):
    mixed_evaluations['v3'] = {}
    for directory in set(mixed_evaluations['v2'].values()):
        target = tmp_path / 'v3' / Path(directory).name
        shutil.copytree(directory, target)
        for summary in target.glob('*_summary.json'):
            record = json.loads(summary.read_text())
            for population in record.values():
                population['allocation_protocol'] = 'online_v3'
            summary.write_text(json.dumps(record))
    for seed in range(1, 11):
        mixed_evaluations['v3'][str(seed)] = str(tmp_path / 'v3' / Path(mixed_evaluations['v2'][str(seed)]).name)
    monkeypatch.setattr(comparison, 'POPULATION', 'test')
    comparison.main()
    report = json.loads((comparison.OUT / 'comparison.json').read_text())
    assert len(report['aggregates']) == 12
    assert set(report['trajectories']) == {'v1', 'v2', 'v3'}
    assert len(report['sources']) == 240
    assert all(a['n_seeds'] == 10 for a in report['aggregates'])
    assert all(a['success_mean'] == pytest.approx(.55) for a in report['aggregates'] if a['version'] == 'v3')
    for name in ('performance_comparison.png', 'episode_steps.png', 'round_comparison.png'):
        assert (comparison.OUT / name).is_file()
