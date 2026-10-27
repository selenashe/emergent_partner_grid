"""Exercise five-seed aggregators and real checkpoint/selection input mappings."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd


def validate(path):
    m = json.loads(Path(path).read_text())
    root = Path(m['analysis_root'])
    os.environ['GRID_PROJECT_ROOT'] = str(root)
    sys.path.insert(0, str(root)); sys.path.insert(0, str(root / 'eval'))
    from eval import relative_speed_representation as relative
    from eval import round_state_prediction_comparison as states
    from eval import run_relative_speed_representation as runner
    from eval import plot_overcooked_paper_match as paper
    for module in (relative, states, runner, paper):
        assert Path(module.__file__).resolve().is_relative_to(root)
    checks = []
    rows = [dict(version='v3', condition='rnn_diverse_influence', axis='round', cutoff=20,
        kind='primary', population='all', training_seed=i, balanced_accuracy=.5+.05*i,
        accuracy=.5+.05*i, recall_0=.5+.05*i, recall_1=.5+.05*i) for i in range(1, 6)]
    result = relative.summary(pd.DataFrame(rows))
    np.testing.assert_allclose(result.balanced_accuracy_mean, [.65])
    assert result.balanced_accuracy_ci_low.iloc[0] <= .65 <= result.balanced_accuracy_ci_high.iloc[0]
    checks.append('five-seed cumulative-probe aggregation and bootstrap')
    fractions = pd.DataFrame([dict(version='v3', condition='rnn_diverse_influence', feature_type='round_end_state',
        round=20, ground_truth_class=0, predicted_class=0, training_seed=i,
        prediction_fraction=.5+.05*i) for i in range(1, 6)])
    result = states.summarize(fractions)
    np.testing.assert_allclose(result['mean'], [.65]); assert result.n_seeds.iloc[0] == 5
    for function, table in ((relative.summary, pd.DataFrame(rows)), (states.summarize, fractions)):
        try: function(table[table.training_seed != 5])
        except ValueError: pass
        else: raise AssertionError('Missing seed accepted')
    checks.append('five-seed discrete-state aggregation and incomplete-cohort rejection')
    records = paper.load_manifests(m['view_manifest'])
    candidates = paper.training_candidates(records)
    assert len(candidates) == 60
    selected = paper.select_candidates(candidates, list(range(1, 6)))
    training = json.loads(Path(m['training_manifest']).read_text())
    prior = Path(training['comparison_root']) / 'dot_plots_t0fix_direct_input/best_seed_selection_t0fix_direct_input.csv'
    previous = {(r['version'], r['condition']): int(r['selected_seed']) for r in csv.DictReader(prior.open())
                if r['cohort'] == 't0fix_direct_input'}
    assert {(r['version'], r['condition']): r['seed'] for r in selected} == previous
    checks.append('all 60 training logs read and 12 best-seed selections match existing dot plots')
    inputs = json.loads(Path(m['representation_inputs']).read_text())
    runner.verify_inputs(inputs)
    assert len(inputs['policies']) == 45
    checks.append('all 45 real policy/config/evaluator hashes and 90 saved rollout inputs verified')
    destination = Path(m['representation_root'])
    if not (destination / 'analysis_run.json').exists():
        runner.prepare(destination, m['representation_inputs'])
    prepared = json.loads((destination / 'analysis_run.json').read_text())
    runner.verify(prepared)
    assert prepared['n_policies'] == len(prepared['policies']) == 45
    assert prepared['seeds'] == list(range(1, 6))
    assert prepared['episodes_per_policy'] == 920
    assert prepared['train_counts'] == [368, 368] and prepared['test_counts'] == [92, 92]
    checks.append('45-policy preparation retains 920 episodes and 736/184 within-profile probe split')
    # Prepare the state-comparison metadata on an isolated fixture; real fits
    # remain pending until the actual primary probe analyses complete.
    with tempfile.TemporaryDirectory(prefix='direct_input_analysis_validation_') as temporary:
        fixture = Path(temporary) / 'prior'; fixture.mkdir()
        fake = dict(prepared, status='completed')
        (fixture / 'analysis_run.json').write_text(json.dumps(fake))
        for rel in ('eval/relative_speed_representation.py', 'eval/representation_analysis.py', 'repo_paths.py',
                    'jaxmarl/environments/coordination_grid/capability_populations.py'):
            target = fixture / 'source' / rel; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(destination / 'source' / rel, target)
        comparison = states.prepare(fixture, Path(temporary) / 'comparison')
        sm = json.loads((comparison / 'analysis_run.json').read_text())
        assert sm['n_policy_entries'] == 45 and sm['n_rounds'] == 20
    checks.append('state-comparison preparation accepts exactly the same 45 policies')
    pipeline = (root / 'eval/run_relative_speed_pipeline.py').read_text()
    assert "glob('comparison_*_seed*.png')))==15" in pipeline
    assert 'len(combined) == 7200' in (root / 'diagnostics/verify_round_state_comparison.py').read_text()
    assert 'len(table) != 7200' in (root / 'eval/round_state_prediction_comparison.py').read_text()
    checks.append('45-policy state tables require 7200 rows and 15 individual figures per condition')
    report = dict(status='passed', checks=checks, policies=60, representation_policies=45, step_penalty=.01,
        scientific_parameters=dict(rounds=20, hidden_dimension=128, profiles=46, episodes_per_profile=20,
                                   primary_probes_per_policy=29, probe_steps=1000, probe_lr=.01,
                                   umap_neighbors=919, umap_min_dist=1.0))
    output = Path(m['infrastructure_root']).parent / 'analysis_validation_t0fix_direct_input.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('manifest', type=Path)
    validate(p.parse_args().manifest)
