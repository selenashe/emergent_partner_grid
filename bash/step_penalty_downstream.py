"""Freeze only the maintained five-seed analysis dependency closure."""
from pathlib import Path
import ast
import hashlib
import json
from train.experiment import ANALYSIS_SOURCE, BATCH
FILES = ('repo_paths.py', 'jaxmarl/environments/coordination_grid/capability_populations.py', 'train/__init__.py', 'train/training_log_format.py', 'eval/__init__.py', 'eval/plot_selected_assignment_dashboard.py', 'eval/analyze_assignment_behaviors.py', 'eval/plot_best_seed_assignment_behaviors.py', 'eval/relative_speed_representation.py', 'eval/run_relative_speed_pipeline.py', 'eval/run_relative_speed_representation.py', 'eval/representation_analysis.py', 'eval/plot_episode_return_dots.py', 'eval/round_state_prediction_comparison.py', 'eval/plot_overcooked_paper_match.py', 'eval/export_round_state_comparison_pngs.py', 'diagnostics/__init__.py', 'diagnostics/verify_relative_speed_results.py', 'diagnostics/verify_round_state_comparison.py', 'diagnostics/validate_t0fix_direct_input_analyses.py')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze(repo, target, batch, comparison_root, curves_root, representation_root,
           *, step_penalty=.01, artifact_suffix='_t0fix_direct_input'):
    if step_penalty != .01:
        raise ValueError('The maintained direct-input cohort uses penalty 0.01')
    target = Path(target)
    records = {}
    for rel in FILES:
        source = ANALYSIS_SOURCE / rel
        destination = target / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text().replace(BATCH, batch)
        if rel.endswith('.py'):
            ast.parse(text, filename=rel)
        destination.write_text(text)
        records[rel] = dict(original_sha256=sha(source), frozen_sha256=sha(destination))
    (target / 'analysis_adaptations.json').write_text(json.dumps(records, indent=2)+'\n')
    return records
