"""Freeze the maintained analyses for a five-seed step-penalty replication.

Edits here affect analysis cohort sizes, output locations, and reward accounting;
policy/environment/trainer sources are copied separately without modification.
"""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import re
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(repo, target, batch, comparison_root, curves_root, representation_root):
    repo, target = Path(repo), Path(target)
    records = {}
    for folder in ['eval', 'train', 'diagnostics']:
        for source in (repo / folder).glob('*.py'):
            destination = target / folder / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    for relative in ['repo_paths.py', 'jaxmarl/environments/coordination_grid/capability_populations.py']:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / relative, destination)
    for p in target.rglob('*.py'):
        relative = str(p.relative_to(target))
        before = p.read_text()
        after = before
        # Explicit cohort references only: never change plot dimensions, rounds,
        # feature-control seeds, or the 30 probe models / 90 rows PER policy.
        for old, new in [
            ('range(1, 11)', 'range(1, 6)'), ('range(1,11)', 'range(1,6)'),
            ('len(group) != 10', 'len(group) != 5'),
            ('integers(0, 10, size=(10000, 10))', 'integers(0, 5, size=(10000, 5))'),
            ('n_seeds=10', 'n_seeds=5'), ('n_policies=90', 'n_policies=45'),
            ('n_policy_entries=90', 'n_policy_entries=45'),
            ('n_episodes=82800', 'n_episodes=41400'),
            ('n_primary_probes=2610', 'n_primary_probes=1305'),
            ('n_umap_embeddings=90', 'n_umap_embeddings=45'),
            ('policies=90', 'policies=45'),
            ("len(manifest['policies'])==90", "len(manifest['policies'])==45"),
            ('len(prior["policies"]) != 90', 'len(prior["policies"]) != 45'),
            ('len(identities) != 90', 'len(identities) != 45'),
            ('len(checks) == 90', 'len(checks) == 45'),
            ("root/'probe_per_seed.csv'))==8100", "root/'probe_per_seed.csv'))==4050"),
            ("root/'umap_all_seeds.csv'))==82800", "root/'umap_all_seeds.csv'))==41400"),
            ("glob('comparison_*_seed*.png')))==30", "glob('comparison_*_seed*.png')))==15"),
            ('valid_count != 10', 'valid_count != 5'),
            ('len(finite) == 10', 'len(finite) == 5'),
            ('len(ax.lines) != 10', 'len(ax.lines) != 5'),
            ('curves_per_panel=10', 'curves_per_panel=5'),
            ('"curves_per_panel": 10', '"curves_per_panel": 5'),
            ('for n in (10, 5)', 'for n in (5,)'),
            ('best_of_10', 'best_of_5'), ('all_10_seeds', 'all_5_seeds'),
            ('best-of-ten', 'best-of-five'), ('best-of-10', 'best-of-5'),
            ('all-ten', 'all-five'), ('all ten', 'all five'), ('All ten', 'All five'),
            ('across ten', 'across five'), ('ten independent', 'five independent'),
            ('ten policy', 'five policy'), ('ten seeds', 'five seeds'), ('ten curves', 'five curves'),
            ('10 training seeds', '5 training seeds'), ('10 seeds', '5 seeds'),
            ('seeds 1–10', 'seeds 1–5'), ('seeds 1-10', 'seeds 1-5'),
            ('90 fixed policies', '45 fixed policies'), ('(90 policies)', '(45 policies)'),
            ('all thirty individual-seed', 'all fifteen individual-seed'),
            ('all 90 base policies', 'all 45 base policies'),
            ('all 90 state comparisons', 'all 45 state comparisons'),
            ('90 individual-seed PNGs', '45 individual-seed PNGs'),
            ('{done}/90', '{done}/45'), ('ncol=10, frameon=False', 'ncol=5, frameon=False'),
            ('4,400 episodes', '2,200 episodes'), ('4,800 episodes', '2,400 episodes'),
            ('completed}/240', 'completed}/120'),
        ]:
            after = after.replace(old, new)
        if relative == 'train/plot_protocol_training_curves.py':
            after = re.sub(r'BATCHES = \(.*?\n\)', 'BATCHES = (' + repr(batch) + ',)', after, count=1, flags=re.S)
        # Float32 episode accumulation can differ from the float64 ledger by
        # about 8.2e-4 at the new penalty over 2,000 ticks. Tolerate 0.003 in
        # episode RETURN comparisons only; keep counts, success, per-tick and
        # per-round checks at their original precision.
        if relative in ['eval/plot_overcooked_paper_match.py',
                        'eval/plot_best_seed_assignment_behaviors.py',
                        'eval/plot_selected_assignment_dashboard.py']:
            after = after.replace('atol=3e-5, rtol=0', 'atol=3e-3, rtol=0')
        if relative == 'eval/plot_profile_performance.py':
            after = after.replace('rtol=1e-6, atol=2e-5',
                                  "rtol=1e-6, atol=(3e-3 if key == 'mean_ep_return' else 2e-5)")
        if relative == 'eval/compare_allocation_protocols.py':
            after = after.replace('1.01 * 20', '1.03 * 20').replace('/ .01', '/ .03')
            after = after.replace('abs(mean_episode_steps - inferred_steps) < .02',
                                  'abs(mean_episode_steps - inferred_steps) < .11')
        if relative == 'eval/plot_round_return_progression.py':
            after = after.replace('1., -.01', '1., -.03').replace('1.01 * success', '1.03 * success').replace(' - .01 * steps', ' - .03 * steps')
            after = after.replace('summary["mean_ep_return"], atol=2e-4', 'summary["mean_ep_return"], atol=3e-3')
        if relative == 'train/plot_selected_training_returns.py':
            after = after.replace('ax.set_ylim(-21, 19)', 'ax.set_ylim(-61, 21)')
        if relative == 'train/plot_per_seed_training_returns.py':
            after = after.replace('All 90 recomputed curves', 'All 45 recomputed curves')
        if relative == 'eval/optimal_training_return.py':
            after = after.replace('.01', '.03').replace('0.01', '0.03')
            after = after.replace('else -1.0', 'else -3.0').replace('or -1 for a timeout', 'or -3 for a timeout')
        if relative == 'diagnostics/validate_optimal_training_return.py':
            after = after.replace('step_penalty=.01', 'step_penalty=.03')
        if relative in ['eval/optimal_training_return.py', 'eval/plot_training_return_ceiling.py']:
            after = after.replace('ROOT / "eval" / directory / "v1_matched_v2_v3"',
                                  f'Path({str(repo)!r}) / "eval" / directory / {batch!r}')
        if relative == 'diagnostics/validate_optimal_training_return.py':
            after = after.replace('ROOT / "eval" / directory / "v1_matched_v2_v3"',
                                  f'Path({str(repo)!r}) / "eval" / directory / {batch!r}')
        # All default output/read paths point to the new isolated batch.
        replacements = {
            'eval/protocol_comparison/v1_matched_20261009_182845': str(comparison_root),
            'train/training_curves/v1_matched_v2_v3_all_5_seeds': str(curves_root),
            'eval/representation_results/relative_speed_v1_matched_v2_v3_20261013_all_5_seeds': str(representation_root),
        }
        for old, new in replacements.items(): after = after.replace(old, new)
        after = after.replace('all_5_seeds/', 'all_5_seeds_step_penalty_0.03/')
        ast.parse(after, filename=relative)
        p.write_text(after)
        records[relative] = {'original_sha256': sha(repo / relative), 'frozen_sha256': sha(p)}
        if before != after:
            diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                               fromfile=relative, tofile=relative))
            patch = target / 'analysis_adaptations' / (relative + '.diff')
            patch.parent.mkdir(parents=True, exist_ok=True)
            patch.write_text(diff)
    (target / 'analysis_adaptations.json').write_text(json.dumps(records, indent=2) + '\n')
    return records
