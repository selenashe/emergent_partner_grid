"""Compare V1_matched against V2 and V3 evaluations."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import compare_allocation_protocols as comparison
from eval.plot_episode_return_dots import plot_episode_return_dots
from repo_paths import resolve_record_paths


def compare(manifest_path, population):
    manifest = resolve_record_paths(json.loads(Path(manifest_path).read_text()))
    if manifest['experiment'] != 'V1_matched':
        raise ValueError('Expected a V1_matched manifest')
    comparison.INPUTS = {v: manifest['evaluation_inputs'][v] for v in ('v1_matched', 'v2', 'v3')}
    comparison.SEEDS = tuple(manifest['evaluation_seeds'])
    comparison.POPULATION = population
    comparison.OUT = Path(manifest['aggregate_root']) / ('familiar' if population == 'train' else 'novel')
    comparison.VERSION_COLORS = dict(comparison.VERSION_COLORS, v1_matched='#9470bb')
    comparison.VERSION_LABELS = dict(comparison.VERSION_LABELS,
                                    v1_matched='V1_matched: fixed allocation + corrected replay')
    comparison.plot_episode_return_dots = partial(
        plot_episode_return_dots, version_order=('v1_matched', 'v2', 'v3'))
    comparison.main()
    report_path = comparison.OUT / 'comparison.json'
    report = json.loads(report_path.read_text())
    change = ('V1_matched corrects pre-observation recurrent reset alignment during PPO replay and '
              'uses a fresh uniform random assignment each round in its no-influence condition.'
              if manifest.get('noinfluence_goal_selection') == 'uniform_random_each_round' else
              'V1_matched uses pre-observation recurrent reset alignment during PPO replay.')
    report['caveats'].extend([
        change,
        'V1_matched uses the same replay reset timing as V2/V3; fixed versus online allocation and other historical task differences remain.',
        'V1_matched is the maintained fixed-allocation comparison baseline.',
    ])
    report['experiment_manifest'] = str(manifest_path)
    report_path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--population', choices=('train', 'test'), required=True)
    args = parser.parse_args()
    compare(args.manifest, args.population)
