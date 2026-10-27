"""Reproduce the three-row selected-seed dashboard with all four conditions."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.plot_best_seed_assignment_behaviors import (
    DEFAULT_MANIFEST, DEFAULT_OUT, DEFAULT_SELECTION, METRICS, STYLES,
    dashboard, load_manifests, read_policy, require, round_metrics,
    sha256, verified_selection,
)


def plot(manifest_path, selection_path, out):
    records = load_manifests(manifest_path)
    saved, selection_source = verified_selection(selection_path, records)
    manifest = records[0][0]
    out.mkdir(parents=True, exist_ok=True)
    groups, selected, sources = {}, {}, []
    for row in saved['results']:
        version, condition, seed = row['version'], row['condition'], row['seed']
        directory = Path(manifest['evaluation_inputs'][version][str(seed)])
        data, provenance = read_policy(directory, version, condition, seed, 'test', out / 'cache')
        require(Path(provenance['source']['path']).resolve() == Path(row['evaluation_rollouts']).resolve(),
                'Selected rollout path differs')
        require(provenance['source']['summary_sha256'] == row['evaluation_summary_sha256'],
                'Selected evaluation summary changed')
        for field in ('dones', 'round_done', 'round_idx'):
            require(provenance['scalar_datasets_sha256'][field] == row['scalar_datasets_sha256'][field],
                    'Selected rollout changed: ' + field)
        require(len(data['profile']) == 440, 'Require all 440 held-out episodes')
        require(np.isclose(provenance['saved_episode_return'], row['mean'], atol=3e-3, rtol=0),
                'Held-out return differs from the selected-seed return figure')
        groups[version, condition] = round_metrics(data)
        selected[version, condition] = row
        sources.append(dict(version=version, condition=condition, seed=seed, **provenance))
    require(len(groups) == 12, 'Require one policy for each of three versions and four conditions')
    metrics = (METRICS[0], METRICS[2], METRICS[3])
    name = 'assignment_behavior_three_metrics_best_training_seeds'
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        dashboard(groups, selected, out, None, styles=STYLES, metrics=metrics, name=name)
    report = dict(selection_source=selection_source, population='test', episodes_per_policy=440,
                  metrics=[metric[0] for metric in metrics],
                  included_conditions=[condition for condition, _, _ in STYLES],
                  selected_seeds=[dict(version=v, condition=c, seed=row['seed'])
                                  for (v, c), row in selected.items()],
                  sources=sources, source_sha256=sha256(Path(__file__).read_bytes()))
    (out / (name + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(out / (name + '.png'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--selection', type=Path, default=DEFAULT_SELECTION)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot(args.manifest, args.selection, args.out_dir)
