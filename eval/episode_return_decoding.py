"""Decision-time capability decoders for the six training-selected RNNs.

Reconstruct the old audit's five round cutoffs and shuffled-label control from
current saved states. Policies stay frozen; only diagnostic linear probes fit.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from eval.episode_return_analysis import BUILD, identity, key, require, sha, write
from eval.relative_speed_representation import (
    class_metrics, masks, original, relative_classes, train_probes,
)

ROUNDS = np.array([1, 2, 5, 10, 20, 20])


def run_decoding(root=BUILD):
    import torch
    from eval.episode_return_reports import export, report, table
    import matplotlib.pyplot as plt

    torch.set_num_threads(1)
    root = Path(root)
    run_file = root / 'run.json'
    if not run_file.exists():
        run_file = root / 't0fix_direct_input/analysis_run.json'
    policies = json.loads(run_file.read_text())['policies']
    chosen = [p for p in policies if p['selected'] and p['condition'] in
              ('rnn_diverse_influence', 'rnn_single_influence')]
    require(len(chosen) == 6, 'Need six training-selected RNN policies')
    out = root / 'lowest_decile_analysis/math_audit/decision_decoding'
    out.mkdir(parents=True, exist_ok=True)
    split = original.make_probe_split(0)
    rows, confusion_rows = [], []
    for record in chosen:
        features, pairs, repetitions, populations = [], [], [], []
        # v2's first controllable allocation is t1; v1/v3 choose at t0.
        offset = int(record['version'] == 'v2')
        for population in ('train', 'test'):
            with h5py.File(record[population + '_file']) as f:
                cap = f['capability'][:, 0, :].astype(int)
                done = f['dones'][:].astype(bool)
                alive = np.arange(done.shape[1])[None, :] <= done.argmax(1)[:, None]
                phase = f['is_t0'][:].astype(bool) & alive
                require(np.all(phase.sum(1) == 20), 'Need 20 real round starts')
                starts = np.nonzero(phase)[1].reshape(len(cap), 20)
                times = starts[:, ROUNDS - 1] + offset
                values = np.empty((len(cap), 6, 128), np.float32)
                for lo in range(0, len(cap), 16):
                    hi = min(lo + 16, len(cap))
                    hidden = f['hidden_state'][lo:hi, :int(times[lo:hi].max()) + 1]
                    values[lo:hi] = hidden[np.arange(hi-lo)[:, None], times[lo:hi]]
                counts = {}
                rep = []
                for pair in map(tuple, cap):
                    rep.append(counts.get(pair, 0)); counts[pair] = rep[-1] + 1
                require(set(counts.values()) == {20}, 'Need 20 repetitions per profile')
                features.append(values.transpose(1, 0, 2))
                pairs.append(cap); repetitions.extend(rep); populations.extend([population]*len(cap))
        features = np.concatenate(features, axis=1)
        pairs = np.concatenate(pairs); repetitions = np.asarray(repetitions)
        train, test = masks(repetitions, pairs, split)
        require((train.sum(), test.sum()) == (736, 184), 'Unexpected decoder partition')
        truth = np.broadcast_to(relative_classes(pairs), (6, 920)).copy()
        truth[-1] = np.random.default_rng(20000).permutation(truth[-1])
        predictions, probabilities, model = train_probes(features, truth, train, [10000]*5+[20000])
        np.savez_compressed(out/(key(record)+'.npz'), features=features, truth=truth,
            predictions=predictions, probabilities=probabilities, train=train, test=test,
            rounds=ROUNDS, pairs=pairs, repetitions=repetitions, population=populations,
            round_tick=offset, **model)
        for i, rd in enumerate(ROUNDS):
            for population in ('all', 'train', 'test'):
                mask = test & (True if population == 'all' else np.asarray(populations) == population)
                metrics, confusion = class_metrics(truth[i, mask], predictions[i, mask])
                common = dict(identity(record), round=int(rd), round_tick=offset,
                    kind='shuffled_labels' if i == 5 else 'primary', population=population,
                    held_out_episodes=int(mask.sum()))
                rows.append(dict(common, **metrics))
                for a in range(2):
                    for b in range(2):
                        confusion_rows.append(dict(common, true_class=a, predicted_class=b,
                                                   count=int(confusion[a, b])))
        print('[decision decoder]', key(record), flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(out/'scores.csv', index=False)
    pd.DataFrame(confusion_rows).to_csv(out/'confusion_counts.csv', index=False)
    write(out/'probe_split.json', split)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained', sharey=True)
    for ax, version in zip(axes, ('v1_matched', 'v2', 'v3')):
        for condition in ('rnn_diverse_influence', 'rnn_single_influence'):
            d = scores[(scores.version == version) & (scores.condition == condition) &
                       (scores.population == 'all') & (scores.kind == 'primary')]
            ax.plot(d['round'], d.balanced_accuracy, 'o-', label=condition.replace('rnn_', '').replace('_influence', ''))
        ax.axhline(.5, color='gray', ls='--'); ax.set_title(version); ax.set_xlabel('Round')
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel('Held-out episode balanced accuracy'); axes[0].legend()
    export(fig, out/'decision_decoding')
    text = '# Decision-time relative-speed decoding\n\n'
    text += ('Six RNN policies selected by final training return, with separate affine 128→2 probes '
        'at rounds 1, 2, 5, 10 and 20, plus a round-20 shuffled-label control. '
        'The target is red-faster (0) versus blue-faster (1). Features are original saved '
        'post-observation GRU states at t0 for v1_matched/v3 and t1 for v2, when the '
        'initial allocation is controllable. No trajectory averaging is used.\n\n'
        'Every one of the 46 capability profiles contributes 16 fitting episodes and four '
        'held-out episodes: 736 fit and 184 test per policy. The shared split uses repetition '
        'indices from source order and numpy.default_rng(0). This measures held-out-episode '
        'decoding; it does not test transfer to unseen capability profiles.\n\n'
        'Each independent class-balanced softmax probe uses full-batch Adam for 1000 steps, '
        'learning rate 0.01, no weight decay, seed 10000; the shuffled control permutes '
        'labels with seed 20000 and initializes its probe with seed 20000. Class weights '
        'use fitting labels only. The fitting helper is shared with the current relative-speed '
        'analysis. The historical producer was unavailable; methods were reconstructed '
        'from its saved arrays and the maintained fitting helper. Policies are not retrained.\n\n'
        '![Decision decoding](decision_decoding.png)\n\n')
    text += table(scores[scores.population == 'all'])
    report(out/'report.md', text)
    write(out/'validation.json', dict(status='passed', policies=6, fitted_probes=36,
        fit_episodes_per_policy=736, test_episodes_per_policy=184,
        selected_by='final unsmoothed training return', profile_holdout=False,
        producer_sha256=sha(Path(__file__)), policy_retraining=False))
    parent = out.parent/'report.md'
    parent_text = parent.read_text()
    link = '[Decision-time capability decoders](decision_decoding/report.md)'
    if link not in parent_text:
        report(parent, parent_text+'\n\n'+link+': five round cutoffs and a shuffled-label control for each of six selected RNNs.\n')
    return scores
