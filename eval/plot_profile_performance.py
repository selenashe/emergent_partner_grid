"""Plot every partner profile using equally weighted original/new learner seeds.

Read only small scalar fields from saved rollouts, leaving large hidden-state and
observation arrays on disk. The existing pooled summaries and figures are intact.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import runpy
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_record_paths
from eval.compare_allocation_protocols import CONDITIONS

METRICS = ('success', 'episode_return', 'episode_steps')
VERSIONS = ('v1', 'v2')
VERSION_COLORS = {'v1': '#5979ad', 'v2': '#de8845'}
CONDITION_COLORS = ('#0072b2', '#e69f00', '#009e73', '#cc79a7')
SHORT_LABELS = ('RNN\ndiverse', 'MLP\ndiverse', 'RNN\nsingle', 'RNN diverse\nno influence')


def profile_metrics(path, expected_profiles, episodes_per_profile=20, rounds_per_episode=20):
    """Group each episode by its saved RED/BLUE delay pair, then measure outcomes.

    A rollout's fixed-length scan continues after the partner episode terminates.
    Ignore that padding, but include the transition that first reports done.
    Validate profile labels against physical capability fields, so changing pool
    order or a mislabeled episode cannot silently mix different partner profiles.
    Each profile is measured within one policy before any learner-seed averaging.
    """
    path = Path(path)
    with h5py.File(path) as f:
        pool = np.asarray(f['capability_pool'])
        indices = np.asarray(f['capability_index_per_ep'])
        dones = np.asarray(f['dones'], dtype=bool)
        round_done = np.asarray(f['round_done'], dtype=bool)
        success = np.asarray(f['success'], dtype=bool)
        rewards = np.asarray(f['rewards'])
        capability = np.asarray(f['capability'])
        round_idx = np.asarray(f['round_idx'])
    if pool.ndim != 2 or pool.shape[1] != 2 or not np.issubdtype(pool.dtype, np.integer):
        raise ValueError(f'{path}: invalid capability pool')
    profiles = [tuple(map(int, row)) for row in pool]
    if len(set(profiles)) != len(profiles) or set(profiles) != set(expected_profiles):
        raise ValueError(f'{path}: missing, duplicate or unexpected partner profiles')
    if dones.ndim != 2 or dones.shape[0] != len(profiles) * episodes_per_profile:
        raise ValueError(f'{path}: unexpected number of profile episodes')
    if not dones.any(axis=1).all():
        raise ValueError(f'{path}: episode has no final done')
    if (indices.shape != (len(dones),) or not np.issubdtype(indices.dtype, np.integer)
            or (indices < 0).any() or (indices >= len(pool)).any()):
        raise ValueError(f'{path}: invalid profile index')
    if any(a.shape != dones.shape for a in (round_done, success, rewards, round_idx)):
        raise ValueError(f'{path}: inconsistent scalar rollout shapes')
    if capability.shape != dones.shape + (2,):
        raise ValueError(f'{path}: inconsistent capability shape')

    # A seed contributes one average per profile. Padding may contain arbitrary
    # rewards, success flags, or capabilities; only valid transitions matter.
    lengths = dones.argmax(axis=1) + 1
    alive = np.arange(dones.shape[1])[None, :] < lengths[:, None]
    if np.any((capability != pool[indices, None, :]) & alive[:, :, None]):
        raise ValueError(f'{path}: capability does not match episode profile')
    completed = round_done & alive
    if not np.all(completed.sum(axis=1) == rounds_per_episode):
        raise ValueError(f'{path}: incomplete or duplicated partner rounds')
    if not np.isfinite(rewards[alive]).all():
        raise ValueError(f'{path}: nonfinite valid rewards')
    # Each episode must finish exactly one instance of rounds 0..R-1. Sorting
    # the completed round indices also aligns success flags without assuming
    # that profile pools or episode rows are in a particular order.
    completed_indices = round_idx[completed].reshape(len(dones), rounds_per_episode)
    order = np.argsort(completed_indices, axis=1)
    if not np.all(np.take_along_axis(completed_indices, order, axis=1)
                  == np.arange(rounds_per_episode)[None, :]):
        raise ValueError(f'{path}: missing or repeated round index')
    round_success = np.take_along_axis(success[completed].reshape(len(dones), rounds_per_episode),
                                     order, axis=1)
    returns = np.where(alive, rewards, 0).sum(axis=1, dtype=np.float64)
    output = []
    for index, (delay_red, delay_blue) in enumerate(profiles):
        selected = indices == index
        if selected.sum() != episodes_per_profile:
            raise ValueError(f'{path}: unequal or missing episodes for profile {(delay_red, delay_blue)}')
        output.append(dict(delay_red=delay_red, delay_blue=delay_blue,
                           n_episodes=int(selected.sum()), n_rounds=int(selected.sum()) * rounds_per_episode,
                           success=float(round_success[selected].mean()),
                           episode_return=float(returns[selected].mean()),
                           episode_steps=float(lengths[selected].mean()),
                           per_round_success=round_success[selected].mean(axis=0).tolist()))
    return output


def check_pooled_summary(rows, summary, path):
    """Confirm that separating profiles reproduces the existing pooled results.

    All profiles have the same evaluation episode count, so their equally
    weighted means must recover that policy's original overall summary.
    Floating-point reward sums may differ by a few last digits only.
    """
    checks = {'success': 'round_success_rate', 'episode_return': 'mean_ep_return'}
    for metric, key in checks.items():
        measured = np.mean([r[metric] for r in rows])
        if not np.isclose(measured, summary[key], rtol=1e-6, atol=2e-5):
            raise ValueError(f'{path}: per-profile {metric} does not reproduce pooled summary')
    if sum(r['n_episodes'] for r in rows) != summary['n_episodes']:
        raise ValueError(f'{path}: episode count differs from pooled summary')
    if sum(r['n_rounds'] for r in rows) != summary['n_rounds']:
        raise ValueError(f'{path}: round count differs from pooled summary')
    curve = np.mean([r['per_round_success'] for r in rows], axis=0)
    if not np.allclose(curve, summary['per_round_success_rate'], rtol=1e-6, atol=2e-7):
        raise ValueError(f'{path}: per-profile round curve differs from pooled summary')


def aggregate_profiles(rows, seeds, profiles, conditions=tuple(CONDITIONS)):
    """Give each learner seed equal weight and show its sample standard deviation.

    Episodes are repeated observations of one trained policy. First average
    within that policy/profile, then average across policies. Do not pretend
    the 200 episodes pooled from ten policies are 200 independent policy seeds.
    Missing, duplicate or additional policy/profile rows are explicit errors.
    """
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('At least two distinct learner seeds are required')
    expected = {(v, c, s, *p) for v in VERSIONS for c in conditions for s in seeds for p in profiles}
    keys = [(r['version'], r['condition'], r['seed'], r['delay_red'], r['delay_blue']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Missing, duplicate or unexpected policy/profile row')
    lookup = dict(zip(keys, rows))
    result = []
    for version in VERSIONS:
        for condition in conditions:
            for delay_red, delay_blue in profiles:
                samples = [lookup[(version, condition, seed, delay_red, delay_blue)] for seed in seeds]
                entry = dict(version=version, condition=condition, delay_red=delay_red,
                             delay_blue=delay_blue, n_seeds=len(seeds))
                for metric in METRICS:
                    values = np.array([r[metric] for r in samples], dtype=float)
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite profile metric')
                    entry[metric + '_mean'] = float(values.mean())
                    entry[metric + '_sd'] = float(values.std(ddof=1))
                curves = np.array([r['per_round_success'] for r in samples], dtype=float)
                entry['per_round_success_mean'] = curves.mean(axis=0).tolist()
                entry['per_round_success_sd'] = curves.std(axis=0, ddof=1).tolist()
                result.append(entry)
    return result


def profile_grid(profiles):
    """Make one panel per profile; use the same metric scale across all panels."""
    columns = 4
    rows = (len(profiles) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(20, rows * 3.25), sharey=True)
    for ax, profile in zip(axes.flat, profiles):
        ax.set_title(f'Partner delays (RED, BLUE) = {profile}', fontsize=11)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    for ax in list(axes.flat)[len(profiles):]:
        ax.set_visible(False)
    return fig, axes


def save_figure(fig, directory, name):
    """Save a zoomable image and vector PDF with the same plot data."""
    for extension in ('png', 'pdf'):
        fig.savefig(directory / f'{name}.{extension}', dpi=160)
    plt.close(fig)


def plot_profiles(rows, aggregates, profiles, seeds, directory, population):
    """Draw paired v1/v2 bars in each profile panel for every model condition.

    Bar height is the ten-policy mean. Error bars show sample SD across those
    policies; faint dots expose every individual seed, including poor runs.
    Ordered delay pairs distinguish swapped RED/BLUE capabilities.
    """
    title = 'Novel' if population == 'test' else 'Familiar'
    lookup = {(r['version'], r['condition'], r['delay_red'], r['delay_blue']): r for r in aggregates}
    specs = [('success', 100, 'Round success (%)'),
             ('episode_return', 1, 'Return per 20-round episode'),
             ('episode_steps', 1, 'Steps per 20-round episode')]
    for metric, scale, label in specs:
        fig, axes = profile_grid(profiles)
        for ax, (delay_red, delay_blue) in zip(axes.flat, profiles):
            for vi, version in enumerate(VERSIONS):
                xs = np.arange(len(CONDITIONS)) + (vi - .5) * .34
                data = [lookup[(version, c, delay_red, delay_blue)] for c in CONDITIONS]
                ax.bar(xs, [d[metric + '_mean'] * scale for d in data], width=.31,
                       color=VERSION_COLORS[version], alpha=.85,
                       yerr=[d[metric + '_sd'] * scale for d in data], capsize=2, linewidth=.7)
                for ci, condition in enumerate(CONDITIONS):
                    values = [r[metric] * scale for r in rows if r['version'] == version
                              and r['condition'] == condition and r['delay_red'] == delay_red
                              and r['delay_blue'] == delay_blue]
                    ax.scatter(xs[ci] + np.linspace(-.07, .07, len(values)), values,
                               s=8, color='black', alpha=.25, zorder=3)
            ax.set_xticks(np.arange(len(CONDITIONS)), SHORT_LABELS, fontsize=8)
            if metric == 'success':
                # SD is a descriptive error bar, not a probability interval. Leave
                # headroom above 100% so those bars remain visible without clipping.
                ax.set_ylim(0, max(105, max((a['success_mean'] + a['success_sd']) * 100
                                          for a in aggregates) + 2))
            elif metric == 'episode_return':
                ax.axhline(0, color='black', lw=.6)
            else:
                ax.set_ylim(bottom=0)
        fig.suptitle(f'{title} partners: {label.lower()} for every profile', fontsize=20, y=.995)
        fig.supylabel(label, fontsize=15)
        fig.legend(handles=[Patch(facecolor=VERSION_COLORS[v], label=v) for v in VERSIONS],
                   loc='lower center', ncol=2, bbox_to_anchor=(.5, .022), fontsize=12)
        fig.text(.5, .009, f'{len(seeds)} learner seeds; bars: mean; error bars: sample SD; dots: individual seeds. '
                 'Profile values are partner wait steps; lower delay means faster movement.', ha='center', fontsize=10)
        fig.tight_layout(rect=(.02, .05, 1, .975))
        save_figure(fig, directory, f'profile_{metric}')


def plot_round_profiles(aggregates, profiles, seeds, directory, population):
    """Show success across rounds within each partner profile, separately by v1/v2.

    The same four condition colors appear in both protocol figures. Shaded
    bands are seed-to-seed sample SD, allowing a reader to see whether a mean
    improvement through the episode is consistent across trained policies.
    """
    title = 'Novel' if population == 'test' else 'Familiar'
    lookup = {(r['version'], r['condition'], r['delay_red'], r['delay_blue']): r for r in aggregates}
    for version in VERSIONS:
        fig, axes = profile_grid(profiles)
        for ax, (delay_red, delay_blue) in zip(axes.flat, profiles):
            for color, condition in zip(CONDITION_COLORS, CONDITIONS):
                entry = lookup[(version, condition, delay_red, delay_blue)]
                mean = np.array(entry['per_round_success_mean']) * 100
                sd = np.array(entry['per_round_success_sd']) * 100
                x = np.arange(1, len(mean) + 1)
                ax.plot(x, mean, color=color, lw=1.5)
                # Display the valid probability range; saved SDs remain unchanged.
                ax.fill_between(x, np.maximum(0, mean - sd), np.minimum(100, mean + sd),
                                color=color, alpha=.10, linewidth=0)
            ax.set_ylim(0, 103)
            ax.set_xlim(1, len(mean))
            ax.set_xticks([1, 5, 10, 15, 20])
        fig.suptitle(f'{title} partners, {version}: success across rounds for every profile', fontsize=20, y=.995)
        fig.supylabel('Round success (%)', fontsize=15)
        fig.supxlabel('Round within the 20-round partner episode', y=.055, fontsize=13)
        handles = [Line2D([0], [0], color=color, lw=2, label=CONDITIONS[c].replace('\n', ' '))
                   for color, c in zip(CONDITION_COLORS, CONDITIONS)]
        fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(.5, .02), fontsize=11)
        fig.text(.5, .009, f'Mean across {len(seeds)} learner seeds; shaded bands: sample SD, clipped to 0–100% for display. '
                 'Profile = (RED delay, BLUE delay).', ha='center', fontsize=10)
        fig.tight_layout(rect=(.02, .085, 1, .975))
        save_figure(fig, directory, f'profile_round_success_{version}')


def run(manifest_path, populations=('test', 'train'), output_root=None):
    """Read the explicit original/extension seed mapping and export reproducible plots.

    Validation happens before each population's plots are saved. No training,
    re-evaluation, best-seed selection or hidden-state analysis is performed.
    Saved metrics and source paths let another reader audit each plotted value.
    """
    manifest_path = Path(manifest_path).resolve()
    m = resolve_record_paths(json.loads(manifest_path.read_text()))
    if m.get('action_selection', 'categorical') != 'categorical':
        raise ValueError('Use the original categorical counterbalanced methods')
    seeds = m['evaluation_seeds']
    # Import the frozen population constants without importing the JAX package.
    population_file = Path(m['versions']['v1']['frozen_source_root']) / 'jaxmarl/environments/coordination_grid/capability_populations.py'
    pools = runpy.run_path(str(population_file))
    output_root = Path(output_root) if output_root else Path(m['aggregate_root'])
    for population in populations:
        profiles = sorted(pools['TEST_CAPABILITY_PAIRS' if population == 'test' else 'TRAIN_CAPABILITY_PAIRS'])
        rows, sources = [], []
        for version in VERSIONS:
            for condition in CONDITIONS:
                for seed in seeds:
                    directory = Path(m['evaluation_inputs'][version][str(seed)])
                    rollout = directory / f'{condition}_seed{seed}_{population}.h5'
                    summary_path = directory / f'{condition}_seed{seed}_summary.json'
                    print(f'[profile metrics] {population} {version} {condition} seed {seed}', flush=True)
                    measured = profile_metrics(rollout, profiles)
                    check_pooled_summary(measured, json.loads(summary_path.read_text())[population], rollout)
                    rows.extend(dict(r, version=version, condition=condition, seed=seed) for r in measured)
                    sources.extend((str(rollout), str(summary_path)))
        aggregates = aggregate_profiles(rows, seeds, profiles)
        directory = output_root / ('novel' if population == 'test' else 'familiar') / 'per_profile'
        directory.mkdir(parents=True, exist_ok=True)
        report = dict(population=population, seeds=seeds, profile_order=[list(p) for p in profiles],
                      profile_definition='(RED movement delay, BLUE movement delay); partner moves every delay+1 steps',
                      episodes_per_profile_per_seed=20, rounds_per_episode=20,
                      weighting='Equal learner-seed weights; first summarize each policy/profile over its episodes',
                      error_bars='Sample standard deviation across learner seeds, ddof=1',
                      manifest=str(manifest_path), sources=sources,
                      code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      aggregates=aggregates)
        (directory / 'profile_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
        fieldnames = ('version', 'condition', 'seed', 'delay_red', 'delay_blue',
                      'n_episodes', 'n_rounds', *METRICS)
        with (directory / 'profile_per_seed.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader(); writer.writerows(rows)
        # Save round-level inputs as well, making every line and SD auditable.
        (directory / 'profile_per_seed_rounds.json').write_text(json.dumps(rows, indent=2) + '\n')
        plot_profiles(rows, aggregates, profiles, seeds, directory, population)
        plot_round_profiles(aggregates, profiles, seeds, directory, population)
        print(f'[saved] {len(profiles)} profiles, {len(seeds)} seeds, {directory}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension-manifest', required=True, type=Path)
    parser.add_argument('--population', choices=('test', 'train', 'both'), default='both')
    parser.add_argument('--out-dir', type=Path, help='Alternative root; writes novel/familiar subfolders')
    args = parser.parse_args()
    run(args.extension_manifest, ('test', 'train') if args.population == 'both' else (args.population,), args.out_dir)
