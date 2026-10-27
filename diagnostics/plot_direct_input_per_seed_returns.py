"""Recreate the reference nine-panel per-seed return figure for the new cohort."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.plot_per_seed_training_returns import CONDITIONS, VERSIONS, make_figure
from train.plot_protocol_training_curves import trailing_mean

SUFFIX = '_t0fix_direct_input'
WINDOW = 46


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--out-dir', type=Path)
    args = p.parse_args()
    m = json.loads(args.manifest.read_text())
    seeds = tuple(m['seeds'])
    assert seeds == (1, 2, 3, 4, 5) and m['status'] == 'completed'
    source = Path(m['curves_root']) / ('updates' + SUFFIX + '.csv')
    out = args.out_dir or Path(m['curves_root']) / ('per_seed_returns' + SUFFIX)
    runs = {}
    wanted = {c for c, _ in CONDITIONS}
    with source.open() as f:
        for r in csv.DictReader(f):
            if r['cohort'] != 't0fix_direct_input' or r['condition'] not in wanted:
                continue
            key = (r['version'], r['condition'], int(r['seed']))
            value = float(r['episode_return']) if float(r['episode_steps']) > 0 else np.nan
            runs.setdefault(key, []).append((int(float(r['update'])), int(float(r['env_steps'])), value))
    assert set(runs) == {(v, c, s) for v in VERSIONS for c, _ in CONDITIONS for s in seeds}
    curves, rows, validation = {}, [], []
    for key, records in sorted(runs.items()):
        raw = np.array(records, dtype=float)
        assert raw.shape == (915, 3) and not np.isinf(raw).any()
        assert np.array_equal(raw[:, 0], np.arange(1, 916))
        assert np.array_equal(raw[:, 1], raw[:, 0] * 65536)
        assert raw[-1, 1] == m['effective_timesteps_per_policy'] == 59965440
        smoothed = trailing_mean(raw[:, 2], WINDOW)
        # Independent direct-window calculation checks masking and the smoothing
        # window, including the initial batches with no completed episode.
        reference = []
        for i in range(len(raw)):
            values = raw[max(0, i-WINDOW+1):i+1, 2]
            values = values[np.isfinite(values)]
            reference.append(values.mean() if len(values) else np.nan)
        np.testing.assert_allclose(smoothed, reference, atol=1e-10, rtol=0, equal_nan=True)
        curves[key] = np.column_stack((raw[:, :2], smoothed))
        version, condition, seed = key
        for (update, steps, value), mean in zip(raw, smoothed):
            rows.append(dict(version=version, condition=condition, seed=seed, update=int(update),
                             env_steps=int(steps), episode_return=value, smoothed_episode_return=mean))
        validation.append(dict(version=version, condition=condition, seed=seed, updates=915,
                               missing_episode_updates=int(np.isnan(raw[:, 2]).sum()),
                               final_env_steps=int(raw[-1, 1]), final_smoothed_return=float(smoothed[-1])))
    out.mkdir(parents=True, exist_ok=True)
    prefix = 'training_episode_return_per_seed' + SUFFIX
    with plt.rc_context({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig = make_figure(curves, WINDOW, 65536, seeds=seeds,
                          title='Training episode return · t0 fix + direct input · every seed')
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(out / (prefix + '.' + ext), dpi=200, facecolor='white')
        plt.close(fig)
    with (out / ('per_seed_returns' + SUFFIX + '.csv')).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = dict(manifest=dict(path=str(args.manifest.resolve()), sha256=sha(args.manifest)),
        source=dict(path=str(source), sha256=sha(source)), script_sha256=sha(__file__),
        plotter_sha256=sha(ROOT / 'train/plot_per_seed_training_returns.py'),
        cohort='t0fix_direct_input', seeds=list(seeds), panels=9, curves_per_panel=5,
        conditions=[c for c, _ in CONDITIONS], excluded_conditions=['rnn_diverse_noinfluence'],
        window_updates=WINDOW, window_steps=WINDOW*65536,
        smoothing='Trailing arithmetic mean over 46 updates; finite completed-episode means only; early windows use available updates.',
        validation='All 45 histories contain 915 ordered updates; all smoothed values independently verified to 1e-10.',
        runs=validation)
    (out / ('provenance' + SUFFIX + '.json')).write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(dict(panels=9, curves_per_panel=5, policies=len(curves), records=len(rows),
                          smoothing_steps=WINDOW*65536, output=str(out / (prefix + '.png'))), indent=2))


if __name__ == '__main__': main()
