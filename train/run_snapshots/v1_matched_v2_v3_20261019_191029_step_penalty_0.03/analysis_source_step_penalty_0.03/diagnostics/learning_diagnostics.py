"""Shared provenance, checkpoint schedule, and fidelity rules for seed diagnostics."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

BUNDLE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('GRID_DIAGNOSTICS_REPO_ROOT', str(BUNDLE)))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUNDLE))
from repo_paths import resolve_path, resolve_record_paths
SOURCES = {
    'v1_matched': ROOT / 'train/run_snapshots/v1_matched_20261009_182845/v1_matched/source',
    'v2': ROOT / 'train/run_snapshots/counterbalanced1096_20261002_235609/v2_balanced_training/source',
    'v3': ROOT / 'train/run_snapshots/counterbalanced1096_20261008_035848_v3/v3_balanced_training/source',
}
COHORT = ROOT / 'train/seed_diagnostics/v1_matched_v2_v3/seed_selection/representative_seeds.csv'
PILOTS = (2, 0, 4, 9)
METRICS = {
    'round_success_rate': ('round_success', .000501),
    'rounds_completed': ('rounds_completed', 0),
    'entropy_t0': ('entropy_t0', .000501),
    'entropy_tge1': ('entropy_movement', .000501),
    'actor_loss': ('actor_loss', .0000501),
    'value_loss': ('value_loss', .0000501),
    'ep_return_mean': ('episode_return', .000501),
    'ep_length_mean': ('episode_steps', .05001),
    'assignment_switch_rate': ('assignment_switch_rate', .000501),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=True) + '\n')
    tmp.replace(path)


def targets():
    rows = list(csv.DictReader(COHORT.open()))
    assert len(rows) == 12 and all(r['condition'] != 'rnn_diverse_noinfluence' for r in rows)
    return rows


def target_name(row):
    return f"{row['version']}_{row['condition']}_seed{row['seed']}"


def load_target(index):
    row = targets()[index]
    for key in ('training_config', 'training_log'):
        if sha(resolve_path(row[key])) != row[key + '_sha256']:
            raise ValueError(f'Historical source changed: {key}')
    config = resolve_record_paths(json.loads(resolve_path(row['training_config']).read_text()))
    assert config['SEED'] == int(row['seed']) and config['NUM_SEEDS'] == 1
    assert config['NUM_UPDATES'] == 915 and config['TOTAL_TIMESTEPS'] == 60000000
    assert config['NUM_ENVS'] * config['NUM_STEPS'] == 65536
    return row, config


def checkpoint_updates(row):
    updates = {0, *range(1, 33), *range(48, 913, 16), 915}
    version, condition = row['version'], row['condition']
    windows = ([(8, 64)] if version == 'v1_matched' and condition.startswith('mlp') else
               [(8, 72), (88, 144)] if version == 'v1_matched' else
               [(296, 376)] if version == 'v2' and condition == 'rnn_diverse_influence' else
               [(0, 64)] if version == 'v3' and condition == 'rnn_single_influence' else [(8, 48)])
    for start, end in windows:
        updates.update(range(start, end + 1))
    return sorted(updates)


def historical(row):
    with (ROOT / '/juice6/u/jshe/emergent_partner_grid/train/training_curves/v1_matched_v2_v3_20261019_191029_step_penalty_0.03/updates.csv').open() as f:
        return {int(r['update']): r for r in csv.DictReader(f)
                if (r['version'], r['condition'], r['seed']) == (row['version'], row['condition'], row['seed'])}


def load_trainer(version, transform=None, bindings=None):
    source = SOURCES[version]
    path = source / 'baselines/IPPO/ippo_rnn_coordination_grid.py'
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(path.parent))
    original = path.read_text()
    changed = transform(original) if transform else original
    module = types.ModuleType('_diagnostic_frozen_trainer')
    module.__file__ = str(path)
    module.__dict__.update(bindings or {})
    sys.modules[module.__name__] = module
    exec(compile(changed, str(path), 'exec'), module.__dict__)
    import jaxmarl
    if Path(jaxmarl.__file__).resolve().parent != (source / 'jaxmarl').resolve():
        raise ValueError('Wrong frozen environment loaded')
    return module, original, changed
