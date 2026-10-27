"""Record the full frozen PPO run without shortening or enlarging scan outputs.

All training transitions are saved losslessly, including original observations,
actions, rewards, old likelihoods/values, advantages, targets, and reset masks.
Every update has exact boundary environment/RNG/queue state. Selected updates
also retain the entire learner, Adam, GRU, environment and PRNG state.
"""
import argparse
import difflib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
BUNDLE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('GRID_DIAGNOSTICS_REPO_ROOT', str(BUNDLE)))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUNDLE))
from diagnostics.learning_diagnostics import (SOURCES, METRICS, load_target, target_name,
    checkpoint_updates, historical, load_trainer, sha, write_json)
from repo_paths import prepare_artifact_path, link_artifact, artifact_root


def instrument(source):
    before = '            initial_hstate = runner_state[5]\n'
    after = '            return runner_state, metric\n'
    if source.count(before) != 1 or source.count(after) != 1:
        raise ValueError('Unexpected frozen trainer structure')
    # Only aliases and an observation callback: no RNG, arithmetic, batching,
    # returned tensors, optimizer operations, or scan lengths are changed.
    return source.replace(before, '            _audit_before = runner_state\n' + before).replace(
        after, '            jax.debug.callback(_audit_callback, _audit_before, runner_state,\n'
               '                traj_batch, advantages, targets, last_val, loss_info, metric)\n' + after)


def run(args):
    import numpy as np
    import jax
    import wandb
    from flax import serialization
    from jax.tree_util import tree_leaves
    row, config = load_target(args.target)
    out = args.run_dir / ('pilots' if args.pilot else 'runs') / target_name(row)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'status.json').exists():
        raise FileExistsError(f'Refusing to overwrite a previous run: {out}')
    state = dict(state='starting', target=row, stop_update=args.stop_update,
                 job_id=os.environ.get('SLURM_JOB_ID'), host=socket.gethostname(),
                 started_unix=time.time(), scratch_root=str(artifact_root()), completed_updates=[])
    write_json(out / 'status.json', state)
    ref = historical(row)
    checkpoints = set(checkpoint_updates(row))
    lock = threading.Lock()
    marker = 'DIAGNOSTIC_PREFIX_SAVED'
    records = {}
    files = []
    started = time.monotonic()
    fidelity = dict(passed=True, checked_updates=0, scalar_checks=0, mismatches=[])

    def save_bytes(logical, value):
        physical = prepare_artifact_path(logical)
        tmp = physical.with_suffix(physical.suffix + '.tmp')
        tmp.write_bytes(serialization.to_bytes(value))
        tmp.replace(physical)
        link_artifact(logical)
        return dict(path=str(logical), physical=str(physical), bytes=physical.stat().st_size, sha256=sha(physical))

    def save_npz(logical, arrays):
        physical = prepare_artifact_path(logical)
        with physical.with_suffix('.npz.tmp').open('wb') as f:
            np.savez_compressed(f, **arrays)
        physical.with_suffix('.npz.tmp').replace(physical)
        link_artifact(logical)
        return dict(path=str(logical), physical=str(physical), bytes=physical.stat().st_size, sha256=sha(physical))

    def snapshot(u, runner):
        entry = save_bytes(out / 'checkpoints' / f'update{u:04d}.msgpack', runner)
        entry.update(kind='full_runner', update=u, optimizer_steps=int(runner[0].step))
        # A separate params file simplifies evaluation without discarding the
        # resumable state; both are on scratch.
        files.append(entry)
        files.append(dict(kind='params', update=u, **save_bytes(
            out / 'checkpoints' / f'update{u:04d}_params.msgpack', runner[0].params)))

    def callback(before, after, traj, advantages, targets_, bootstrap, losses, metrics):
        with lock:
            u = int(metrics['update_step'])
            flat = {k: float(v) for k, v in metrics.items()}
            records[u] = flat
            mismatches = []
            for key, (column, tolerance) in METRICS.items():
                if key not in flat or column not in ref[u]:
                    continue
                expected = float(ref[u][column])
                if not np.isfinite(expected):
                    continue
                fidelity['scalar_checks'] += 1
                error = abs(flat[key] - expected)
                if not np.isfinite(error) or error > tolerance:
                    mismatches.append(dict(update=u, metric=key, replay=flat[key], historical=expected,
                                           absolute_error=error, tolerance=tolerance))
            fidelity['checked_updates'] += 1
            fidelity['mismatches'].extend(mismatches)
            fidelity['passed'] = not fidelity['mismatches']
            if u == 1:
                snapshot(0, before)
            if u in checkpoints:
                snapshot(u, after)
            arrays = {}
            def flatten(prefix, value):
                if isinstance(value, dict):
                    for k, v in value.items():
                        flatten(prefix + '/' + k if prefix else k, v)
                else:
                    a = np.asarray(value)
                    # Grid, allocation one-hot, and is_t0 are binary in these
                    # frozen environments. Check before lossless bool storage.
                    if prefix.startswith('obs/'):
                        if not ((a == 0) | (a == 1)).all():
                            raise ValueError(f'Nonbinary observation: {prefix}')
                        a = a.astype(bool)
                    arrays[prefix] = a
            flatten('', traj._asdict())
            arrays.update(advantages=np.asarray(advantages), targets=np.asarray(targets_),
                          bootstrap_value=np.asarray(bootstrap), initial_hstate=np.asarray(before[5]))
            # Recover the exact per-transition scheduled episode IDs on the
            # host using the original stable worker-order dispatch algorithm.
            ids = np.asarray(before[7].slot_episode).copy()
            next_id = int(before[7].next_episode)
            episode_ids = np.empty(traj.done.shape, np.int32)
            for tick in range(len(episode_ids)):
                episode_ids[tick] = ids
                ended = np.asarray(traj.done[tick], bool)
                n = int(ended.sum())
                ids[ended] = next_id + np.arange(n, dtype=np.int32)
                next_id += n
            if not np.array_equal(ids, after[7].slot_episode) or next_id != int(after[7].next_episode):
                raise ValueError('Per-transition queue reconstruction failed')
            arrays['scheduled_episode_id'] = episode_ids
            # Compact round-start/completion ledger for every rollout,
            # including rounds that cross update boundaries.
            starts = np.asarray(traj.pre_step_time) == 0
            ends = np.asarray(traj.info['round_done'], bool)
            event_tick, event_worker = np.nonzero(starts | ends)
            for key, value in dict(tick=event_tick, worker=event_worker,
                    started=starts[event_tick,event_worker], completed=ends[event_tick,event_worker],
                    scheduled_episode_id=episode_ids[event_tick,event_worker],
                    round_idx=np.asarray(traj.info['round_idx'])[event_tick,event_worker],
                    layout_idx=np.asarray(traj.info['layout_idx'])[event_tick,event_worker],
                    capability=np.asarray(traj.info['capability'])[event_tick,event_worker],
                    round_time=np.asarray(traj.info['round_time'])[event_tick,event_worker],
                    success=np.asarray(traj.info['success'])[event_tick,event_worker],
                    assignment=np.asarray(traj.info['partner_assignment'])[event_tick,event_worker]).items():
                arrays['round_events/' + key] = value
            arrays['initial_rng'] = np.asarray(before[6])
            arrays['epoch_minibatch_losses'] = np.asarray(losses[0])
            for k, a in zip(('value_loss', 'actor_loss', 'entropy', 'entropy_t0', 'entropy_movement'), losses[1]):
                arrays['sgd/' + k] = np.asarray(a)
            files.append(dict(kind='training_transitions', update=u,
                **save_npz(out / 'training' / f'update{u:04d}.npz', arrays)))
            # Exact boundary states recover terminal poses, partner delay
            # counters, random v2 assignments, and round resets for animation.
            boundary = dict(before_env=before[1], after_env=after[1],
                before_rng=before[6], after_rng=after[6], before_queue=before[7], after_queue=after[7])
            files.append(dict(kind='environment_boundaries', update=u,
                **save_bytes(out / 'training' / f'update{u:04d}_boundaries.msgpack', boundary)))
            write_json(out / 'metrics.json', records)
            write_json(out / 'fidelity.json', fidelity)
            state.update(state='recording', last_update=u, completed_updates=sorted(records),
                         elapsed_seconds=time.monotonic() - started,
                         historical_metrics_match=fidelity['passed'], bytes_saved=sum(f['bytes'] for f in files))
            write_json(out / 'artifacts.json', files)
            write_json(out / 'status.json', state)
            print(f'[diagnostic saved] u={u} history_match={not mismatches} bytes={state["bytes_saved"]}', flush=True)
            if mismatches:
                raise RuntimeError('HISTORICAL_FIDELITY_MISMATCH: ' + json.dumps(mismatches[:3]))
            if u == args.stop_update and u < 915:
                raise RuntimeError(marker)

    trainer, original, changed = load_trainer(row['version'], instrument, {'_audit_callback': callback})
    (out / 'instrumentation.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), changed.splitlines(True), fromfile='frozen', tofile='recorder')))
    source = SOURCES[row['version']]
    source_path = source / 'baselines/IPPO/ippo_rnn_coordination_grid.py'
    layouts = sorted(Path(config['ENV_KWARGS']['layouts_dir']).glob('*.json'))
    write_json(out / 'provenance.json', dict(target=row, runner_sha256=sha(__file__),
        trainer_sha256=sha(source_path), environment_sha256=sha(source / 'jaxmarl/environments/coordination_grid/coordination_grid.py'),
        schedule_sha256=sha(config['COUNTERBALANCED_SCHEDULE_PATH']),
        devices=[d.device_kind for d in jax.devices()],
        packages={k: importlib.metadata.version(k) for k in ('jax', 'jaxlib', 'flax', 'optax', 'distrax', 'numpy')},
        deterministic_environment={k: os.environ.get(k) for k in ('TF_CUDNN_USE_AUTOTUNE', 'TF_CUDNN_DETERMINISTIC', 'XLA_FLAGS')},
        checkpoint_updates=sorted(checkpoints), original_scan_updates=915, config=config,
        layout_catalog=[dict(index=i, path=str(p), sha256=sha(p)) for i, p in enumerate(layouts)],
        trace_order='update ascending, rollout tick ascending, worker ID ascending; also preserves each worker chronological order',
        caveat='New reproduction: rounded historical agreement alone does not prove unsaved original trajectories identical.'))
    wandb.init(mode='disabled', project=config['PROJECT'], config=config)
    try:
        result = jax.vmap(jax.jit(trainer.make_train(config)))(
            jax.random.split(jax.random.PRNGKey(config['SEED']), config['NUM_SEEDS']))
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, result)
        jax.effects_barrier()
    except Exception as exc:
        if not (args.stop_update < 915 and args.stop_update in records and fidelity['passed'] and marker in str(exc)):
            state.update(state='failed', error=str(exc), traceback=traceback.format_exc())
            write_json(out / 'status.json', state)
            raise
    if sorted(records) != list(range(1, args.stop_update + 1)) or not fidelity['passed']:
        raise ValueError('Incomplete or historically inconsistent replay')
    if args.stop_update == 915:
        from jaxmarl.wrappers.baselines import load_params
        params = jax.tree_util.tree_map(lambda a: np.asarray(a)[0], result['runner_state'][0].params)
        original_params = load_params(config['SAVE_PARAMS_PATH'])
        leaves, treedef = jax.tree_util.tree_flatten(params)
        other, otherdef = jax.tree_util.tree_flatten(original_params)
        comparison = dict(structure_equal=treedef == otherdef,
            tensors_equal=treedef == otherdef and all(np.array_equal(a, b) for a, b in zip(leaves, other)),
            max_absolute_difference=max(float(np.max(np.abs(a-b))) for a,b in zip(leaves, other)) if treedef == otherdef else None)
        write_json(out / 'final_parameter_comparison.json', comparison)
        if not comparison['tensors_equal']:
            state.update(state='final_parameters_differ', final_parameter_comparison=comparison)
            write_json(out / 'status.json', state)
            raise ValueError('Final parameters differ from historical checkpoint')
    state.update(state='complete', elapsed_seconds=time.monotonic() - started)
    write_json(out / 'status.json', state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=int, required=True, choices=range(12))
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--stop-update', type=int, default=915)
    a = p.parse_args()
    if not 1 <= a.stop_update <= 915:
        p.error('stop-update must be 1..915')
    run(a)

if __name__ == '__main__':
    main()
