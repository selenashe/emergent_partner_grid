"""Fixed-packet behavioral timeline and paired assignment interventions.

Evaluation is a separate process and never consumes the learner's random keys.
Complete 20-round episodes retain recurrent history. Traces include every valid
step's pre/post positions, partner clocks, assignments, sampled/applied actions,
rewards, round boundaries, and initial state for subsequent GIF reconstruction.
"""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS', '1')
BUNDLE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('GRID_DIAGNOSTICS_REPO_ROOT', str(BUNDLE)))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUNDLE))
from diagnostics.learning_diagnostics import load_target, target_name, load_trainer, write_json, sha
from repo_paths import prepare_artifact_path, link_artifact

# Pre-specified windows from the existing return screen, including preceding
# checkpoints. These trigger paired interventions, not a claim of causality.
ONSET = {2: 19, 3: 19, 7: 114, 8: 25, 5: 21, 9: 20, 11: 20}


def run(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    row, config = load_target(args.target)
    out = args.run_dir / ('pilots' if args.pilot else 'runs') / target_name(row)
    status = json.loads((out / 'status.json').read_text())
    if status['state'] != 'complete':
        raise ValueError('Training replay must finish and pass fidelity checks before evaluation')
    trainer, _, _ = load_trainer(row['version'])
    from jaxmarl.environments.coordination_grid import CoordinationGrid, TEST_CAPABILITY_PAIRS
    panel_file = args.run_dir / 'evaluation_panel.json'
    panel = json.loads(panel_file.read_text())
    packets_all = np.asarray(panel['layout_packets'], np.int32)
    checkpoint_entries = [r for r in json.loads((out / 'artifacts.json').read_text()) if r['kind'] == 'params']
    if args.smoke:
        checkpoint_entries = [max(checkpoint_entries, key=lambda r:r['update'])]
    populations = [('heldout', np.asarray(TEST_CAPABILITY_PAIRS, np.int32)),
                   ('training', np.asarray(trainer.resolve_training_capability_pool(config), np.int32))]
    if args.smoke:
        populations = [(name, caps[:2]) for name,caps in populations]
    rows, summaries, artifact_rows = [], [], []
    eval_status = out / 'evaluation_status.json'
    write_json(eval_status, dict(state='running', started_unix=time.time(), panel_sha256=sha(panel_file)))
    intervention_updates = {0, 16, 32, 64, 915}
    if args.target in ONSET:
        u = ONSET[args.target]
        intervention_updates.update((u-1, u, u+1))
    fields = ('agent_pos', 'time', 'round_idx', 'layout_idx', 'capability', 'partner_goal',
              'partner_assignment', 'partner_move_ctr', 'last_ego_allocation')
    runners = {}

    def create_runner(capabilities, repetitions):
        kwargs = dict(config['ENV_KWARGS'])
        kwargs.update(augment_symmetries=False, partner_capability_pairs=capabilities.tolist())
        env = CoordinationGrid(**kwargs)
        network, init_memory, _ = trainer.build_network(config, env.n_ego_actions)
        caps = jnp.repeat(jnp.asarray(capabilities), repetitions, axis=0)
        packets = jnp.tile(jnp.asarray(packets_all[:repetitions]), (len(capabilities), 1))
        batch = len(caps)
        k_reset, k_step = jax.random.split(jax.random.PRNGKey(panel['evaluation_seed']))
        reset_keys = jax.random.split(k_reset, batch)
        if row['version'] == 'v1_matched':
            obs0, state0 = jax.vmap(env.reset_from_schedule)(caps, packets)
        else:
            obs0, state0 = jax.vmap(env.reset_from_schedule)(caps, packets, reset_keys)

        @jax.jit
        def rollout(params, intervention):
            def step(carry, unused):
                obs, state, memory, reset, ended, key = carry
                inp = jax.tree_util.tree_map(lambda x: x[jnp.newaxis], obs['agent_0'])
                memory, pi, value = network.apply(params, memory, (inp, reset[jnp.newaxis]))
                key, ka, ks = jax.random.split(key, 3)
                requested = pi.sample(seed=ka).squeeze(0)
                assignment = state.partner_assignment
                # Modes: 0 baseline; 1 hold initial; 2 permit first online
                # request then hold; 3 fixed ego-red; 4 fixed ego-blue.
                threshold = jnp.where(intervention == 2, 2, 1)
                online = row['version'] != 'v1_matched'
                hold = ((intervention == 1) | (intervention == 2)) & (state.time >= threshold) & online
                applied = jnp.where(hold, 3*(requested//3) + assignment, requested)
                fixed = (intervention == 3) | (intervention == 4)
                chosen = jnp.where(intervention == 3, 1, 2)
                # v2 initialization is random and remains so. The assignment
                # clamp begins at its first legal movement/action request.
                can_apply = ((state.time == 0) if not online else
                             ((state.time >= 1) if row['version'] == 'v2' else jnp.ones(batch, bool)))
                applied = jnp.where(fixed & can_apply, 3*(requested//3) + chosen, applied)
                obs2, state2, reward, done, info = jax.vmap(env.step_env, in_axes=(0,0,{'agent_0':0}))(
                    jax.random.split(ks, batch), state, {'agent_0': applied})
                trace = {k: getattr(state, k) for k in fields}
                trace.update({f'post/{k}': getattr(state2,k) for k in fields})
                trace.update(valid=~ended, action=requested, applied_action=applied,
                    reward=reward['agent_0'], done=done['__all__'], success=info['success'],
                    round_done=info['round_done'], effective_assignment=info['partner_assignment'],
                    ego_move_effective=info['ego_move_effective'], partner_move_effective=info['partner_move_effective'],
                    value=value.squeeze(0), entropy=pi.entropy().squeeze(0))
                return (obs2, state2, memory, done['__all__'], ended | done['__all__'], key), trace
            _, trace = jax.lax.scan(step, (obs0, state0, init_memory(batch),
                jnp.zeros(batch, bool), jnp.zeros(batch, bool), k_step),
                None, length=env.rounds_per_episode * env.max_steps)
            return trace
        return rollout, state0, caps, packets

    for entry in sorted(checkpoint_entries, key=lambda e:e['update']):
        u = entry['update']
        params = serialization.msgpack_restore(Path(entry['path']).read_bytes())
        modes = [(0, 'baseline')]
        if u in intervention_updates or args.smoke:
            modes += [(3, 'fixed_ego_red'), (4, 'fixed_ego_blue')]
            if row['version'] != 'v1_matched':
                modes += [(1, 'hold_initial'), (2, 'hold_after_first_request')]
        repetitions = 1 if args.smoke else (20 if u in {0, 915} or u == ONSET.get(args.target) else 4)
        for population, capabilities in populations:
            cache_key = (population, repetitions)
            if cache_key not in runners:
                runners[cache_key] = create_runner(capabilities, repetitions)
            rollout, state0, caps, packets = runners[cache_key]
            batch = len(caps)
            for mode, name in modes:
                tr = jax.tree_util.tree_map(np.asarray, rollout(params, jnp.int32(mode)))
                if not np.all((tr['round_done'] & tr['valid']).sum(axis=0) == 20):
                    raise ValueError('Incomplete evaluation episodes')
                logical = out / 'evaluation' / f'update{u:04d}_{population}_{name}.npz'
                physical = prepare_artifact_path(logical)
                with physical.with_suffix('.npz.tmp').open('wb') as f:
                    np.savez_compressed(f, **tr, layout_packets=np.asarray(packets), profiles=np.asarray(caps),
                        initial_rng=np.asarray(jax.random.PRNGKey(panel['evaluation_seed'])))
                physical.with_suffix('.npz.tmp').replace(physical)
                link_artifact(logical)
                artifact_rows.append(dict(update=u, population=population, intervention=name,
                    path=str(logical), physical=str(physical), bytes=physical.stat().st_size, sha256=sha(physical)))
                round_rows = []
                for ep in range(batch):
                    for r in range(20):
                        mask = tr['valid'][:,ep] & (tr['round_idx'][:,ep] == r)
                        ticks = np.flatnonzero(mask)
                        a, b = ticks[0], ticks[-1]
                        if not tr['round_done'][b,ep]:
                            raise ValueError('Missing round completion')
                        allocs = tr['effective_assignment'][ticks,ep]
                        switches = int((allocs[1:] != allocs[:-1]).sum())
                        # fixed_allocation starts unset; t0 creates assignment. v2 starts
                        # with the environment's random assignment.
                        dr, db = map(int, capabilities[ep//repetitions])
                        match = (allocs[-1] == (2 if dr < db else 1)) if dr != db else np.nan
                        round_rows.append(dict(update=u, population=population, intervention=name,
                            profile=ep//repetitions, repetition=ep%repetitions, episode=ep, round=r,
                            layout_index=int(tr['layout_idx'][a,ep]), d_R=dr, d_B=db,
                            first_tick=int(a), last_tick=int(b), steps=len(ticks),
                            success=bool(tr['success'][b,ep]), return_=float(tr['reward'][ticks,ep].sum()),
                            initial_ego_red=bool(allocs[0] == 1), final_ego_red=bool(allocs[-1] == 1),
                            partner_matching=float(match), switches=switches,
                            overridden_actions=int((tr['action'][ticks,ep] != tr['applied_action'][ticks,ep]).sum())))
                ledger = logical.with_suffix('.rounds.csv.gz')
                with gzip.open(prepare_artifact_path(ledger), 'wt', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(round_rows[0])); w.writeheader(); w.writerows(round_rows)
                link_artifact(ledger)
                summary = dict(update=u, population=population, intervention=name, episodes=batch,
                    return_=float(np.mean((tr['reward']*tr['valid']).sum(axis=0))),
                    success=float(np.mean([r['success'] for r in round_rows])),
                    final_ego_red=float(np.mean([r['final_ego_red'] for r in round_rows])),
                    partner_matching=float(np.nanmean([r['partner_matching'] for r in round_rows])),
                    switches_per_round=float(np.mean([r['switches'] for r in round_rows])),
                    fraction_rounds_switched=float(np.mean([r['switches']>0 for r in round_rows])),
                    fraction_rounds_10plus_switches=float(np.mean([r['switches']>=10 for r in round_rows])))
                summaries.append(summary)
                write_json(out / 'evaluation_summary.json', summaries)
                write_json(out / 'evaluation_artifacts.json', artifact_rows)
                print('[evaluation saved] '+json.dumps(summary), flush=True)
        write_json(eval_status, dict(state='running', last_checkpoint=u, completed_conditions=len(summaries)))
    write_json(eval_status, dict(state='complete', completed_conditions=len(summaries),
        panel_sha256=sha(panel_file), interventions='Paired closed-loop assignment clamps; future observations and GRU recomputed.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=int, required=True, choices=range(12))
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--smoke', action='store_true')
    run(p.parse_args())

if __name__ == '__main__':
    main()
