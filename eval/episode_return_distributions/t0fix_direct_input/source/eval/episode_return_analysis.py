"""Terminal-inclusive return, tail, and assignment diagnostics for the current cohort.

This module only reads saved evaluations and policies. It never trains a policy
or substitutes newly collected trajectories for the recorded evaluation data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from train.experiment import BATCH, CONDITIONS, ROOT, SEEDS, TRAINING_MANIFEST, VIEW_MANIFEST, VERSIONS

LABEL = 't0fix_direct_input'
BASE = ROOT / 'eval/episode_return_distributions'
BUILD = BASE / '.t0fix_direct_input_build'
FIELDS = ('dones', 'round_done', 'round_idx', 'round_time', 'is_t0', 'rewards',
          'success', 'partner_assignment', 'partner_goal', 'ego_alloc_action', 'layout_idx')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def identity(record):
    return {k: record[k] for k in ('version', 'condition', 'seed')}


def key(record):
    return f"{record['version']}_{record['condition']}_seed{record['seed']}"


def records():
    from train.cohort_integrity import verify
    manifest = verify(TRAINING_MANIFEST)
    from eval.plot_overcooked_paper_match import load_manifests, training_candidates, select_candidates
    candidates = training_candidates(load_manifests(VIEW_MANIFEST))
    selected = {key(x) for x in select_candidates(candidates, SEEDS)}
    by_id = {key(x): x for x in candidates}
    result = []
    for job in manifest['training_jobs']:
        data = manifest['versions'][job['version']]
        checkpoint = Path(data['checkpoint_root']) / (job['tag'] + '.safetensors')
        config = checkpoint.with_name(checkpoint.stem + '_config.json')
        cfg = json.loads(config.read_text())
        require(cfg['GRID_ENCODER'] == 'flat' and cfg['INCLUDE_IS_T0'] is True, 'Wrong observation pipeline')
        require(cfg['RUN_UPDATES'] == 915 and cfg['ENV_KWARGS']['step_penalty'] == .01, 'Wrong training budget/reward')
        audit = config.with_name(config.name.removesuffix('_config.json') + '_sampling_audit.npz')
        with np.load(audit, allow_pickle=False) as saved:
            profiles = saved['capability_pairs'].astype(int).tolist()
        record = dict(identity(job), tag=job['tag'], checkpoint=str(checkpoint), config=str(config),
            checkpoint_sha256=sha(checkpoint), config_sha256=sha(config), gamma=cfg['GAMMA'], penalty=.01,
            source_root=manifest['source_variants'][job['source_variant']]['frozen_source_root'],
            allocation_protocol=data['allocation_protocol'], layouts_dir=manifest['frozen_layouts_dir'],
            training_profiles=profiles, selected=key(job) in selected,
            **{k: by_id[key(job)][k] for k in ('training_return', 'training_log', 'training_log_sha256', 'final_update', 'environment_steps')})
        record.update({p + '_file': str(Path(data['evaluation_root']) / (job['tag'] + '_' + p + '.h5'))
                       for p in ('train', 'test')})
        record['summary_file'] = str(Path(data['evaluation_root']) / (job['tag'] + '_summary.json'))
        result.append(record)
    require(len(result) == 60 and len(selected) == 12, 'Incomplete cohort/selection')
    return result


def discount_returns(reward, done, gamma):
    """Zero terminal bootstrap, retaining the terminal reward and masking padding."""
    reward = np.asarray(reward, dtype=np.float64)
    done = np.asarray(done, bool)
    lengths = done.argmax(axis=1) + 1
    require(done.any(axis=1).all(), 'Missing first episode termination')
    result = np.zeros_like(reward)
    carry = np.zeros(len(reward))
    for t in range(reward.shape[1] - 1, -1, -1):
        carry = np.where(t < lengths, reward[:, t] + gamma * carry * ~done[:, t], 0)
        result[:, t] = carry
    return result


def extract(a, version, influence, penalty=.01, rounds=20):
    """Validate first-done boundaries and reconstruct effective round switches."""
    done = np.asarray(a['dones'], bool)
    require(done.ndim == 2 and done.any(axis=1).all(), 'Missing episode terminal')
    E, T = done.shape
    require(all(a[k].shape == (E, T) for k in FIELDS), 'Inconsistent scalar rollout shapes')
    lengths = done.argmax(axis=1) + 1
    alive = np.arange(T)[None, :] < lengths[:, None]
    initial = a['is_t0'].astype(bool) & alive
    ending = a['round_done'].astype(bool) & alive
    require((initial.sum(1) == rounds).all() and (ending.sum(1) == rounds).all(), 'Incomplete rounds')
    starts = np.nonzero(initial)[1].reshape(E, rounds)
    ends = np.nonzero(ending)[1].reshape(E, rounds)
    require((starts[:, 0] == 0).all() and np.array_equal(starts[:, 1:], ends[:, :-1] + 1), 'Round gaps')
    require(np.array_equal(ends[:, -1], lengths - 1), 'Missing final round terminal')
    expected_round = np.broadcast_to(np.arange(rounds), (E, rounds))
    require(np.array_equal(a['round_idx'][initial].reshape(E, rounds), expected_round), 'Unordered round starts')
    require(np.array_equal(a['round_idx'][ending].reshape(E, rounds), expected_round), 'Unordered round ends')
    ep, tick = np.nonzero(alive)
    rid = a['round_idx'][alive].astype(int)
    require(np.all((rid >= 0) & (rid < rounds)), 'Invalid round label')
    require(np.array_equal(a['round_time'][alive], tick - starts[ep, rid] + 1), 'Round-time inconsistency')
    assignment = a['partner_assignment']
    require(np.isin(assignment[alive], [1, 2]).all(), 'Unknown effective assignment')
    switched = np.zeros_like(alive)
    switched[:, 1:] = ((assignment[:, 1:] != assignment[:, :-1]) &
                       (a['round_idx'][:, 1:] == a['round_idx'][:, :-1]) & alive[:, 1:] & ~initial[:, 1:])
    if version == 'v1_matched' or not influence:
        require(not switched.any(), 'Uncontrollable allocation changed')
    rewards = a['rewards'].astype(np.float64)
    require(np.isfinite(rewards[alive]).all(), 'Nonfinite recorded rewards')
    ids = ep * rounds + rid
    def reduce(value):
        return np.bincount(ids, weights=np.asarray(value)[alive], minlength=E*rounds).reshape(E, rounds)
    round_return = reduce(rewards)
    success = a['success'][ending].reshape(E, rounds).astype(bool)
    steps = ends - starts + 1
    ledger = (1 + penalty) * success - penalty * steps
    require(np.allclose(round_return, ledger, atol=3e-5, rtol=0), 'Reward ledger does not close')
    return dict(lengths=lengths, alive=alive, starts=starts, ends=ends, switched=switched,
                returns=round_return, success=success, steps=steps, switches=reduce(switched).astype(int))


def tail_selection(returns, populations, episode_indices):
    """Exactly 10%, stable return/population/index order, plus tie-inclusive flags."""
    returns = np.asarray(returns)
    require(np.isfinite(returns).all() and len(returns) % 10 == 0, 'Invalid decile cohort')
    order = np.lexsort((episode_indices, populations, returns))
    n = len(returns) // 10
    mask = np.zeros(len(returns), bool); mask[order[:n]] = True
    threshold = returns[order[n-1]]
    tied = returns == threshold
    return mask, np.argsort(order)+1, returns <= threshold, dict(
        n_tail=n, threshold=float(threshold), boundary_tied_episodes=int(tied.sum()),
        boundary_ties_selected=int((tied & mask).sum()), n_tail_including_boundary_ties=int((returns <= threshold).sum()))


def trend(early, late, seed=2026, draws=4000):
    delta = np.asarray(late) - np.asarray(early)
    require(len(delta) > 0 and np.isfinite(delta).all(), 'Invalid episode trend')
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    mean = float(delta.mean())
    label = 'increasing' if low > 0 else 'decreasing' if high < 0 else 'approximately flat' if abs(mean) < .05 else 'no clear trend'
    return dict(early5=float(np.mean(early)), late5=float(np.mean(late)), late_minus_early=mean,
                delta_ci_low=float(low), delta_ci_high=float(high), trend=label,
                fraction_episodes_improving=float((delta > 0).mean()))


def layout_grids(layout_dir):
    paths = sorted(Path(layout_dir).glob('*.json'))
    grids = []
    for p in paths:
        d = json.loads(p.read_text());grid = np.zeros((7, 7, 5), np.float32)
        grid[:, :, 0] = np.asarray(d['grid']) == 1
        for channel, field in [(1, 'red_goal'), (2, 'blue_goal'), (3, 'ego_start'), (4, 'partner_start')]:
            grid[tuple(d[field]) + (channel,)] = 1
        grids.append(grid)
    require(len(paths) == 1096, 'Wrong scientific corpus')
    return paths, np.asarray(grids)


class Oracle:
    """Read geometry/task bounds, never import or evaluate old trained models."""
    def __init__(self, layout_dir):
        self.paths, _ = layout_grids(layout_dir)
        self.tables = {}
        self.sources = []
        for population, folder in [('train', 'training_return_ceiling'), ('test', 'heldout_return_ceiling')]:
            path = ROOT/'eval'/folder/'v1_matched_v2_v3/per_layout_profile.csv'
            table = pd.read_csv(path)
            require(table.groupby('layout_index').layout_id.nunique().eq(1).all(), 'Ambiguous oracle layout order')
            order = table.groupby('layout_index').layout_id.first()
            require(order.tolist() == [p.stem for p in self.paths], 'Oracle/corpus layout order mismatch')
            self.tables[population] = {(int(x.layout_index), int(x.d_R), int(x.d_B)): x for x in table.itertuples()}
            self.sources.append(dict(path=str(path), sha256=sha(path)))

    def values(self, population, layout, dr, db, assignment, version, influence, horizon=100):
        row = self.tables[population][(int(layout), int(dr), int(db))]
        fields = ('online_initial_partner_red_steps', 'online_initial_partner_blue_steps') if version != 'v1_matched' and influence else ('fixed_partner_red_steps', 'fixed_partner_blue_steps')
        red, blue = (int(getattr(row, k)) for k in fields)
        initial = blue if assignment == 1 else red
        def reward(t):return 1 - .01*(t-1) if t <= horizon else -.01*horizon
        return initial, min(red, blue), reward(initial), reward(min(red, blue))


def extract_policy(record, root=BUILD):
    """Build reproducible episode, round, and switch-event data for one policy."""
    target = Path(root)/'t0fix_direct_input/cache'/key(record)
    target.mkdir(parents=True, exist_ok=True)
    if (target/'completion.json').exists():
        cached = json.loads((target/'completion.json').read_text())
        require(cached['checkpoint_sha256'] == record['checkpoint_sha256'] and cached['config_sha256'] == record['config_sha256'], 'Stale policy cache')
        if cached.get('producer_sha256') == sha(__file__):
            for item in cached['sources']:
                source=Path(item['path'])
                require(source.stat().st_size==item['size'] and source.stat().st_mtime_ns==item['mtime_ns'],'Changed evaluation input')
            return str(target)
    print('[extract]', key(record), flush=True)
    oracle = Oracle(record['layouts_dir'])
    episode_rows=[];round_rows=[];events=[];sources=[]
    cfg=json.loads(Path(record['config']).read_text());influence=bool(cfg['INFLUENCE'])
    profiles={tuple(x) for x in record['training_profiles']}
    summaries=json.loads(Path(record['summary_file']).read_text())
    for population, expected in [('train',480),('test',440)]:
        path=Path(record[population+'_file'])
        with h5py.File(path) as f:
            a={k:f[k][:] for k in FIELDS};pairs=f['capability'][:,0,:];pool=f['capability_pool'][:]
            profile=f['capability_index_per_ep'][:]
            require(len(a['dones'])==expected and len(pool)==(24 if population=='train' else 22),'Wrong profile/episode coverage')
            require(np.array_equal(pool[profile],pairs),'Capability index mismatch')
        d=extract(a,record['version'],influence)
        require(np.isclose(d['returns'].sum(1).mean(),summaries[population]['mean_ep_return'],atol=2e-4,rtol=0),'Saved summary return mismatch')
        require(np.isclose(d['success'].mean(),summaries[population]['round_success_rate'],atol=1e-6,rtol=0),'Saved summary success mismatch')
        if 'assignment_switch_count_by_round' in summaries[population]:
            require(np.array_equal(d['switches'].sum(0),summaries[population]['assignment_switch_count_by_round']),'Saved summary switch counts differ')
        sources.append(dict(path=str(path),physical_path=str(path.resolve()),size=path.stat().st_size,
            mtime_ns=path.stat().st_mtime_ns,scalar_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in a.items()}))
        reps=collections_counter={}
        for e in range(expected):
            dr,db=map(int,pairs[e]);pair=(dr,db);rep=reps.get(pair,0);reps[pair]=rep+1
            common=dict(identity(record),population=population,episode_index=e,profile_index=int(profile[e]),repetition=rep,
                delay_red=dr,delay_blue=db,seen_during_training=pair in profiles,source_file=str(path))
            er=[]
            for rd in range(20):
                first,last=int(d['starts'][e,rd]),int(d['ends'][e,rd]);sl=slice(first,last+1)
                assignments=a['partner_assignment'][e,sl];sw=d['switched'][e,sl]
                li=int(a['layout_idx'][e,first]);require(np.all(a['layout_idx'][e,sl]==li),'Layout changed inside round')
                faster=np.where(assignments==2,dr<db,db<dr)
                oracle_initial,oracle_best,ri,rb=oracle.values(population,li,dr,db,int(assignments[0]),record['version'],influence,int(cfg['ENV_KWARGS']['max_steps']))
                rr=float(d['returns'][e,rd]);require(rr<=ri+3e-5,'Observed reward exceeds conditional geometry oracle')
                clock=0;forced=reload=preempt=eligible=0
                for local in range(1,len(assignments)):
                    delay=dr if assignments[local]==2 else db
                    changed=bool(sw[local]);preclock=clock
                    if changed:clock=delay
                    can_move=clock==0
                    eligible+=can_move;forced+=not can_move
                    reload+=changed and delay>0;preempt+=changed and preclock==0 and delay>0
                    if changed:
                        events.append(dict(common,round=rd+1,round_tick=local,episode_tick=first+local,layout_index=li,
                            from_ego_assignment=int(assignments[local-1]),to_ego_assignment=int(assignments[local]),
                            destination_delay=int(delay),toward_partner_faster=bool(faster[local]),
                            partner_cannot_move_on_switch=delay>0,preempted_ready_cadence=preclock==0 and delay>0))
                    clock=delay if can_move else clock-1
                request=a['ego_alloc_action'][e,sl]
                requests=int(((request[1:]!=request[:-1]) & (request[1:]!=0)).sum())
                row=dict(common,round=rd+1,layout_index=li,layout_id=oracle.paths[li].stem,
                    first_episode_tick=first,last_episode_tick=last,steps=int(d['steps'][e,rd]),
                    success=bool(d['success'][e,rd]),timeout=not bool(d['success'][e,rd]),round_return=rr,
                    switches=int(sw.sum()),subsequent_switches=int(sw.sum()-(bool(sw[1]) if record['version']=='v2' and len(sw)>1 else 0)),
                    request_changes=requests,switches_per_100_ticks=100*sw.sum()/max(1,len(sw)-1),
                    initial_ego_assignment=int(assignments[0]),final_ego_assignment=int(assignments[-1]),
                    initial_partner_faster=bool(faster[0]),final_partner_faster=bool(faster[-1]),fraction_ticks_partner_faster=float(faster.mean()),
                    cooldown_forced_stay_ticks=int(forced),switch_reload_forced_stay_ticks=int(reload),switches_preempting_ready_clock=int(preempt),partner_move_eligible_ticks=int(eligible),
                    oracle_initial_steps=oracle_initial,oracle_best_steps=oracle_best,oracle_initial_return=ri,oracle_best_return=rb,
                    assignment_opportunity_gap=rb-ri,execution_gap=ri-rr,initial_assignment_unfinishable=oracle_initial>int(cfg['ENV_KWARGS']['max_steps']))
                round_rows.append(row);er.append(row)
            rd=pd.DataFrame(er);steps=int(d['lengths'][e]);ret=float(rd.round_return.sum())
            stable=rd.switches==0
            episode_rows.append(dict(common,episode_return=ret,episode_steps=steps,successful_rounds=int(rd.success.sum()),
                timeouts=int(rd.timeout.sum()),switches=int(rd.switches.sum()),subsequent_switches=int(rd.subsequent_switches.sum()),request_changes=int(rd.request_changes.sum()),
                switched_rounds=int((rd.switches>0).sum()),burst_rounds_5plus=int((rd.switches>=5).sum()),burst_rounds_10plus=int((rd.switches>=10).sum()),
                switches_per_100_ticks=float(100*rd.switches.sum()/max(1,steps-20)),
                initial_partner_faster=float(rd.initial_partner_faster.mean()),final_partner_faster=float(rd.final_partner_faster.mean()),
                mean_successful_round_steps=float(rd.loc[rd.success,'steps'].mean()) if rd.success.any() else 0.,
                cooldown_forced_stay_ticks=int(rd.cooldown_forced_stay_ticks.sum()),switch_reload_forced_stay_ticks=int(rd.switch_reload_forced_stay_ticks.sum()),
                switches_preempting_ready_clock=int(rd.switches_preempting_ready_clock.sum()),
                oracle_best_return=float(rd.oracle_best_return.sum()),oracle_initial_return=float(rd.oracle_initial_return.sum()),
                assignment_opportunity_gap=float(rd.assignment_opportunity_gap.sum()),execution_gap=float(rd.execution_gap.sum()),
                early5_return=float(rd.round_return.iloc[:5].mean()),late5_return=float(rd.round_return.iloc[-5:].mean()),
                late_minus_early=float(rd.round_return.iloc[-5:].mean()-rd.round_return.iloc[:5].mean()),
                stable_slow_rounds=int((stable & ~rd.final_partner_faster).sum()),stable_slow_timeouts=int((stable & ~rd.final_partner_faster & rd.timeout).sum()),
                switched_timeouts=int((~stable & rd.timeout).sum()),ten_plus_switch_timeouts=int(((rd.switches>=10)&rd.timeout).sum()),
                maximum_switches_in_round=int(rd.switches.max()),unfinishable_initial_assignments=int(rd.initial_assignment_unfinishable.sum()),
                layout_sequence=';'.join(rd.layout_id),round_return_sequence=';'.join(f'{x:.5f}' for x in rd.round_return),
                switch_count_sequence=';'.join(map(str,rd.switches)),timeout_rounds=';'.join(map(str,rd.loc[rd.timeout,'round']))))
        require(set(reps.values())=={20},'Unequal profile repetitions')
    episodes=pd.DataFrame(episode_rows);rounds=pd.DataFrame(round_rows)
    mask,rank,ties,selection=tail_selection(episodes.episode_return.to_numpy(),episodes.population.to_numpy(),episodes.episode_index.to_numpy())
    episodes['lowest_decile']=mask;episodes['return_rank']=rank;episodes['tail_including_boundary_ties']=ties
    episodes['group']=np.where((episodes.delay_red==1)&(episodes.delay_blue==4),'training_partner',np.where(episodes.population=='train','other_reference','held_out'))
    flags=episodes[['population','episode_index','lowest_decile','return_rank','group']]
    rounds=rounds.merge(flags,on=['population','episode_index'],validate='many_to_one')
    event_frame=pd.DataFrame(events)
    if event_frame.empty:event_frame=pd.DataFrame(columns=list(common)+['round','round_tick','episode_tick','lowest_decile'])
    else:event_frame=event_frame.merge(flags,on=['population','episode_index'],validate='many_to_one')
    episodes.to_csv(target/'episode_metrics.csv.gz',index=False)
    rounds.to_csv(target/'round_metrics.csv.gz',index=False)
    event_frame.to_csv(target/'switch_events.csv.gz',index=False)
    t=episodes.loc[mask];rest=episodes.loc[~mask]
    diagnosis=dict(identity(record),training_return=record['training_return'],**selection,
        **trend(t.early5_return,t.late5_return),all_mean_return=float(episodes.episode_return.mean()),
        tail_mean_return=float(t.episode_return.mean()),rest_mean_return=float(rest.episode_return.mean()),
        tail_timeout_rounds=int(t.timeouts.sum()),tail_timeouts_per_episode=float(t.timeouts.mean()),
        tail_switches_per_round=float(t.switches.mean()/20),rest_switches_per_round=float(rest.switches.mean()/20),
        tail_switches_per_100_ticks=float(100*t.switches.sum()/(t.episode_steps.sum()-20*len(t))),
        rest_switches_per_100_ticks=float(100*rest.switches.sum()/(rest.episode_steps.sum()-20*len(rest))),
        tail_final_partner_faster=float(t.final_partner_faster.mean()),tail_actually_unseen_fraction=float((~t.seen_during_training).mean()),
        tail_novel_fraction=float((t.population=='test').mean()),tail_assignment_opportunity_gap=float(t.assignment_opportunity_gap.mean()),
        tail_execution_gap=float(t.execution_gap.mean()),
        lost_success_credit=float(1.01*(rest.successful_rounds.mean()-t.successful_rounds.mean())),
        extra_tick_penalty=float(.01*(t.episode_steps.mean()-rest.episode_steps.mean())))
    require(np.isclose(diagnosis['rest_mean_return']-diagnosis['tail_mean_return'],diagnosis['lost_success_credit']+diagnosis['extra_tick_penalty'],atol=3e-5),'Tail gap ledger does not close')
    mixture=t.groupby(['population','delay_red','delay_blue']).size()/len(t)
    profile_mean=episodes.groupby(['population','delay_red','delay_blue']).episode_return.mean()
    diagnosis['profile_standardized_mean']=float((mixture*profile_mean).sum())
    diagnosis['profile_composition_gap']=float(episodes.episode_return.mean()-diagnosis['profile_standardized_mean'])
    diagnosis['within_profile_gap']=diagnosis['profile_standardized_mean']-diagnosis['tail_mean_return']
    diagnosis['tie_inclusive_delta']=float(episodes.loc[ties,'late_minus_early'].mean())
    write(target/'diagnosis.json',diagnosis)
    write(target/'completion.json',dict(identity(record),status='completed',episodes=920,rounds=18400,tail_episodes=92,
        checkpoint_sha256=record['checkpoint_sha256'],config_sha256=record['config_sha256'],producer_sha256=sha(__file__),sources=sources,oracles=oracle.sources,
        method='first-done inclusive; recorded reward ledger; effective within-round switches; exact conditional geometry bounds',new_rollouts=False))
    return str(target)
