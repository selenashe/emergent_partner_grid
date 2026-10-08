"""Streaming audit and raw event sampling; historical files are read-only."""
from pathlib import Path
import hashlib
import json
import runpy
import subprocess
import h5py
import numpy as np
from eval.representation_analysis import episode_valid_length
from repo_paths import ROOT, resolve_path

CONDITIONS = ('rnn_diverse_influence', 'rnn_single_influence', 'rnn_diverse_noinfluence')
FIELDS = ('hidden_state', 'capability', 'layout_idx', 'round_idx', 'round_done', 'success',
          'rewards', 'dones', 'ego_alloc_action', 'partner_assignment', 'partner_goal',
          'round_time', 'is_t0', 'assignment_changed', 'ego_request_changed')
RICH_FIELDS = ('observation', 'encoded_input', 'carry_before', 'logits', 'flat_action', 'pre_state', 'step_key')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')

def populations():
    m = runpy.run_path(str(ROOT / 'jaxmarl/environments/coordination_grid/capability_populations.py'))
    return {k: np.asarray(m[k + '_CAPABILITY_PAIRS'], dtype=int) for k in ('TRAIN', 'TEST')}

def split_repetitions(n, seed):
    # Audit guide:
    # Partition repetition IDs within profiles into discovery, validation, and test
    # sets. Keep every event from an episode in the same split so nearby hidden states
    # do not leak across fit and test sets.
    #
    if n < 4:
        raise ValueError('At least four independent repetitions are required')
    order = np.random.default_rng(seed).permutation(n)
    a, b = max(2, int(.6*n)), max(1, int(.2*n))
    return {int(r): ('train' if i < a else 'validation' if i < a+b else 'test') for i,r in enumerate(order)}

def round_boundaries(record):
    # Audit guide:
    # Recover ordered round starts and inclusive endings within the first-done prefix.
    # Hidden states are post-observation states, while recorded round time is post-
    # transition; preserve that offset when naming events.
    #
    ends = np.flatnonzero(record['round_done'])
    if not np.array_equal(record['round_idx'][ends], np.arange(20)):
        raise ValueError('Expected exactly 20 ordered round endings')
    starts = np.r_[0, ends[:-1]+1]
    if ends[-1] != len(record['round_idx'])-1:
        raise ValueError('Final terminal transition must end round 19')
    expected = np.repeat(np.arange(20), ends-starts+1)
    if not np.array_equal(record['round_idx'], expected):
        raise ValueError('Fabricated or malformed round transition')
    if not np.array_equal(np.flatnonzero(record['is_t0']), starts):
        raise ValueError('Round starts disagree with pre-observation is_t0')
    return starts, ends


def validate_schema(handle):
    shape=handle['hidden_state'].shape
    if len(shape)!=3 or shape[-1]<=1 or handle['hidden_state'].dtype.kind!='f':
        raise ValueError('Hidden states must be real recurrent (episode, step, feature) floats')
    e,t,_=shape
    boolean={'round_done','success','dones','is_t0','assignment_changed','ego_request_changed'}
    for name in FIELDS:
        if name not in handle:continue
        expected=(e,t,2) if name=='capability' else shape if name=='hidden_state' else (e,t)
        if handle[name].shape!=expected:
            raise ValueError(f'{name}: trajectory shape mismatch')
        kind=handle[name].dtype.kind
        if name in boolean and kind!='b':raise ValueError(f'{name}: expected boolean dtype')
        if name not in boolean and name not in {'hidden_state','rewards'} and kind not in 'iu':
            raise ValueError(f'{name}: expected integer dtype')
        if name=='rewards' and kind!='f':raise ValueError('rewards: expected float dtype')

def inventory(config):
    # Audit guide:
    # Identify existing rollout/checkpoint evidence and check protocol, shapes, and
    # provenance before computing geometry. Missing inputs are a collection requirement,
    # not permission to substitute another experimental batch.
    #
    manifest = ROOT / 'train/manifests' / ('sbatch_' + config['batch'] + '.json')
    m = json.loads(manifest.read_text())
    files = []
    for version in config['protocols']:
        v = m['versions'][version]
        source = resolve_path(v['frozen_source_root'])
        verified = {p: sha(source/p) == s for p,s in v['source_sha256'].items()}
        if not all(verified.values()):
            raise ValueError('Frozen source hash mismatch: ' + str(source))
        for condition in config['conditions']:
            for seed in config['seeds']:
                stem = f'{condition}_seed{seed}'
                cfgpath = resolve_path(v['checkpoint_root']) / (stem+'_config.json')
                weights = cfgpath.with_name(stem+'.safetensors')
                cfg = json.loads(cfgpath.read_text())
                protocol = v['allocation_protocol']
                if cfg.get('ALLOCATION_PROTOCOL', 'fixed_v1') != protocol:
                    raise ValueError('Checkpoint protocol mismatch')
                for sl in ('train', 'test'):
                    path = resolve_path(v['evaluation_root']) / (stem+'_'+sl+'.h5')
                    with h5py.File(path) as f:
                        validate_schema(f)
                        if f.attrs.get('allocation_protocol', 'fixed_v1') != protocol:
                            raise ValueError('Rollout protocol mismatch')
                        absent = sorted(set(FIELDS)-set(f.keys()))
                        # v1 did not log v2 request/switch flags; derive timing from real round starts.
                        required = set(FIELDS) - {'assignment_changed','ego_request_changed'}
                        if required-set(f.keys()):
                            raise ValueError(f'{path}: missing {required-set(f.keys())}')
                        stat = path.stat()
                        files.append(dict(path=str(path), version=version, protocol=protocol,
                            condition=condition, seed=seed, slice=sl, batch=config['batch'],
                            action_selection=cfg.get('ACTION_SELECTION','categorical'),
                            shapes={k:list(f[k].shape) for k in f}, dtypes={k:str(f[k].dtype) for k in f},
                            missing_fields=absent, missing_rich_fields=list(RICH_FIELDS),
                            size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                            fingerprint=hashlib.sha256((str(path)+str(stat.st_size)+str(stat.st_mtime_ns)+str(f['hidden_state'].shape)).encode()).hexdigest(),
                            checkpoint=str(weights), checkpoint_sha256=sha(weights), config=str(cfgpath), config_sha256=sha(cfgpath),
                            source=str(source), source_hashes_verified=True,
                            manifest=str(manifest), manifest_sha256=sha(manifest)))
    return files

def stream_events(files, config, out):
    """Read each episode independently; save raw start/mid/end states, never prefix averages."""
    # Audit guide:
    # Read one complete episode at a time and retain raw hidden states at each round
    # start, midpoint, and end. Unlike the standard probe pipeline these are event
    # samples, not cumulative prefix averages. Group all sixty samples of an episode in
    # one split; target-layout exclusions do not remove that layout from all preceding
    # histories.
    #
    samples, rows, audits, split_rows, summaries, transitions = [], [], [], [], [], []
    geometry = json.loads((ROOT/'data_prep/grids_capability_selected_balanced_1096/diagnostics/geometry_goal_dependence/summary.json').read_text())
    layout_files = sorted((resolve_path(json.loads(Path(files[0]['config']).read_text())['ENV_KWARGS']['layouts_dir'])).glob('*.json'))
    layout_features = []
    for path in layout_files:
        ld = json.loads(path.read_text()); md = ld['metadata']
        layout_features.append([md[k] for k in ('ego_to_red','ego_to_blue','partner_to_red','partner_to_blue')] + ld['ego_start'] + ld['partner_start'])
    pop = populations()
    for item in files:
        valid_steps = 0; lengths = []; reps_seen = {}; pool_expected = pop['TRAIN' if item['slice']=='train' else 'TEST']
        with h5py.File(item['path']) as f:
            pool = np.asarray(f['capability_pool'])
            if not np.array_equal(pool,pool_expected):
                raise ValueError('Capability pool ordering mismatch')
            emax = len(f['dones'])
            split = split_repetitions(emax//len(pool), config['analysis_seed'])
            for ep in range(emax):
                n = episode_valid_length(np.asarray(f['dones'][ep]))
                rec = {k:np.asarray(f[k][ep,:n]) for k in FIELDS if k in f}
                cap = rec['capability'][0]; ci = int(f['capability_index_per_ep'][ep]); rep = reps_seen.get(ci,0); reps_seen[ci] = rep+1
                if not np.all(rec['capability']==cap) or not np.array_equal(cap,pool[ci]):
                    raise ValueError('Capability constancy/profile ordering failed')
                if not np.isfinite(rec['hidden_state']).all() or not np.isfinite(rec['rewards']).all():
                    raise ValueError('Nonfinite trajectory')
                starts,ends = round_boundaries(rec)
                if not np.all(rec['round_time'] == np.arange(n)-np.repeat(starts,ends-starts+1)+1):
                    raise ValueError('Post-transition round_time alignment failed')
                uid = f"{item['slice']}:{ep}"
                split_rows.append(dict(episode=uid,profile=cap.tolist(),repetition=rep,split=split[rep]))
                lengths.append(n); valid_steps += n
                decisions = (~rec['is_t0']) if item['protocol']=='online_v2' else rec['is_t0']
                assigned_faster = (rec['partner_assignment']==2)==(cap[0]<cap[1])
                summaries.append(dict(episode=uid,profile=cap.tolist(),split=split[rep],n_steps=n,
                    success=float(rec['success'][ends].mean()),reward=float(rec['rewards'].sum()),
                    faster_fraction=float(assigned_faster[decisions].mean())))
                if config.get('smoke') and ci not in (0, 1, len(pool)//2, len(pool)//2+1):
                    continue
                for r,(a,b) in enumerate(zip(starts,ends)):
                    for phase,t in enumerate((a,(a+b)//2,b)):
                        li = int(rec['layout_idx'][t]); layout_split = ('test' if int(hashlib.sha256(str(li).encode()).hexdigest()[:8],16)%5==0 else 'train')
                        samples.append(rec['hidden_state'][t])
                        rows.append([uid,rep,split[rep],item['slice'],int(cap[0]),int(cap[1]),r,phase,int(t),li,layout_split,int(rec['partner_assignment'][t]),int(rec['ego_alloc_action'][t]),int(rec['partner_goal'][t]),int(rec['round_time'][t]),int(rec['is_t0'][t]),*layout_features[li]])
                    if r < 19:
                        delta=rec['hidden_state'][b+1]-rec['hidden_state'][b]
                        transitions.append(dict(episode=uid,round=r,delta_norm=float(np.linalg.norm(delta))))
            audits.append(dict(path=item['path'],episodes=emax,valid_steps=valid_steps,
                scan_padding_steps=int(emax*f['dones'].shape[1]-valid_steps),length_min=min(lengths),length_max=max(lengths),
                all_episodes_complete=True,ordered_rounds_verified=True,capability_verified=True,finite_verified=True,
                time_alignment_verified=True))
    import pandas as pd
    columns=['episode','repetition','split','capability_slice','d_red','d_blue','round','phase','t','layout','layout_split','assignment','request','goal','round_time','is_t0','ego_to_red','ego_to_blue','partner_to_red','partner_to_blue','ego_row','ego_col','partner_row','partner_col']
    index=pd.DataFrame(rows,columns=columns)
    out.mkdir(parents=True,exist_ok=True)
    index.to_csv(out/'sample_index.csv',index=False)
    write_json(out/'split_manifest.json',dict(episodes=split_rows,rule='60/20/20 deterministic repetition split per profile',layout_rule='sha256(layout index) modulo 5; target-only split, histories can overlap',analysis_seed=config['analysis_seed']))
    write_json(out/'data_validation.json',dict(files=audits,corpus_audit_verified_rule=geometry['capability_only_rule']))
    write_json(out/'episode_metrics.json',summaries)
    pd.DataFrame(transitions).to_csv(out/'round_boundary_transitions.csv',index=False)
    hidden=np.asarray(samples,dtype=np.float64)
    return hidden,index
