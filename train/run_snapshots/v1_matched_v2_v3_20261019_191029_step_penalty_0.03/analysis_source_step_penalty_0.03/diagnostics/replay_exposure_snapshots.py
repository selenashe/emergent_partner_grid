"""Minimal exposure snapshots with the original 915-update output shape.

Keep all original returned tensors and the outer scan length. Read existing
integer exposure counters at updates 12 and 20 through callbacks; stop only
after both the update-20 snapshot and scalar metrics are safely recorded.
"""

import argparse
import csv
import difflib
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import types

os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE","1")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from repo_paths import resolve_record_paths,prepare_artifact_path,link_artifact
from diagnostics.replay_early_exposure import SOURCE,CONFIGS,DEFAULT_OUT,historical


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instrument(original):
    anchor='            jax.debug.callback(_log_cb, metric)\n'
    host='                wandb.log(flat)\n'
    if original.count(anchor)!=1 or original.count(host)!=1:raise ValueError("Unexpected trainer")
    snapshot='''            def _emit_exposure_snapshot(_):
                jax.debug.callback(_audit_snapshot_callback, new_update_step,
                    episode_cursor.layout_counts, episode_cursor.episode_counts,
                    episode_cursor.next_episode, episode_cursor.slot_episode,
                    env_state.round_idx, env_state.time)
                return jnp.int32(0)
            jax.lax.cond((new_update_step == 12) | (new_update_step == 20),
                _emit_exposure_snapshot, lambda _: jnp.int32(0), operand=None)
'''
    return original.replace(anchor,snapshot+anchor).replace(host,'                _audit_metrics_callback(flat)\n'+host)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,choices=(1,2),required=True)
    parser.add_argument("--out-dir",type=Path,default=DEFAULT_OUT/"counter_snapshots")
    args=parser.parse_args();out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    import importlib.metadata
    import numpy as np
    import jax
    import wandb
    if (out/f"seed{args.seed}_provenance.json").exists():raise ValueError("Completed output exists")
    status_path=out/f"seed{args.seed}_status.json"
    status=dict(seed=args.seed,state="starting",hostname=socket.gethostname(),job_id=os.environ.get("SLURM_JOB_ID"))
    status_path.write_text(json.dumps(status,indent=2)+"\n")
    records={};snapshots={};lock=threading.Lock()
    metric_path=out/f"seed{args.seed}_metrics.json"
    marker="EARLY_EXPOSURE_DIAGNOSTIC_COMPLETE"

    def stop_when_ready():
        if 20 in records and 12 in snapshots and 20 in snapshots:
            raise RuntimeError(marker)

    def metric_callback(metric):
        with lock:
            update=int(metric["update_step"])
            records[update]={k:float(v) for k,v in metric.items()}
            metric_path.write_text(json.dumps(records,indent=2)+"\n")
            stop_when_ready()

    def snapshot_callback(update,counts,eps,allocated,ids,round_idx,round_time):
        with lock:
            update=int(update)
            path=out/"raw"/f"seed{args.seed}_update{update}_exposure.npz"
            np.savez_compressed(prepare_artifact_path(path),layout_counts=np.asarray(counts),
                episode_counts=np.asarray(eps),allocated_episodes=np.asarray(allocated),
                active_episode_ids=np.asarray(ids),active_round_idx=np.asarray(round_idx),active_round_time=np.asarray(round_time))
            link_artifact(path);snapshots[update]=str(path)
            print(f"[exposure snapshot] seed={args.seed} update={update} {path}",flush=True)
            stop_when_ready()

    source_path=SOURCE/"baselines/IPPO/ippo_rnn_coordination_grid.py"
    original=source_path.read_text();changed=instrument(original)
    diff_path=out/"instrumentation.diff"
    diff_path.write_text("".join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),
        fromfile="frozen_trainer",tofile="counter_snapshot_callbacks")))
    sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(SOURCE/"baselines/IPPO"))
    module=types.ModuleType("_frozen_counter_snapshot_trainer");module.__file__=str(source_path)
    module._audit_snapshot_callback=snapshot_callback;module._audit_metrics_callback=metric_callback
    sys.modules[module.__name__]=module
    exec(compile(changed,str(source_path),"exec"),module.__dict__)
    config_path=CONFIGS/f"rnn_diverse_influence_seed{args.seed}_config.json"
    config=resolve_record_paths(json.loads(config_path.read_text()))
    assert config["NUM_UPDATES"]==915 and config["TOTAL_TIMESTEPS"]==60000000
    wandb.init(mode="disabled",project=config["PROJECT"],config=config)
    status.update(state="compiling_and_running",devices=[d.device_kind for d in jax.devices()])
    status_path.write_text(json.dumps(status,indent=2)+"\n")
    started=time.monotonic();caught=None
    try:
        train_jit=jax.jit(module.make_train(config))
        result=jax.vmap(train_jit)(jax.random.split(jax.random.PRNGKey(config["SEED"]),config["NUM_SEEDS"]))
        jax.tree_util.tree_map(lambda x:x.block_until_ready() if hasattr(x,"block_until_ready") else x,result)
    except Exception as exc:
        caught=str(exc)
        if not (20 in records and 12 in snapshots and 20 in snapshots):raise
    rows=[];prefix=0;contiguous=True
    for r in historical(args.seed,20):
        u=int(r["update"]);m=records[u]
        match=int(m["rounds_completed"])==int(float(r["rounds_completed"])) and abs(m["round_success_rate"]-float(r["round_success"]))<=.00051
        if contiguous and match:prefix+=1
        else:contiguous=False
        rows.append(dict(seed=args.seed,update=u,replay_rounds=int(m["rounds_completed"]),historical_rounds=int(float(r["rounds_completed"])),
            replay_success=m["round_success_rate"],historical_success=float(r["round_success"]),
            replay_actor_loss=m["actor_loss"],historical_actor_loss=float(r["actor_loss"]),
            replay_value_loss=m["value_loss"],historical_value_loss=float(r["value_loss"]),matches=match))
    with (out/f"seed{args.seed}_historical_comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    provenance=dict(seed=args.seed,original_scan_updates=915,original_lr_updates=config["NUM_UPDATES"],
        matched_historical_aggregate_prefix_updates=prefix,snapshots=snapshots,
        elapsed_seconds=time.monotonic()-started,hostname=socket.gethostname(),
        device_kind=status["devices"],source_sha256=sha(source_path),instrumentation_sha256=sha(diff_path),
        runner_source_sha256=sha(__file__),packages={k:importlib.metadata.version(k) for k in ("jax","jaxlib","flax","optax","distrax","numpy")},
        stopped_by_callback=bool(caught),snapshots_sha256={k:sha(v) for k,v in snapshots.items()},
        caveat="Counter snapshots from a new reproduction. Check per-update historical agreement before attributing exposure to historical training.")
    (out/f"seed{args.seed}_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    status.update(state="complete",matched_historical_prefix_updates=prefix)
    status_path.write_text(json.dumps(status,indent=2)+"\n");print(json.dumps(status),flush=True)


if __name__=="__main__":main()
