"""Replay a frozen PPO prefix, preserving its original full-run LR schedule.

The historical runs did not retain early per-grid traces. This diagnostic adds
outputs after the original callback and stops the outer scan after 20 updates.
Fidelity is assessed against historical metrics; outputs remain reproductions.
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
import time
import types

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_path, resolve_record_paths, prepare_artifact_path, link_artifact

SOURCE = ROOT / "train/run_snapshots/v1_matched_20261009_182845/v1_matched/source"
CONFIGS = ROOT / "train/train_logs/v1_matched_balanced_training/v1_matched_20261009_182845"
DEFAULT_OUT = ROOT / "train/seed_diagnostics/v1_matched_v2_v3/early_window_13_20"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instrument(source, updates):
    anchor = '            jax.debug.callback(_log_cb, metric)\n'
    scan = '            _update_step, runner_state, None, config["NUM_UPDATES"]'
    if source.count(anchor) != 1 or source.count(scan) != 1:
        raise ValueError("Unexpected frozen trainer structure")
    # Added only after all collection, GAE, optimizer updates, audits and the
    # original metric callback. Existing scientific calculations stay intact.
    extra = '''
            metric = dict(metric)
            metric["audit_trace"] = {
                "layout_idx": traj_batch.info["layout_idx"],
                "capability": traj_batch.info["capability"],
                "round_idx": traj_batch.info["round_idx"],
                "round_time": traj_batch.info["round_time"],
                "round_done": traj_batch.info["round_done"],
                "success": traj_batch.info["success"],
                "done": traj_batch.done,
                "pre_step_time": traj_batch.pre_step_time,
                "action": traj_batch.action,
                "reward": traj_batch.reward,
            }
'''
    return source.replace(anchor, anchor + extra).replace(
        scan, f'            _update_step, runner_state, None, {updates}')


def historical(seed, updates):
    path = ROOT / "/juice6/u/jshe/emergent_partner_grid/train/training_curves/v1_matched_v2_v3_20261019_191029_step_penalty_0.03/updates.csv"
    with path.open() as file:
        return [r for r in csv.DictReader(file) if r["version"] == "v1_matched"
                and r["condition"] == "rnn_diverse_influence" and int(r["seed"]) == seed
                and int(r["update"]) <= updates]


def run(args):
    import importlib.metadata
    import numpy as np
    import jax
    import jax.numpy as jnp
    import wandb
    from flax import serialization

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / f"seed{args.seed}_status.json"
    if (out / f"seed{args.seed}_provenance.json").exists():
        raise ValueError("Completed reproduction exists; choose another output directory")
    status = dict(seed=args.seed, state="starting", hostname=socket.gethostname(),
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"), started_unix=time.time())
    status_path.write_text(json.dumps(status, indent=2)+"\n")
    config_path = CONFIGS / f"rnn_diverse_influence_seed{args.seed}_config.json"
    config = resolve_record_paths(json.loads(config_path.read_text()))
    if config["NUM_UPDATES"] != 915 or config["TOTAL_TIMESTEPS"] != 60000000:
        raise ValueError("Original full-run schedule must be retained")
    source_path = SOURCE / "baselines/IPPO/ippo_rnn_coordination_grid.py"
    original = source_path.read_text()
    changed = instrument(original, args.updates)
    diff_path = out / "instrumentation.diff"
    diff = "".join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
        fromfile="frozen_trainer", tofile="instrumented_prefix"))
    if diff_path.exists() and diff_path.read_text() != diff:
        raise ValueError("Different instrumentation in shared output directory")
    diff_path.write_text(diff)
    sys.path.insert(0, str(SOURCE))
    sys.path.insert(0, str(SOURCE / "baselines/IPPO"))
    module = types.ModuleType("_frozen_early_exposure_replay")
    module.__file__ = str(source_path)
    sys.modules[module.__name__] = module
    exec(compile(changed, str(source_path), "exec"), module.__dict__)
    wandb.init(mode="disabled", project=config["PROJECT"], config=config)
    status.update(state="compiling_and_running", devices=[str(d) for d in jax.devices()])
    status_path.write_text(json.dumps(status, indent=2)+"\n")
    started = time.monotonic()
    # Match original main's jit/vmap order and PRNG construction exactly.
    train_jit = jax.jit(module.make_train(config))
    rngs = jax.random.split(jax.random.PRNGKey(config["SEED"]), config["NUM_SEEDS"])
    result = jax.vmap(train_jit)(rngs)
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x,"block_until_ready") else x, result)
    elapsed = time.monotonic()-started
    if config["NUM_UPDATES"] != 915:
        raise ValueError("Learning-rate schedule was shortened")
    metrics = {k: np.asarray(v)[0] for k,v in result["metrics"].items() if k != "audit_trace"}
    trace = {k: np.asarray(v)[0] for k,v in result["metrics"]["audit_trace"].items()}
    for k in ("layout_idx", "round_time", "pre_step_time"):
        trace[k] = trace[k].astype(np.int16)
    for k in ("capability", "round_idx", "action"):
        trace[k] = trace[k].astype(np.int8)
    for k in ("round_done", "success", "done"):
        trace[k] = trace[k].astype(bool)
    raw_path = out / "raw" / f"seed{args.seed}_trace.npz"
    np.savez_compressed(prepare_artifact_path(raw_path), **trace)
    link_artifact(raw_path)
    state_path = out / "raw" / f"seed{args.seed}_update{args.updates}_runner.msgpack"
    prepare_artifact_path(state_path).write_bytes(serialization.to_bytes(result["runner_state"]))
    link_artifact(state_path)
    rows=[]
    reference=historical(args.seed,args.updates)
    for i,r in enumerate(reference):
        rows.append(dict(seed=args.seed, update=i+1, env_steps=(i+1)*65536,
            replay_rounds=int(metrics["rounds_completed"][i]), historical_rounds=int(float(r["rounds_completed"])),
            replay_success=float(metrics["round_success_rate"][i]), historical_success=float(r["round_success"]),
            replay_return=float(metrics["ep_return_mean"][i]), historical_return=float(r["episode_return"]),
            replay_actor_loss=float(metrics["actor_loss"][i]), historical_actor_loss=float(r["actor_loss"]),
            replay_value_loss=float(metrics["value_loss"][i]), historical_value_loss=float(r["value_loss"]),
            replay_entropy_t0=float(metrics["entropy_t0"][i]), historical_entropy_t0=float(r["entropy_t0"]),
            replay_entropy_movement=float(metrics["entropy_tge1"][i]), historical_entropy_movement=float(r["entropy_movement"])))
    with (out/f"seed{args.seed}_historical_comparison.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    prefix=0
    for r in rows:
        if r["replay_rounds"]!=r["historical_rounds"] or abs(r["replay_success"]-r["historical_success"])>.00051:
            break
        prefix+=1
    metric_path=out/f"seed{args.seed}_metrics.json"
    metric_path.write_text(json.dumps({k:v.tolist() for k,v in metrics.items()},indent=2)+"\n")
    provenance=dict(seed=args.seed, prefix_updates=args.updates, original_updates=915,
        transitions=args.updates*65536, elapsed_seconds=elapsed, matched_historical_aggregate_prefix_updates=prefix,
        actual_devices=[dict(platform=d.platform,device_kind=d.device_kind) for d in jax.devices()],
        hostname=socket.gethostname(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        original_config=str(config_path),original_config_sha256=sha(config_path),
        frozen_trainer=str(source_path),frozen_trainer_sha256=sha(source_path),
        frozen_environment_sha256=sha(SOURCE/"jaxmarl/environments/coordination_grid/coordination_grid.py"),
        schedule_sha256=sha(resolve_path(config["COUNTERBALANCED_SCHEDULE_PATH"])),
        runner_source_sha256=sha(__file__),instrumentation_sha256=sha(diff_path),
        packages={k:importlib.metadata.version(k) for k in ("jax","jaxlib","flax","optax","distrax","numpy")},
        deterministic_environment={k:os.environ.get(k) for k in ("TF_CUDNN_USE_AUTOTUNE","TF_CUDNN_DETERMINISTIC","XLA_FLAGS")},
        trace=str(raw_path),trace_sha256=sha(raw_path),runner_state=str(state_path),
        caution="New instrumented reproduction. Matching rounded historical aggregates does not certify every original action or state.")
    (out/f"seed{args.seed}_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    status.update(state="complete",elapsed_seconds=elapsed,matched_historical_prefix_updates=prefix)
    status_path.write_text(json.dumps(status,indent=2)+"\n")
    print(json.dumps(status),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("run",))
    parser.add_argument("--seed",type=int,choices=(1,2),required=True)
    parser.add_argument("--updates",type=int,default=20)
    parser.add_argument("--out-dir",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if not 1<=args.updates<=40:
        parser.error("This diagnostic is limited to at most 40 PPO updates")
    run(args)


if __name__=="__main__":
    main()
