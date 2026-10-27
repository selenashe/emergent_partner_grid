"""Compare v1_matched seeds 1/2 on their exact scheduled initial stimuli.

Static reconstruction uses saved schedules and validated exact task ceilings.
Optional CPU replay collects only the original first rollout, before any PPO
update. It is a reproduction, not a recovery of unsaved historical trajectories.
"""

import argparse
import ast
from collections import deque
import csv
import difflib
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_path, resolve_record_paths

CHECKPOINTS = ROOT / "train/train_logs/v1_matched_balanced_training/v1_matched_20261009_182845"
SOURCE = ROOT / "train/run_snapshots/v1_matched_20261009_182845/v1_matched/source"
ORACLES = ROOT / "eval/training_return_ceiling/v1_matched_v2_v3"
HISTORIES = ROOT / "/juice6/u/jshe/emergent_partner_grid/train/training_curves/v1_matched_v2_v3_20261019_191029_step_penalty_0.03"
PHASES = ((1,12), (13,20), (21,30), (31,46), (47,64), (65,92))


def read_csv(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def history():
    data = {s: [] for s in (1,2)}
    for r in read_csv(HISTORIES/"updates.csv"):
        s = int(r["seed"])
        if r["version"] == "v1_matched" and r["condition"] == "rnn_diverse_influence" and s in data:
            data[s].append({k: float(v) for k,v in r.items() if k not in ("version","condition")})
    return data


def setup(out):
    configs = {s: resolve_record_paths(json.loads((CHECKPOINTS/f"rnn_diverse_influence_seed{s}_config.json").read_text())) for s in (1,2)}
    audits = {s: json.loads((CHECKPOINTS/f"rnn_diverse_influence_seed{s}_sampling_audit.json").read_text()) for s in (1,2)}
    schedule_paths = [resolve_path(configs[s]["COUNTERBALANCED_SCHEDULE_PATH"]) for s in (1,2)]
    if schedule_paths[0].resolve() != schedule_paths[1].resolve():
        raise ValueError("Different schedule files")
    schedule_hash = sha(schedule_paths[0])
    if any(a["schedule_sha256"] != schedule_hash for a in audits.values()):
        raise ValueError("Schedule hash mismatch")
    with np.load(schedule_paths[0]) as f:
        plan = {k: np.asarray(f[k]) for k in f.files}
    paths = sorted(Path(configs[1]["ENV_KWARGS"]["layouts_dir"]).glob("*.json"))
    oracle_metadata = json.loads((ORACLES/"summary.json").read_text())
    if len(paths) != 1096 or not json.loads((ORACLES/"validation.json").read_text())["passed"]:
        raise ValueError("Need the validated full-corpus task ceilings")
    oracle_rows = read_csv(ORACLES/"per_layout_profile.csv")
    oracle = {(int(r["layout_index"]),int(r["d_R"]),int(r["d_B"])): r for r in oracle_rows}
    for s in (1,2):
        if configs[s]["NUM_ENVS"] != 256 or configs[s]["NUM_STEPS"] != 256:
            raise ValueError("Unexpected rollout dimensions")
    # Initial queue assigns episode IDs 0...255 to the same worker IDs.
    assigned = []
    for slot in range(256):
        packet, offset = divmod(slot,24)
        cap = plan["capability_pairs"][plan["profile_order"][packet,offset]]
        for rnd, index in enumerate(plan["layouts"][packet]):
            r = oracle[(int(index),*map(int,cap))]
            red, blue = int(r["fixed_partner_red_steps"]), int(r["fixed_partner_blue_steps"])
            assigned.append(dict(worker=slot, episode_id=slot, packet=packet, round_index=rnd,
                layout_index=int(index), layout_id=r["layout_id"], red_delay=int(cap[0]), blue_delay=int(cap[1]),
                best_informed_completion_steps=min(red,blue), partner_red_completion_steps=red,
                partner_blue_completion_steps=blue,
                informed_random_assignment_return=float(r["fixed_random_assignment_return"])))
    used = {r["layout_index"] for r in assigned}
    reference = {Path(r["path"]).name:r["sha256"] for r in oracle_metadata["layout_records"]}
    for index in used:
        if sha(paths[index]) != reference[paths[index].name]:
            raise ValueError("Layout/oracle provenance mismatch")
    write_csv(out/"initial_worker_packets.csv", assigned)
    first = [r for r in assigned if r["round_index"]==0]
    grids = []
    for index in dict.fromkeys(r["layout_index"] for r in first):
        group = [r for r in first if r["layout_index"] == index]
        raw = json.loads(paths[index].read_text())
        m = raw["metadata"]
        grids.append(dict(packet=group[0]["packet"], layout_index=index, layout_id=group[0]["layout_id"],
            workers=len(group), ego_red_distance=m["ego_to_red"], ego_blue_distance=m["ego_to_blue"],
            partner_red_distance=m["partner_to_red"], partner_blue_distance=m["partner_to_blue"],
            dead_ends=m["num_dead_ends"], wall_fraction=m["realized_wall_density"],
            mean_best_informed_steps=float(np.mean([r["best_informed_completion_steps"] for r in group])),
            min_best_informed_steps=min(r["best_informed_completion_steps"] for r in group),
            max_best_informed_steps=max(r["best_informed_completion_steps"] for r in group),
            mean_informed_random_assignment_return=float(np.mean([r["informed_random_assignment_return"] for r in group]))))
    write_csv(out/"initial_round_grids.csv", grids)
    # Compare task difficulty under the actually-started cumulative exposures.
    pairs = plan["capability_pairs"]
    best = np.array([[min(int(oracle[(i,*map(int,c))]["fixed_partner_red_steps"]),
                              int(oracle[(i,*map(int,c))]["fixed_partner_blue_steps"]))
                     for i in range(1096)] for c in pairs])
    random_return = np.array([[float(oracle[(i,*map(int,c))]["fixed_random_assignment_return"])
                              for i in range(1096)] for c in pairs])
    exposure = []
    for s in (1,2):
        with np.load(resolve_path(audits[s]["arrays"])) as f:
            weights = np.asarray(f["rounds_started"])
        exposure.append(dict(seed=s, scope="all_training_started_rounds", rounds=int(weights.sum()),
            weighted_best_informed_steps=float((weights*best).sum()/weights.sum()),
            weighted_informed_random_assignment_return=float((weights*random_return).sum()/weights.sum())))
    write_csv(out/"cumulative_exposure_difficulty.csv", exposure)
    histories = history()
    phases = []
    for lo,hi in PHASES:
        for s in (1,2):
            rs = [r for r in histories[s] if lo<=r["update"]<=hi]
            n = sum(r["rounds_completed"] for r in rs)
            phases.append(dict(seed=s, first_update=lo, last_update=hi,
                million_steps_start=(lo-1)*65536/1e6,million_steps_end=hi*65536/1e6,
                rounds_completed=int(n), steps_per_completed_round=len(rs)*65536/n,
                approximate_round_success=sum(r["round_success"]*r["rounds_completed"] for r in rs)/n))
    write_csv(out/"early_learning_windows.csv", phases)
    metadata = dict(seeds=[1,2], condition="rnn_diverse_influence", version="v1_matched",
        initial_workers=256, unique_initial_grids=len(grids), initial_assigned_rounds=len(assigned),
        identical_initial_worker_grid_profile_and_packet=True,
        schedule_path=str(schedule_paths[0]), schedule_sha256=schedule_hash,
        initial_packet_grid_count=len(used), initial_best_informed_steps_mean=float(np.mean([r["best_informed_completion_steps"] for r in first])),
        all_corpus_best_informed_steps_mean=float(best.mean()),
        cumulative_exposure=exposure,
        source_hashes={str(p):sha(p) for p in [Path(__file__),
            SOURCE/"baselines/IPPO/counterbalanced_scheduler.py", SOURCE/"baselines/IPPO/ippo_rnn_coordination_grid.py",
            SOURCE/"jaxmarl/environments/coordination_grid/coordination_grid.py",
            ORACLES/"per_layout_profile.csv", HISTORIES/"updates.csv"]},
        caveats=["Only final cumulative per-layout exposure matrices were saved; exact early post-update exposure matrices and action trajectories were not saved.",
            "Initial schedule reconstruction is exact. First 20 layouts per initially assigned worker are paired, but workers progress at policy-dependent speeds.",
            "Oracle completion is a physical lower bound with full capability information; it does not measure novice exploration difficulty.",
            "Cumulative exposure statistics cover the full training run, not the first updates.",
            "The initial CPU rollout is a reproduction; matching historical aggregates cannot certify every historical action."])
    return configs,plan,paths,oracle,first,grids,histories,phases,metadata


def initial_replay(out, config, oracle, histories):
    # Execute a copy in memory; preserve the original configuration, network,
    # initial PRNG construction, scheduled resets, action draws and transitions.
    path = SOURCE/"baselines/IPPO/ippo_rnn_coordination_grid.py"
    original = path.read_text()
    anchor = "            # --------- bootstrap value + GAE ---------"
    fields = {name:f'traj_batch.info["{name}"]' for name in
              ("layout_idx","round_idx","round_time","round_done","success","capability")}
    fields.update(action="traj_batch.action",reward="traj_batch.reward",pre_step_time="traj_batch.pre_step_time",
        ego_cell='jnp.argmax(traj_batch.obs["grid"][...,3].reshape(config["NUM_STEPS"],config["NUM_ENVS"],-1),axis=-1)',
        partner_cell='jnp.argmax(traj_batch.obs["grid"][...,4].reshape(config["NUM_STEPS"],config["NUM_ENVS"],-1),axis=-1)')
    statement = "            return runner_state, {\n" + "".join(
        f"                {name!r}: {value},\n" for name,value in fields.items()) + "            }\n\n"
    scan = '            _update_step, runner_state, None, config["NUM_UPDATES"]'
    if original.count(anchor)!=1 or original.count(scan)!=1:
        raise ValueError("Unexpected frozen source")
    changed = original.replace(anchor,statement+anchor).replace(scan,'            _update_step, runner_state, None, 1')
    (out/"initial_rollout_instrumentation.diff").write_text("".join(difflib.unified_diff(
        original.splitlines(True),changed.splitlines(True),fromfile="frozen_trainer",tofile="in_memory_collection_only")))
    sys.path.insert(0,str(SOURCE))
    sys.path.insert(0,str(SOURCE/"baselines/IPPO"))
    import jax
    import jax.numpy as jnp
    module = types.ModuleType("_initial_grid_diagnostic_trainer")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(changed,str(path),"exec"),module.__dict__)
    fn = jax.jit(jax.vmap(module.make_train(dict(config))))
    rounds, samples, summaries = [], [], []
    for seed in (1,2):
        # Original main uses split(PRNGKey(SEED), NUM_SEEDS), with NUM_SEEDS=1.
        keys = jax.random.split(jax.random.PRNGKey(seed),1)
        result = fn(keys)
        a = {k:np.asarray(v)[0,0] for k,v in result["metrics"].items()}
        for slot in range(256):
            start = 0
            for end in np.flatnonzero(a["round_done"][:,slot]):
                index = int(a["layout_idx"][start,slot])
                dr,db = map(int,a["capability"][start,slot])
                o = oracle[(index,dr,db)]
                alloc = int(a["action"][start,slot]%3)
                chosen = int(o["fixed_partner_blue_steps" if alloc==1 else "fixed_partner_red_steps"])
                rounds.append(dict(seed=seed,worker=slot,round_index=int(a["round_idx"][start,slot]),
                    start_worker_tick=start,end_worker_tick=int(end),layout_index=index,layout_id=o["layout_id"],
                    red_delay=dr,blue_delay=db,initial_ego_allocation=alloc,
                    actual_steps=int(end-start+1),success=bool(a["success"][end,slot]),
                    chosen_assignment_oracle_steps=chosen,
                    best_assignment_oracle_steps=min(int(o["fixed_partner_red_steps"]),int(o["fixed_partner_blue_steps"]))))
                start = int(end)+1
            # Retain round-zero paths for one worker in each initial grid packet.
            if slot%24==0:
                stop = int(np.flatnonzero(a["round_done"][:,slot])[0])
                for tick in range(stop+1):
                    samples.append(dict(seed=seed,worker=slot,tick=tick,layout_index=int(a["layout_idx"][tick,slot]),
                        ego_cell=int(a["ego_cell"][tick,slot]),partner_cell=int(a["partner_cell"][tick,slot]),
                        action=int(a["action"][tick,slot]),reward=float(a["reward"][tick,slot]),
                        success=bool(a["success"][tick,slot])))
        rs = [r for r in rounds if r["seed"]==seed]
        initial = [r for r in rs if r["round_index"]==0]
        historical = histories[seed][0]
        count,len0 = len(rs),len(initial)
        if len0!=256:
            raise ValueError("Every initial round must finish within 100 ticks")
        success = float(np.mean([r["success"] for r in rs]))
        summaries.append(dict(seed=seed,replayed_rounds=count,replayed_success=success,
            historical_rounds=int(historical["rounds_completed"]),historical_success=historical["round_success"],
            matches_historical_printed_aggregates=bool(count==historical["rounds_completed"] and abs(success-historical["round_success"])<=.00051),
            initial_rounds=len0,initial_round_success=float(np.mean([r["success"] for r in initial])),
            initial_round_mean_steps=float(np.mean([r["actual_steps"] for r in initial])),
            initial_chosen_assignment_oracle_mean_steps=float(np.mean([r["chosen_assignment_oracle_steps"] for r in initial]))))
        print("Initial rollout reproduction",summaries[-1],flush=True)
    write_csv(out/"initial_rollout_rounds.csv",rounds)
    write_csv(out/"initial_round_example_paths.csv",samples)
    replay = dict(backend="cpu",jax_version=jax.__version__,gradient_updates=0,
        collection_transitions_per_seed=65536,original_config_ppo_updates=915,
        frozen_source_unchanged=sha(path)==hashlib.sha256(original.encode()).hexdigest(),summaries=summaries)
    (out/"initial_rollout_summary.json").write_text(json.dumps(replay,indent=2)+"\n")
    return replay


def random_baseline(out, first, paths, trials=256):
    """Measure novice exploration difficulty with a fixed, seed-neutral policy.

    Equal weight to both initial assignments; each of five moves is uniform.
    Use the validated oracle's geometry and independently check vectorized
    fixed-allocation transitions against its scalar transition implementation.
    """
    tree=ast.parse((ROOT/"eval/optimal_training_return.py").read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("geometry","transition")]
    for node in nodes:
        node.decorator_list=[]
    scope=dict(np=np,json=json,Path=Path,deque=deque,DELTAS=((0,-1),(0,1),(1,0),(-1,0),(0,0)))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<validated oracle geometry/transition>","exec"),scope)
    indices=list(dict.fromkeys(r["layout_index"] for r in first))
    geometries=[scope["geometry"](paths[i]) for i in indices]
    moves=np.stack([g[1] for g in geometries])
    next_action=np.stack([g[2] for g in geometries])
    goals=np.stack([g[4] for g in geometries])
    starts=np.array([g[5] for g in geometries])
    mapping={index:i for i,index in enumerate(indices)}

    def vector_step(layout,ego,partner,goal,counter,action,caps):
        delay=caps[np.arange(len(goal)),goal]
        pmove=next_action[layout,goal,partner]
        next_partner=np.where(counter==0,moves[layout,partner,pmove],partner)
        next_counter=np.where(counter==0,delay,np.maximum(counter-1,0))
        next_ego=moves[layout,ego,action]
        blocked=(next_ego==next_partner)|((next_ego==partner)&(next_partner==ego))
        return np.where(blocked,ego,next_ego),np.where(blocked,partner,next_partner),next_counter

    rng=np.random.default_rng(20261015)
    # Verify arbitrary collision/delay states, including cells away from starts.
    for _ in range(512):
        li=int(rng.integers(len(indices)))
        _,mv,pn,dist,_,_=geometries[li]
        free=np.flatnonzero(np.all(dist>=0,axis=0))
        e,p=map(int,rng.choice(free,2,replace=False))
        c=np.array([int(rng.integers(1,10)),int(rng.integers(1,10))])
        goal,ctr,action=map(int,(rng.integers(2),rng.integers(10),rng.integers(5)))
        expected=scope["transition"](e,p,goal,ctr,action,goal,c,mv,pn,False)
        actual=vector_step(np.array([li]),np.array([e]),np.array([p]),np.array([goal]),np.array([ctr]),np.array([action]),c[None])
        if tuple(int(x[0]) for x in actual)!=(expected[0],expected[1],expected[3]):
            raise ValueError("Random baseline dynamics disagree with validated transition")
    if trials%2 or trials<2:
        raise ValueError("Use a positive even number of trials")
    layout=np.repeat([mapping[r["layout_index"]] for r in first],trials)
    caps=np.repeat([[r["red_delay"],r["blue_delay"]] for r in first],trials,axis=0)
    goal=np.tile(np.repeat(np.arange(2),trials//2),len(first))
    ego,partner=starts[layout,0].copy(),starts[layout,1].copy()
    counter=np.zeros(len(layout),int)
    success=np.zeros(len(layout),bool)
    length=np.ones(len(layout),int)
    for tick in range(2,101):
        ego,partner,counter=vector_step(layout,ego,partner,goal,counter,rng.integers(5,size=len(layout)),caps)
        alive=~success
        length[alive]=tick
        success|=((ego==goals[layout,0])&(partner==goals[layout,1]))|((ego==goals[layout,1])&(partner==goals[layout,0]))
    success,length=success.reshape(-1,trials),length.reshape(-1,trials)
    rows=[]
    for i,r in enumerate(first):
        rows.append(dict(layout_index=r["layout_index"],layout_id=r["layout_id"],red_delay=r["red_delay"],
            blue_delay=r["blue_delay"],trials=trials,random_policy_success=float(success[i].mean()),
            random_policy_mean_steps=float(length[i].mean()),
            random_policy_round_return=float((1.01*success[i]-.01*length[i]).mean())))
    write_csv(out/"random_policy_grid_profile.csv",rows)
    grid_rows=[]
    for index in indices:
        group=[r for r in rows if r["layout_index"]==index]
        grid_rows.append(dict(layout_index=index,layout_id=group[0]["layout_id"],profiles=len(group),
            trials_per_profile=trials,random_policy_success=float(np.mean([r["random_policy_success"] for r in group])),
            random_policy_mean_steps=float(np.mean([r["random_policy_mean_steps"] for r in group]))))
    write_csv(out/"random_policy_grid_summary.csv",grid_rows)
    result=dict(policy="Uniform fixed RED/BLUE allocation, then uniform five-way movement; frozen scripted partner.",
        seed=20261015,trials_per_initial_worker_profile=trials,total_rounds=len(first)*trials,
        vectorized_transition_checks=512,grid_summaries=grid_rows,
        overall_success=float(success.mean()),overall_mean_steps=float(length.mean()))
    (out/"random_policy_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def figures(out,paths,grids,histories):
    fig,axes = plt.subplots(3,4,figsize=(13,10))
    for ax,g in zip(axes.flat,grids):
        raw = json.loads(paths[g["layout_index"]].read_text())
        walls = np.asarray(raw["grid"])==1
        ax.imshow(walls,cmap="Greys",vmin=0,vmax=1)
        for name,label,color in (("red_goal","R","#d64743"),("blue_goal","B","#297bc0"),
                                 ("ego_start","E","#b2760c"),("partner_start","P","#159354")):
            y,x = raw[name]
            ax.scatter(x,y,s=220,c=color,marker="s" if "goal" in name else "o",edgecolor="white")
            ax.text(x,y,label,color="white",ha="center",va="center",fontsize=9,weight="bold")
        ax.set_title(f"Packet {g['packet']}: {g['layout_id']} · {g['workers']} workers\n"
                     f"Best informed: {g['min_best_informed_steps']}–{g['max_best_informed_steps']} ticks",fontsize=9)
        ax.set_xticks([]);ax.set_yticks([])
    for ax in axes.flat[len(grids):]:
        ax.axis("off")
    fig.suptitle("Exact initial grids shared by v1_matched seeds 1 and 2\nE: ego start; P: partner start; R/B: goals",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(out/"shared_initial_grids.png",dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for seed,color in ((1,"#227ab7"),(2,"#d77a26")):
        rs=histories[seed][:92]
        x=np.array([r["env_steps"] for r in rs])/1e6
        axes[0].plot(x,[r["round_success"]*100 for r in rs],label=f"Seed {seed}",color=color)
        axes[1].plot(x,[65536/r["rounds_completed"] for r in rs],label=f"Seed {seed}",color=color)
    axes[0].set_ylabel("Completed-round success (%)")
    axes[1].set_ylabel("Transitions per completed round")
    for ax in axes:
        ax.set_xlabel("Training transitions (millions)");ax.grid(alpha=.2);ax.legend()
    fig.suptitle("Historical logs: similar initial performance, divergence after approximately 0.8M steps")
    fig.tight_layout()
    fig.savefig(out/"early_learning.png",dpi=160)
    plt.close(fig)
    baseline_path=out/"random_policy_grid_summary.csv"
    if baseline_path.exists():
        baseline=read_csv(baseline_path)
        examples=[max(baseline,key=lambda r:float(r["random_policy_success"])),
                  min(baseline,key=lambda r:float(r["random_policy_success"]))]
        fig,axes=plt.subplots(1,2,figsize=(8,4.5))
        for ax,r in zip(axes,examples):
            raw=json.loads(paths[int(r["layout_index"])].read_text())
            ax.imshow(np.asarray(raw["grid"])==1,cmap="Greys",vmin=0,vmax=1)
            for name,label,color in (("red_goal","R","#d64743"),("blue_goal","B","#297bc0"),
                                     ("ego_start","E","#b2760c"),("partner_start","P","#159354")):
                y,x=raw[name]
                ax.scatter(x,y,s=330,c=color,marker="s" if "goal" in name else "o",edgecolor="white")
                ax.text(x,y,label,color="white",ha="center",va="center",fontsize=11,weight="bold")
            ax.set_title(f"{r['layout_id']}\nUniform-movement success: {100*float(r['random_policy_success']):.1f}%")
            ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle("Two initial grids seen by both seeds\nE: ego start; P: partner start; R/B: goals",fontsize=12)
        fig.tight_layout(rect=(0,0,1,.88))
        fig.savefig(out/"difficulty_examples.png",dpi=150)
        plt.close(fig)


def report(out,grids,phases,metadata,replay):
    grid_table="".join(f"<tr><td>{g['packet']}</td><td>{g['layout_id']}</td><td>{g['workers']}</td>"
        f"<td>{g['min_best_informed_steps']}–{g['max_best_informed_steps']}</td><td>{g['mean_best_informed_steps']:.2f}</td></tr>" for g in grids)
    phase_table="".join(f"<tr><td>{r['first_update']}–{r['last_update']}</td><td>{r['seed']}</td>"
        f"<td>{100*r['approximate_round_success']:.1f}%</td><td>{r['steps_per_completed_round']:.1f}</td></tr>" for r in phases)
    exposure_table="".join(f"<tr><td>{r['seed']}</td><td>{r['weighted_best_informed_steps']:.4f}</td>"
        f"<td>{r['weighted_informed_random_assignment_return']:.5f}</td></tr>" for r in metadata["cumulative_exposure"])
    caveats="".join(f"<li>{html.escape(x)}</li>" for x in metadata["caveats"])
    replay_section=""
    if replay:
        replay_table="".join(f"<tr><td>{r['seed']}</td><td>{r['initial_round_success']*100:.1f}%</td>"
            f"<td>{r['initial_round_mean_steps']:.2f}</td><td>{r['initial_chosen_assignment_oracle_mean_steps']:.2f}</td>"
            f"<td>{r['replayed_rounds']}/{r['historical_rounds']}</td><td>{r['matches_historical_printed_aggregates']}</td></tr>" for r in replay["summaries"])
        replay_section=f"""<h2>Reproduction before the first gradient update</h2><p>A copy of the frozen trainer collected
one original-size rollout per seed with its original RNG construction. No optimizer update was executed. Collection,
scheduled resets, network initialization, and action sampling were retained. This is a CPU reproduction, not an unsaved historical trace.</p>
<table><tr><th>Seed</th><th>First 256 rounds success</th><th>First-round mean ticks</th><th>Chosen-assignment physical optimum</th><th>Rollout completions reproduced/historical</th><th>Matches printed aggregates</th></tr>{replay_table}</table>
<p>The task optimum for a chosen assignment can differ between seeds because assignment is an action.
Such a difference is caused by policy choices, not by a different grid allocation.</p>"""
    baseline=metadata.get("random_policy_baseline")
    baseline_section=""
    if baseline:
        baseline_table="".join(f"<tr><td>{r['layout_id']}</td><td>{100*r['random_policy_success']:.1f}%</td>"
            f"<td>{r['random_policy_mean_steps']:.1f}</td></tr>" for r in baseline["grid_summaries"])
        baseline_section=f"""<h2>Difficulty for an untrained-style controller</h2>
<p>A uniform random movement policy is a complementary measure of exploration difficulty. It chooses each initial
assignment equally often, then each of five movements uniformly. I simulated {baseline['total_rounds']:,} rounds on
the exact initial grid/profile pairs with 256 trials per pair. The deterministic movement/delay/collision implementation
passed 512 comparisons with the validated scalar transition. This controller has no learned parameters and is shared
as a neutral reference for both seeds. These are Monte Carlo estimates, not learned-policy scores.</p>
<img src="difficulty_examples.png" alt="Two shared grids with different uniform-movement difficulty">
<table><tr><th>Shared initial grid</th><th>Random-controller success</th><th>Mean ticks</th></tr>{baseline_table}</table>"""
    document=f"""<!doctype html><meta charset="utf-8"><title>Initial grid exposure: seeds 1 and 2</title>
<style>body{{font:16px/1.55 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#202634}}img{{width:100%}}table{{border-collapse:collapse;font-size:14px;width:100%}}td,th{{border:1px solid #ccd3dd;padding:7px;text-align:left}}</style>
<h1>Did v1_matched seed 2 receive harder early grids?</h1>
<p><b>The initial assignment was identical, worker by worker.</b> Both jobs used the same frozen schedule, verified against
the recorded SHA-256. Worker IDs 0–255 start with episode IDs 0–255 in both jobs. Each episode's partner capability and
20-layout sequence are determined solely by that shared schedule. Learner seed does not enter the lookup.</p>
<p>The first 256 rounds therefore use the same 11 grids with the same profile attached to each worker. The initial
20-layout packet per worker is also shared: 5,120 assigned round positions involving {metadata['initial_packet_grid_count']} distinct layouts.
This does not mean the first 5,120 rounds to finish globally are identical, because workers progress at different speeds.</p>
<img src="shared_initial_grids.png" alt="Shared initial grid stimuli">
<h2>Some grids are physically harder, but both seeds receive those same cases</h2>
<p>Difficulty below is the exact minimum successful completion time for a fully informed controller choosing its
fixed assignment and navigating with the executed partner, movement delays and collisions. Values vary with capability.
They do not measure how difficult it is for a novice policy to discover a successful path. The first-round mean optimum
is {metadata['initial_best_informed_steps_mean']:.2f} ticks, versus {metadata['all_corpus_best_informed_steps_mean']:.2f}
over the whole uniformly weighted layout/profile corpus.</p>
<table><tr><th>Initial packet</th><th>Grid</th><th>Workers in each seed</th><th>Best informed ticks: range over assigned profiles</th><th>Mean</th></tr>{grid_table}</table>
{baseline_section}
<h2>The early historical performance is initially similar</h2><img src="early_learning.png" alt="Early training success and round lengths">
<p>At update 1, seed 1 completes 641 rounds with 38.1% success; seed 2 completes 650 with 37.8% success.
Across updates 1–12 (the first 0.79M transitions), their weighted success rates are approximately 37.7% and 38.0%.
The large difference develops after that, during learning. Success uses rounded per-update values weighted by completed rounds.</p>
<table><tr><th>PPO updates</th><th>Seed</th><th>Approximate success</th><th>Transitions/completed round</th></tr>{phase_table}</table>
{replay_section}
<h2>What can change after initialization?</h2>
<p>Within each worker's assigned partner episode, the grid order stays fixed. Faster workers advance farther through
their packets. A worker completing all 20 rounds takes the next unused episode from a common asynchronous queue.
Thus different behavior changes the mix of grids present at the same training update. Later grid-mix differences can
amplify a learning difference, even though the initial curricula are paired. They are not evidence of a separate random
grid draw for learner seed 2.</p>
<p>As a supplementary check, the full-run started-round exposures have these weighted physical difficulties. This
comparison is cumulative and cannot reconstruct the early mixture.</p>
<table><tr><th>Seed</th><th>Best informed completion ticks</th><th>Informed navigation with random assignment: round return</th></tr>{exposure_table}</table>
<h2>What this establishes</h2><p>The explanation that seed 2 was initially allocated a harder curriculum is unsupported:
the initial stimuli are paired exactly, and recorded performance starts almost the same. The subsequent divergence is
consistent with differences in learning from those experiences. Initial parameters, action draws and optimization
minibatch permutations all depend on the learner RNG, so they remain distinct candidates. The present evidence does
not isolate which one causes the divergence or exclude feedback from later exposure mixtures.</p>
<p>A direct next test would reproduce the first roughly 20–30 PPO updates while logging per-worker grid/profile,
assignment, success, path length and optimizer diagnostics. Exchange parameter initialization, action RNG, and minibatch
RNG in separate paired runs. A fixed early grid/profile panel would compare learned policies at equal updates.</p>
<h2>Limits and evidence</h2><ul>{caveats}</ul><p><a href="audit.json">Provenance</a> ·
<a href="initial_worker_packets.csv">Exact initial worker packets</a> · <a href="initial_round_grids.csv">Grid difficulty</a> ·
<a href="early_learning_windows.csv">Historical early windows</a> · <a href="cumulative_exposure_difficulty.csv">Cumulative exposure</a></p></html>"""
    (out/"report.html").write_text(document)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=ROOT/"train/seed_diagnostics/v1_matched_v2_v3/initial_stimuli")
    parser.add_argument("--replay-initial",action="store_true")
    parser.add_argument("--reuse-initial-rollout",action="store_true")
    args=parser.parse_args()
    out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    configs,plan,paths,oracle,first,grids,histories,phases,metadata=setup(out)
    print("Verified identical initial grid/profile packets",flush=True)
    if args.replay_initial and args.reuse_initial_rollout:
        parser.error("Choose replay or reuse")
    replay=initial_replay(out,configs[1],oracle,histories) if args.replay_initial else None
    if args.reuse_initial_rollout:
        replay=json.loads((out/"initial_rollout_summary.json").read_text())
    metadata["random_policy_baseline"]=random_baseline(out,first,paths)
    figures(out,paths,grids,histories)
    metadata["initial_rollout_reproduction"]=replay
    (out/"audit.json").write_text(json.dumps(metadata,indent=2)+"\n")
    report(out,grids,phases,metadata,replay)
    print(json.dumps({k:metadata[k] for k in ("unique_initial_grids","initial_packet_grid_count",
        "initial_best_informed_steps_mean","all_corpus_best_informed_steps_mean","cumulative_exposure")},indent=2))


if __name__=="__main__":
    main()
