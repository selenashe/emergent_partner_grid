"""Audit empirical grid exposure during v1_matched PPO updates 13--20.

Requires instrumented prefix traces from diagnostics/replay_early_exposure.py.
Independently score stimuli with exact task ceilings and a shared random
controller. Separate presented starts, completed rounds and transition weights.
"""

import argparse
import ast
from collections import deque
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from repo_paths import resolve_path
from diagnostics.replay_early_exposure import DEFAULT_OUT, CONFIGS, SOURCE

ORACLES=ROOT/"eval/training_return_ceiling/v1_matched_v2_v3"
LO,HI=13,20


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs():
    config=json.loads((CONFIGS/"rnn_diverse_influence_seed1_config.json").read_text())
    with np.load(resolve_path(config["COUNTERBALANCED_SCHEDULE_PATH"])) as f:
        plan={k:np.asarray(f[k]) for k in f.files}
    paths=sorted(resolve_path(config["ENV_KWARGS"]["layouts_dir"]).glob("*.json"))
    metadata=json.loads((ORACLES/"summary.json").read_text())
    if not json.loads((ORACLES/"validation.json").read_text())["passed"]:
        raise ValueError("Exact task-ceiling validation did not pass")
    expected={Path(r["path"]).name:r["sha256"] for r in metadata["layout_records"]}
    if len(paths)!=1096 or any(sha(p)!=expected[p.name] for p in paths):
        raise ValueError("Corpus no longer matches the oracle's validated inputs")
    oracle={(int(r["layout_index"]),int(r["d_R"]),int(r["d_B"])):r
            for r in read_csv(ORACLES/"per_layout_profile.csv")}
    return plan,paths,oracle


def build_reference(out,trials=128):
    destination=out/"stimulus_reference.csv"
    meta_path=out/"stimulus_reference.json"
    if destination.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta["trials_per_case"]!=trials or meta["csv_sha256"]!=sha(destination):
            raise ValueError("Reference exists with different settings or bytes")
        print("Verified existing stimulus reference",flush=True)
        return
    plan,paths,oracle=inputs()
    tree=ast.parse((ROOT/"eval/optimal_training_return.py").read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("geometry","transition")]
    for n in nodes:n.decorator_list=[]
    scope=dict(np=np,json=json,Path=Path,deque=deque,DELTAS=((0,-1),(0,1),(1,0),(-1,0),(0,0)))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<validated oracle primitives>","exec"),scope)
    geometries=[scope["geometry"](p) for p in paths]
    moves=np.stack([g[1] for g in geometries])
    pn=np.stack([g[2] for g in geometries])
    goals=np.stack([g[4] for g in geometries])
    starts=np.array([g[5] for g in geometries])
    caps=plan["capability_pairs"]

    def step(layout,ego,partner,goal,counter,action,cap):
        delay=cap[np.arange(len(goal)),goal]
        proposed_partner=np.where(counter==0,moves[layout,partner,pn[layout,goal,partner]],partner)
        proposed_ego=moves[layout,ego,action]
        blocked=(proposed_ego==proposed_partner)|((proposed_ego==partner)&(proposed_partner==ego))
        return (np.where(blocked,ego,proposed_ego),np.where(blocked,partner,proposed_partner),
                np.where(counter==0,delay,np.maximum(counter-1,0)))

    rng=np.random.default_rng(20261015)
    for _ in range(512):
        li=int(rng.integers(len(paths)))
        _,mv,nxt,dist,_,_=geometries[li]
        e,p=map(int,rng.choice(np.flatnonzero(np.all(dist>=0,axis=0)),2,replace=False))
        cap=caps[int(rng.integers(24))]
        goal,counter,action=map(int,(rng.integers(2),rng.integers(10),rng.integers(5)))
        expected=scope["transition"](e,p,goal,counter,action,goal,cap,mv,nxt,False)
        actual=step(np.array([li]),np.array([e]),np.array([p]),np.array([goal]),np.array([counter]),np.array([action]),cap[None])
        assert tuple(int(x[0]) for x in actual)==(expected[0],expected[1],expected[3])
    if trials%2 or trials<2:raise ValueError("Trials must be positive and even")
    rows=[]
    ncases=len(paths)*24
    for offset in range(0,ncases,512):
        case=np.arange(offset,min(ncases,offset+512))
        layouts=np.repeat(case//24,trials)
        cap=np.repeat(caps[case%24],trials,axis=0)
        goal=np.tile(np.repeat(np.arange(2),trials//2),len(case))
        ego,partner=starts[layouts,0].copy(),starts[layouts,1].copy()
        counter=np.zeros(len(layouts),int)
        success=np.zeros(len(layouts),bool)
        length=np.ones(len(layouts),int)
        for tick in range(2,101):
            ego,partner,counter=step(layouts,ego,partner,goal,counter,rng.integers(5,size=len(layouts)),cap)
            length[~success]=tick
            success|=((ego==goals[layouts,0])&(partner==goals[layouts,1]))|((ego==goals[layouts,1])&(partner==goals[layouts,0]))
        success=success.reshape(-1,trials);length=length.reshape(-1,trials)
        for i,c in enumerate(case):
            index,ci=divmod(int(c),24);dr,db=map(int,caps[ci])
            o=oracle[(index,dr,db)]
            raw=geometries[index][0];m=raw["metadata"]
            red,blue=int(o["fixed_partner_red_steps"]),int(o["fixed_partner_blue_steps"])
            rows.append(dict(case=c,layout_index=index,layout_id=o["layout_id"],profile=ci,red_delay=dr,blue_delay=db,
                oracle_best_steps=min(red,blue),oracle_partner_red_steps=red,oracle_partner_blue_steps=blue,
                oracle_random_assignment_return=float(o["fixed_random_assignment_return"]),
                random_success=float(success[i].mean()),random_steps=float(length[i].mean()),
                random_success_ego_blue=float(success[i,:trials//2].mean()),
                random_success_ego_red=float(success[i,trials//2:].mean()),
                ego_red_distance=m["ego_to_red"],ego_blue_distance=m["ego_to_blue"],
                partner_red_distance=m["partner_to_red"],partner_blue_distance=m["partner_to_blue"],
                wall_fraction=m["realized_wall_density"],dead_ends=m["num_dead_ends"],junctions=m["num_junctions"]))
        if offset%4096==0:print(f"Scored {len(rows)}/{ncases} grid/profile cases",flush=True)
    write_csv(destination,rows)
    meta=dict(cases=ncases,layouts=len(paths),profiles=24,trials_per_case=trials,
        simulated_rounds=ncases*trials,seed=20261015,vector_transition_checks=512,
        policy="Uniform RED/BLUE assignment and uniform five-way movement with the frozen scripted partner",
        csv_sha256=sha(destination),source_sha256=sha(__file__),oracle_source_sha256=sha(ROOT/"eval/optimal_training_return.py"),
        oracle_table_sha256=sha(ORACLES/"per_layout_profile.csv"),
        caveat="Shared Monte Carlo reference, not a learned-policy score or exact random-policy expectation.")
    meta_path.write_text(json.dumps(meta,indent=2)+"\n")


def reconstruct(out,seed,plan):
    provenance=json.loads((out/f"seed{seed}_provenance.json").read_text())
    path=resolve_path(provenance["trace"])
    if sha(path)!=provenance["trace_sha256"]:raise ValueError("Trace changed")
    with np.load(path) as f:a={k:np.asarray(f[k]) for k in f.files}
    updates,ticks,workers=a["layout_idx"].shape
    assert (updates,ticks,workers)==(20,256,256)
    pair_to_id={tuple(map(int,c)):i for i,c in enumerate(plan["capability_pairs"])}
    profile=np.zeros((updates,ticks,workers),np.int8)
    chosen=np.zeros_like(profile)
    episode=np.arange(workers);next_episode=workers
    active=np.full(workers,-1,int);assignment=np.zeros(workers,np.int8)
    rows=[]
    for u in range(updates):
        for t in range(ticks):
            packet,offset=episode//24,episode%24
            expected_profile=plan["profile_order"][packet,offset]
            expected_caps=plan["capability_pairs"][expected_profile]
            if not np.array_equal(a["capability"][u,t],expected_caps):raise ValueError("Queue capability mismatch")
            expected_layout=plan["layouts"][packet,a["round_idx"][u,t]]
            if not np.array_equal(a["layout_idx"][u,t],expected_layout):raise ValueError("Queue layout mismatch")
            assert np.array_equal(a["pre_step_time"][u,t]+1,a["round_time"][u,t])
            profile[u,t]=expected_profile
            for w in np.flatnonzero(a["pre_step_time"][u,t]==0):
                assignment[w]=a["action"][u,t,w]%3
                assert assignment[w] in (1,2)
                dr,db=map(int,expected_caps[w])
                rows.append(dict(seed=seed,encounter=len(rows)+1,worker=int(w),episode_id=int(episode[w]),
                    episode_round=int(a["round_idx"][u,t,w]),layout_index=int(expected_layout[w]),
                    profile=int(expected_profile[w]),red_delay=dr,blue_delay=db,
                    start_update=u+1,start_tick=u*256+t,end_update=None,end_tick=None,
                    chosen_ego_allocation=int(assignment[w]),steps=None,success=None))
                active[w]=len(rows)-1
            chosen[u,t]=assignment
            for w in np.flatnonzero(a["round_done"][u,t]):
                r=rows[active[w]]
                r.update(end_update=u+1,end_tick=u*256+t,steps=int(a["round_time"][u,t,w]),success=int(a["success"][u,t,w]))
                assert r["steps"]==r["end_tick"]-r["start_tick"]+1
            ended=np.flatnonzero(a["done"][u,t])
            episode[ended]=next_episode+np.arange(len(ended));next_episode+=len(ended)
    write_csv(out/f"seed{seed}_round_encounters.csv",rows)
    return a,profile,chosen,rows,provenance


def weighted_mean(features,weights):
    return {name:float(np.dot(values,weights)/weights.sum()) for name,values in features.items()}


def matched_summary(rounds,with_assignment=False):
    groups={}
    for seed in (1,2):
        counts={}
        for r in rounds[seed]:
            if r["end_update"] is None or not LO<=r["end_update"]<=HI:continue
            key=(r["layout_index"],r["profile"])+( (r["chosen_ego_allocation"],) if with_assignment else () )
            n,wins=counts.get(key,(0,0));counts[key]=(n+1,wins+r["success"])
        groups[seed]=counts
    common=sorted(set(groups[1])&set(groups[2]))
    weight=np.array([min(groups[1][k][0],groups[2][k][0]) for k in common],float)
    result={}
    rows=[]
    for k,w in zip(common,weight):
        n1,y1=groups[1][k];n2,y2=groups[2][k]
        rows.append(dict(layout_index=k[0],profile=k[1],chosen_ego_allocation=k[2] if with_assignment else None,
            seed1_count=n1,seed1_success=y1/n1,seed2_count=n2,seed2_success=y2/n2,weight=w))
    for s in (1,2):
        rate=np.array([groups[s][k][1]/groups[s][k][0] for k in common])
        result[f"seed{s}_success"]=float(np.dot(weight,rate)/weight.sum()) if len(common) else None
    result.update(cells=len(common),matched_weight=float(weight.sum()),
        stratification="grid,profile,chosen allocation" if with_assignment else "grid,profile",
        caveat="Descriptive reweighting on shared completed cases; worker history and censoring remain uncontrolled.")
    return result,rows


def analyze(out):
    plan,paths,oracle=inputs()
    reference=read_csv(out/"stimulus_reference.csv")
    reference.sort(key=lambda r:int(r["case"]))
    ncases=len(reference)
    names=("oracle_best_steps","oracle_random_assignment_return","random_success","random_steps",
           "ego_red_distance","ego_blue_distance","partner_red_distance","partner_blue_distance",
           "wall_fraction","dead_ends","red_delay","blue_delay")
    features={k:np.array([float(r[k]) for r in reference]) for k in names}
    data,profiles,choices,rounds,provenance={},{},{},{},{}
    summaries=[];counts={};per_update=[]
    for s in (1,2):
        data[s],profiles[s],choices[s],rounds[s],provenance[s]=reconstruct(out,s,plan)
        case=data[s]["layout_idx"].astype(int)*24+profiles[s]
        selected=case[LO-1:HI].ravel()
        counts[s]={"transitions":np.bincount(selected,minlength=ncases)}
        for scope in ("starts","completions"):
            rs=[r for r in rounds[s] if (LO<=r["start_update"]<=HI if scope=="starts" else r["end_update"] is not None and LO<=r["end_update"]<=HI)]
            counts[s][scope]=np.bincount([r["layout_index"]*24+r["profile"] for r in rs],minlength=ncases)
        for scope,weights in counts[s].items():
            summary=dict(seed=s,scope=scope,n=int(weights.sum()),unique_grids=int(np.count_nonzero(weights.reshape(-1,24).sum(axis=1))),
                unique_grid_profile_cases=int(np.count_nonzero(weights)),**weighted_mean(features,weights))
            summaries.append(summary)
        for u in range(20):
            weights=np.bincount(case[u][data[s]["pre_step_time"][u]==0],minlength=ncases)
            ended=data[s]["round_done"][u]
            per_update.append(dict(seed=s,update=u+1,million_steps=(u+1)*65536/1e6,
                success=float(data[s]["success"][u][ended].mean()),rounds_completed=int(ended.sum()),
                transitions_per_round=65536/ended.sum(),**weighted_mean(features,weights)))
    write_csv(out/"window_difficulty.csv",summaries)
    write_csv(out/"per_update_difficulty.csv",per_update)
    exposure=[]
    for i,r in enumerate(reference):
        if any(counts[s][scope][i] for s in (1,2) for scope in counts[s]):
            exposure.append({**r,**{f"seed{s}_{scope}":int(w[i]) for s in (1,2) for scope,w in counts[s].items()}})
    write_csv(out/"window_grid_profile_exposure.csv",exposure)
    matched,matched_rows=matched_summary(rounds,False)
    chosen_matched,chosen_rows=matched_summary(rounds,True)
    write_csv(out/"matched_grid_profile_success.csv",matched_rows)
    write_csv(out/"matched_grid_profile_assignment_success.csv",chosen_rows)
    actual={}
    assignment_rows=[]
    for s in (1,2):
        rs=[r for r in rounds[s] if r["end_update"] is not None and LO<=r["end_update"]<=HI]
        actual[s]=dict(rounds=len(rs),success=float(np.mean([r["success"] for r in rs])),mean_steps=float(np.mean([r["steps"] for r in rs])))
        for r in rs:
            o=oracle[(r["layout_index"],r["red_delay"],r["blue_delay"])]
            chosen=int(o["fixed_partner_blue_steps" if r["chosen_ego_allocation"]==1 else "fixed_partner_red_steps"])
            best=min(int(o["fixed_partner_blue_steps"]),int(o["fixed_partner_red_steps"]))
            assignment_rows.append(dict(seed=s,layout_index=r["layout_index"],profile=r["profile"],success=r["success"],
                actual_steps=r["steps"],best_steps=best,chosen_steps=chosen,
                chosen_assignment_feasible=chosen<=100,chosen_assignment_oracle_return=1-.01*(chosen-1) if chosen<=100 else -1,
                excess_steps_if_chosen_feasible=r["steps"]-chosen if chosen<=100 else None))
    write_csv(out/"completed_round_assignment_difficulty.csv",assignment_rows)
    assignment_summary=[]
    for s in (1,2):
        rs=[r for r in assignment_rows if r["seed"]==s]
        assignment_summary.append(dict(seed=s,mean_chosen_assignment_oracle_steps=float(np.mean([r["chosen_steps"] for r in rs])),
            chosen_assignment_feasible_fraction=float(np.mean([r["chosen_assignment_feasible"] for r in rs])),
            chosen_assignment_oracle_return=float(np.mean([r["chosen_assignment_oracle_return"] for r in rs])),
            mean_excess_steps_for_feasible_choices=float(np.mean([r["excess_steps_if_chosen_feasible"] for r in rs if r["chosen_assignment_feasible"]]))))
    write_csv(out/"assignment_summary.csv",assignment_summary)
    first_mismatch=None
    for r1,r2 in zip(rounds[1],rounds[2]):
        if (r1["layout_index"],r1["profile"])!=(r2["layout_index"],r2["profile"]):
            first_mismatch=dict(encounter=r1["encounter"],seed1=r1,seed2=r2);break
    distributions=[]
    for scope in counts[1]:
        p=counts[1][scope]/counts[1][scope].sum();q=counts[2][scope]/counts[2][scope].sum()
        distributions.append(dict(scope=scope,grid_profile_total_variation=float(.5*np.abs(p-q).sum()),
            grid_only_total_variation=float(.5*np.abs(p.reshape(-1,24).sum(axis=1)-q.reshape(-1,24).sum(axis=1)).sum())))
    result=dict(window_updates=[LO,HI],window_transitions=[(LO-1)*65536,HI*65536],
        historical_aggregate_prefix_matches={s:provenance[s]["matched_historical_aggregate_prefix_updates"] for s in (1,2)},
        provenance=provenance,actual_completed=actual,difficulty=summaries,assignment_summary=assignment_summary,
        matched_grid_profile=matched,matched_grid_profile_assignment=chosen_matched,
        distribution_differences=distributions,first_different_chronological_grid_profile=first_mismatch,
        reference_sha256=sha(out/"stimulus_reference.csv"),analysis_source_sha256=sha(__file__),
        caveats=["These are instrumented reproductions; historical early trajectories were not saved.",
            "Matching rounded historical aggregate metrics does not certify every historical action.",
            "Starts, completions and transition weighting answer different questions; long rounds increase transition exposure.",
            "Reference random-policy difficulty is Monte Carlo estimated and shared across both seeds.",
            "Chosen-assignment oracle time 101 is a sentinel for no successful route within the 100-tick horizon.",
            "Two fixed learner seeds do not support population-level claims about seed variability.",
            "Matching grid/profile and assignment does not control partner-episode history, policy state or boundary selection."])
    (out/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    figures(out,per_update)
    report(out,result)
    print(json.dumps({k:result[k] for k in ("historical_aggregate_prefix_matches","actual_completed","matched_grid_profile","matched_grid_profile_assignment")},indent=2))


def figures(out,rows):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for seed,color in ((1,"#237db5"),(2,"#d57d24")):
        rs=[r for r in rows if r["seed"]==seed];x=[r["million_steps"] for r in rs]
        for ax,name,label,scale in ((axes[0,0],"success","Observed round success (%)",100),
            (axes[0,1],"random_success","Shared random-controller success on presented cases (%)",100),
            (axes[1,0],"oracle_best_steps","Best informed completion time of presented cases",1),
            (axes[1,1],"transitions_per_round","Training transitions per completed round",1)):
            ax.plot(x,[r[name]*scale for r in rs],label=f"Seed {seed}",color=color,marker=".")
            ax.set_ylabel(label);ax.set_xlabel("Training transitions (millions)");ax.grid(alpha=.2)
            ax.axvspan(12*65536/1e6,20*65536/1e6,color="#999",alpha=.1)
    axes[0,0].legend();fig.suptitle("Observed learning and independent stimulus difficulty; shaded interval: updates 13–20")
    fig.tight_layout();fig.savefig(out/"window_comparison.png",dpi=160);plt.close(fig)


def report(out,result):
    table="".join(f"<tr><td>{r['scope']}</td><td>{r['seed']}</td><td>{r['n']:,}</td>"
        f"<td>{r['unique_grids']}</td><td>{r['oracle_best_steps']:.3f}</td><td>{100*r['random_success']:.2f}%</td>"
        f"<td>{r['dead_ends']:.3f}</td><td>{100*r['wall_fraction']:.2f}%</td></tr>" for r in result["difficulty"])
    matched="".join(f"<tr><td>{r['stratification']}</td><td>{r['cells']}</td><td>{r['matched_weight']:.0f}</td>"
        f"<td>{100*r['seed1_success']:.2f}%</td><td>{100*r['seed2_success']:.2f}%</td></tr>" for r in
        (result["matched_grid_profile"],result["matched_grid_profile_assignment"]))
    fidelity=result["historical_aggregate_prefix_matches"]
    warning=("Both reproductions match historical completed-round counts and rounded success through all 20 updates."
        if all(x==20 for x in fidelity.values()) else
        "At least one reproduction diverges from historical aggregate metrics before update 20. Treat the exposure results as new replicate data, not recovered historical exposure.")
    caveats="".join(f"<li>{html.escape(c)}</li>" for c in result["caveats"])
    starts={r["seed"]:r for r in result["difficulty"] if r["scope"]=="starts"}
    actual=result["actual_completed"]
    raw_gap=100*(actual[1]["success"]-actual[2]["success"])
    match=result["matched_grid_profile"]
    matched_gap=100*(match["seed1_success"]-match["seed2_success"])
    narrative=(f"The observed success gap is {raw_gap:.2f} percentage points. A shared random controller scores "
        f"{100*starts[1]['random_success']:.2f}% on seed 1's presented grid/profile mix and "
        f"{100*starts[2]['random_success']:.2f}% on seed 2's mix. After giving both learned policies identical "
        f"weights on shared completed grid/profile cases, their success gap is {matched_gap:.2f} percentage points. "
        "These comparisons describe stimulus composition and conditional behavior; they do not identify the cause of learning divergence.")
    doc=f"""<!doctype html><meta charset="utf-8"><title>Empirical grid exposure during updates 13–20</title>
<style>body{{font:16px/1.55 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#202634}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ccd3dd;padding:7px;text-align:left}}img{{width:100%}}</style>
<h1>Empirical grid difficulty: v1_matched seeds 1 and 2, updates 13–20</h1>
<p>{narrative}</p>
<p><b>{warning}</b> The original per-grid trajectories were not saved. These runs retain the original 60M-step
configuration and 915-update learning-rate schedule, exact 256×256 rollout dimensions, all optimizer settings,
frozen source, schedule and learner seeds. Only the outer scan stops after 20 updates and emits extra trace outputs.
The full trace verifies each grid/profile against its current worker episode in the asynchronous scheduler.</p>
<p>Window: transitions {(LO-1)*65536:,}–{HI*65536:,}. Historical aggregate-prefix agreement: seed 1,
{fidelity[1]} updates; seed 2, {fidelity[2]} updates. Counts and successes are checked at every update, not only the endpoint.</p>
<img src="window_comparison.png" alt="Observed success and independent grid difficulty during learning">
<h2>Independent difficulty of the cases actually presented</h2>
<p>Every layout/profile has an exact fully informed task optimum and a common neutral-controller baseline.
The baseline chooses each initial allocation equally and moves uniformly over five actions; its outcomes are
independent of either learned policy. It is Monte Carlo estimated. Geometry features describe walls, distances and dead ends.</p>
<table><tr><th>Weighting</th><th>Seed</th><th>Observations</th><th>Unique grids</th><th>Best informed ticks</th><th>Random-controller success</th><th>Dead ends</th><th>Walls</th></tr>{table}</table>
<p><b>Starts</b> count new rounds presented in the interval, including those unfinished at its end.
<b>Completions</b> count rounds finishing in the interval, including those begun earlier; these define the historical success rate.
<b>Transitions</b> weight every tick actually experienced. Longer unsuccessful rounds receive more weight in that last view,
so a transition-weighted difference can be a consequence of policy difficulty rather than a separate sampling cause.</p>
<h2>Behavior on shared cases</h2>
<p>These descriptive comparisons use only common completed cases, giving each cell weight min(seed-1 count, seed-2 count).
Thus both policies receive exactly the same grid/profile weighting. A second comparison additionally matches their chosen
allocation. Episode history and learned policy states are not controlled, so these are not causal estimates.</p>
<table><tr><th>Matched variables</th><th>Shared cells</th><th>Matched weight</th><th>Seed 1 success</th><th>Seed 2 success</th></tr>{matched}</table>
<h2>Exact encounter records</h2><p>The encounter CSVs enumerate each round start in time order, breaking simultaneous
starts by worker index. They include worker, episode ID, round, grid, capability, chosen allocation, start/end updates,
duration and outcome. They retain rounds crossing update boundaries and distinguish unfinished rounds.</p>
<p><a href="seed1_round_encounters.csv">Seed 1 encounters</a> · <a href="seed2_round_encounters.csv">Seed 2 encounters</a> ·
<a href="window_grid_profile_exposure.csv">Per-case exposure and difficulty</a> ·
<a href="window_difficulty.csv">Difficulty summary</a> · <a href="assignment_summary.csv">Chosen assignment difficulty</a> ·
<a href="seed1_historical_comparison.csv">Seed 1 historical comparison</a> ·
<a href="seed2_historical_comparison.csv">Seed 2 historical comparison</a> · <a href="audit.json">Audit metadata</a></p>
<h2>Limits</h2><ul>{caveats}</ul>"""
    (out/"report.html").write_text(doc)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("reference","analyze"))
    parser.add_argument("--out-dir",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--trials",type=int,default=128)
    args=parser.parse_args();out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.command=="reference":build_reference(out,args.trials)
    else:analyze(out)


if __name__=="__main__":main()
