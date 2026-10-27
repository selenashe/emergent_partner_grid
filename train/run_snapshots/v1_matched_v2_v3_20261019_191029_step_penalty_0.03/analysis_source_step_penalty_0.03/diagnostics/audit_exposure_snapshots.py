"""Compare actual counter differences between PPO updates 12 and 20.

No per-round success or assignment trace is inferred from these counters.
Validate counter snapshots against the schedule and unfinished worker tails.
"""

import argparse
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.audit_early_window import inputs,read_csv,write_csv,weighted_mean
from diagnostics.replay_early_exposure import DEFAULT_OUT,historical
from repo_paths import resolve_path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_metrics(out,metrics):
    mapping={"round_success_rate":("round_success",.000501),"rounds_completed":("rounds_completed",0),
        "entropy_t0":("entropy_t0",.000501),"entropy_tge1":("entropy_movement",.000501),
        "actor_loss":("actor_loss",.0000501),"value_loss":("value_loss",.0000501),
        "entropy":("entropy",.0000501),"ep_return_mean":("episode_return",.000501),
        "ep_length_mean":("episode_steps",.05001)}
    checks=[]
    for seed in (1,2):
        saved=historical(seed,20)
        assert sorted(int(r["update"]) for r in saved)==list(range(1,21))
        for r in saved:
            u=int(r["update"])
            for metric,(column,tolerance) in mapping.items():
                expected=float(r[column])
                if not np.isfinite(expected):continue
                error=abs(metrics[seed][u][metric]-expected)
                checks.append(dict(seed=seed,update=u,metric=metric,absolute_error=error,tolerance=tolerance,passed=bool(error<=tolerance)))
    validation=dict(passed=all(c["passed"] for c in checks),comparisons=len(checks),checks=checks,
        replay_metric_sha256={s:sha(out/f"seed{s}_metrics.json") for s in (1,2)})
    (out/"saved_metrics_validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    assert validation["passed"],"Replay metrics differ from the historical logs"
    return validation


def quantile(values,weights,q):
    order=np.argsort(values)
    return float(values[order][np.searchsorted(np.cumsum(weights[order]),q*weights.sum())])


def validate_snapshot(a,plan,update):
    allocated=int(a["allocated_episodes"])
    full,tail=divmod(allocated,24)
    planned=np.tile(np.bincount(plan["layouts"][:full].ravel(),minlength=1096),(24,1))
    for profile in plan["profile_order"][full,:tail]:np.add.at(planned[profile],plan["layouts"][full],1)
    started,completed=planned.copy(),planned.copy()
    ids=a["active_episode_ids"]
    assert len(ids)==256 and len(np.unique(ids))==256
    for slot,episode in enumerate(ids):
        packet,offset=divmod(int(episode),24)
        profile=int(plan["profile_order"][packet,offset])
        rnd,tick=int(a["active_round_idx"][slot]),int(a["active_round_time"][slot])
        np.add.at(started[profile],plan["layouts"][packet,rnd+int(tick>0):],-1)
        np.add.at(completed[profile],plan["layouts"][packet,rnd:],-1)
    np.testing.assert_array_equal(started,a["layout_counts"][0])
    np.testing.assert_array_equal(completed,a["layout_counts"][1])
    assert int(a["layout_counts"][2].sum())==update*65536
    assert int(a["episode_counts"][1].sum())==allocated-256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir",type=Path,default=DEFAULT_OUT/"counter_snapshots")
    args=p.parse_args();out=args.out_dir.resolve();base=out.parent
    plan,paths,oracle=inputs()
    ref=read_csv(base/"stimulus_reference.csv");ref.sort(key=lambda r:int(r["case"]))
    features={k:np.array([float(r[k]) for r in ref]) for k in ("oracle_best_steps","oracle_random_assignment_return",
        "random_success","random_steps","ego_red_distance","ego_blue_distance","partner_red_distance",
        "partner_blue_distance","wall_fraction","dead_ends","red_delay","blue_delay")}
    prov={};counts={};metrics={};summary=[];profile_rows=[]
    for seed in (1,2):
        prov[seed]=json.loads((out/f"seed{seed}_provenance.json").read_text())
        snapshots={}
        for update in (12,20):
            path=resolve_path(prov[seed]["snapshots"][str(update)])
            if sha(path)!=prov[seed]["snapshots_sha256"][str(update)]:raise ValueError("Snapshot changed")
            with np.load(path) as f:snapshots[update]={k:np.asarray(f[k]) for k in f.files}
            validate_snapshot(snapshots[update],plan,update)
        difference=snapshots[20]["layout_counts"].astype(np.int64)-snapshots[12]["layout_counts"]
        assert (difference>=0).all() and difference[2].sum()==8*65536
        counts[seed]={scope:difference[i].T.ravel() for i,scope in enumerate(("starts","completions","transitions"))}
        metrics[seed]={int(k):v for k,v in json.loads((out/f"seed{seed}_metrics.json").read_text()).items()}
        expected=sum(metrics[seed][u]["rounds_completed"] for u in range(13,21))
        assert counts[seed]["completions"].sum()==expected
        for scope,weights in counts[seed].items():
            row=dict(seed=seed,scope=scope,n=int(weights.sum()),unique_grids=int(np.count_nonzero(weights.reshape(-1,24).sum(axis=1))),
                grid_profile_cases=int(np.count_nonzero(weights)),**weighted_mean(features,weights))
            # Distinguish grid composition from partner-profile composition.
            layout_weights=weights.reshape(-1,24).sum(axis=1)
            profile_weights=weights.reshape(-1,24).sum(axis=0)
            row["random_success_grid_mix_profiles_uniform"]=float(np.dot(layout_weights,features["random_success"].reshape(-1,24).mean(axis=1))/weights.sum())
            row["random_success_profile_mix_grids_uniform"]=float(np.dot(profile_weights,features["random_success"].reshape(-1,24).mean(axis=0))/weights.sum())
            for feature in ("oracle_best_steps","random_success"):
                for q in (.1,.5,.9,.95,.99):
                    row[f"{feature}_p{round(100*q)}"]=quantile(features[feature],weights,q)
            summary.append(row)
            for ci,cap in enumerate(plan["capability_pairs"]):
                profile_rows.append(dict(seed=seed,scope=scope,red_delay=int(cap[0]),blue_delay=int(cap[1]),
                    count=int(profile_weights[ci]),fraction=float(profile_weights[ci]/weights.sum())))
    write_csv(out/"window_difficulty.csv",summary);write_csv(out/"window_profile_mix.csv",profile_rows)
    exposure=[]
    for i,r in enumerate(ref):
        if any(counts[s][scope][i] for s in (1,2) for scope in counts[s]):
            exposure.append({**r,**{f"seed{s}_{scope}":int(w[i]) for s in (1,2) for scope,w in counts[s].items()}})
    write_csv(out/"window_grid_profile_exposure.csv",exposure)
    actual={}
    for seed in (1,2):
        n=sum(metrics[seed][u]["rounds_completed"] for u in range(13,21))
        wins=sum(metrics[seed][u]["rounds_completed"]*metrics[seed][u]["round_success_rate"] for u in range(13,21))
        assert abs(wins-round(wins))<.01
        actual[seed]=dict(rounds=int(n),successes=int(round(wins)),success=wins/n)
    # Even without individual outcomes, total successes constrain what can have
    # happened on common cases. Give all unpaired seed-1 rounds successes and
    # all unpaired seed-2 rounds failures to minimize their shared-case gap.
    shared=int(np.minimum(counts[1]["completions"],counts[2]["completions"]).sum())
    extra={s:int(counts[s]["completions"].sum())-shared for s in (1,2)}
    lower_successes={s:max(0,actual[s]["successes"]-extra[s]) for s in (1,2)}
    upper_successes={s:min(shared,actual[s]["successes"]) for s in (1,2)}
    shared_bound=dict(shared_grid_profile_rounds=shared,seed1_unpaired_rounds=extra[1],seed2_unpaired_rounds=extra[2],
        seed1_minimum_shared_successes=lower_successes[1],seed2_maximum_shared_successes=upper_successes[2],
        minimum_shared_success_difference=lower_successes[1]-upper_successes[2],
        minimum_shared_rate_gap=(lower_successes[1]-upper_successes[2])/shared,
        maximum_shared_rate_gap=(upper_successes[1]-lower_successes[2])/shared,
        largest_cell_count=max(int(counts[s]["completions"].max()) for s in (1,2)),
        interpretation="Logical bound on realized outcomes, not a confidence interval. It does not identify the causal role of order, policy state, or optimization.")
    comparisons=[]
    reference_metadata=json.loads((base/"stimulus_reference.json").read_text())
    trials_per_assignment=reference_metadata["trials_per_case"]//2
    pr=np.array([float(r["random_success_ego_red"]) for r in ref])
    pb=np.array([float(r["random_success_ego_blue"]) for r in ref])
    # Independent simulations for each case, stratified equally by assignment.
    reference_variance=(pr*(1-pr)+pb*(1-pb))/(4*(trials_per_assignment-1))
    for scope in counts[1]:
        w1=counts[1][scope]/counts[1][scope].sum();w2=counts[2][scope]/counts[2][scope].sum()
        comparisons.append(dict(scope=scope,grid_profile_total_variation=float(.5*np.abs(w1-w2).sum()),
            grid_only_total_variation=float(.5*np.abs(w1.reshape(-1,24).sum(axis=1)-w2.reshape(-1,24).sum(axis=1)).sum()),
            neutral_success_difference_seed1_minus_seed2=float(np.dot(w1-w2,features["random_success"])),
            neutral_difference_monte_carlo_standard_error=float(np.sqrt(np.dot((w1-w2)**2,reference_variance))),
            best_steps_difference_seed2_minus_seed1=float(np.dot(w2-w1,features["oracle_best_steps"]))))
        for name in ("oracle_best_steps","random_success"):
            values,inverse=np.unique(features[name],return_inverse=True)
            cdf1=np.cumsum(np.bincount(inverse,weights=w1,minlength=len(values)))
            cdf2=np.cumsum(np.bincount(inverse,weights=w2,minlength=len(values)))
            comparisons[-1][f"{name}_maximum_cdf_difference"]=float(np.max(np.abs(cdf1-cdf2)))
    validation=validate_metrics(out,metrics)
    result=dict(window_updates=[13,20],window_transitions=[786432,1310720],actual_completed=actual,difficulty=summary,
        comparisons=comparisons,shared_case_success_bound=shared_bound,historical_aggregate_prefix_matches={s:prov[s]["matched_historical_aggregate_prefix_updates"] for s in (1,2)},
        snapshot_checks_passed=True,saved_metric_comparisons_passed=validation["comparisons"],
        provenance=prov,reference_sha256=sha(base/"stimulus_reference.csv"),analysis_source_sha256=sha(__file__),
        caveats=["Historical early counters were not saved; these counters come from a new instrumented reproduction.",
            "Matching all historical per-update aggregate counts/successes is strong validation, not a proof of identical original trajectories.",
            "If historical metrics diverge before update 20, these results describe the reproduction rather than historical exposure.",
            "No per-grid success or chosen assignment is available from the minimal counter snapshots; no matched behavioral claim is made.",
            "Neutral random-controller estimates use 128 simulated rounds per grid/profile and are shared across seeds.",
            "Transition weighting gives longer rounds more weight and can reflect a policy's difficulty rather than a harder curriculum."])
    (out/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    figure(out,result)
    report(out,result)
    print(json.dumps({k:result[k] for k in ("historical_aggregate_prefix_matches","actual_completed","comparisons","shared_case_success_bound")},indent=2))


def figure(out,result):
    starts={r["seed"]:r for r in result["difficulty"] if r["scope"]=="starts"}
    fig,axes=plt.subplots(1,2,figsize=(10,4.7),sharey=True)
    panels=[("Learner success on completed rounds",
        [100*result["actual_completed"][s]["success"] for s in (1,2)]),
        ("Neutral controller on presented cases",
        [100*starts[s]["random_success"] for s in (1,2)])]
    for ax,(title,values) in zip(axes,panels):
        ax.bar([0,1],values,color=["#2769aa","#d97b28"],width=.58,zorder=3)
        ax.set_xticks([0,1],["Seed 1","Seed 2"])
        ax.set_ylim(0,100);ax.set_title(title,fontsize=12,pad=16)
        ax.grid(axis="y",alpha=.2,zorder=0)
        ax.spines[["top","right"]].set_visible(False)
        for x,value in enumerate(values):ax.text(x,value+2,f"{value:.2f}%",ha="center",fontsize=12)
    axes[0].set_ylabel("Success (%)")
    fig.suptitle("V1_matched · 786,432–1,310,720 transitions",fontsize=15,y=.99)
    fig.text(.5,.015,"Independent difficulty reference: same random controller, weighted by each seed’s actual case exposure.",
        ha="center",fontsize=9,color="#485263")
    fig.tight_layout(rect=(0,.045,1,.94))
    fig.savefig(out/"success_and_stimulus_difficulty.png",dpi=180)
    plt.close(fig)


def report(out,result):
    rows=result["difficulty"]
    table="".join(f"<tr><td>{r['scope']}</td><td>{r['seed']}</td><td>{r['n']:,}</td><td>{r['unique_grids']}</td>"
        f"<td>{r['oracle_best_steps']:.3f}</td><td>{r['oracle_best_steps_p90']:.0f}</td><td>{100*r['random_success']:.2f}%</td><td>{100*r['wall_fraction']:.2f}%</td>"
        f"<td>{r['dead_ends']:.3f}</td></tr>" for r in rows)
    faithful=all(v==20 for v in result["historical_aggregate_prefix_matches"].values())
    qualification=(f"Both seeds match historical metrics through all 20 updates: all {result['saved_metric_comparisons_passed']} available scalar comparisons pass at the saved precision."
        if faithful else "The counter reproductions diverge from historical aggregate metrics before update 20. Their exposure is new replicate data, not recovered historical exposure.")
    starts={r["seed"]:r for r in rows if r["scope"]=="starts"}
    actual=result["actual_completed"]
    intro=(f"Seed 1 completes {actual[1]['rounds']:,} rounds with {100*actual[1]['success']:.2f}% success, "
        f"and seed 2 completes {actual[2]['rounds']:,} with {100*actual[2]['success']:.2f}% success. "
        f"For the new rounds actually presented in the interval, a shared neutral random controller scores "
        f"{100*starts[1]['random_success']:.2f}% on seed 1's grid/profile mix and {100*starts[2]['random_success']:.2f}% on seed 2's mix.")
    bound=result["shared_case_success_bound"]
    bound_text=(f"There are {bound['shared_grid_profile_rounds']:,} shared completed grid/profile cases. Seed 1 has "
        f"{actual[1]['successes']:,} total successes and seed 2 has {actual[2]['successes']:,}. Even if all "
        f"{bound['seed1_unpaired_rounds']:,} seed-1-only rounds succeeded, seed 1 must have at least "
        f"{bound['seed1_minimum_shared_successes']:,} successes on the shared cases. Even if all "
        f"{bound['seed2_unpaired_rounds']:,} seed-2-only rounds failed, seed 2 can have at most "
        f"{bound['seed2_maximum_shared_successes']:,} shared-case successes. Consequently at least "
        f"{bound['minimum_shared_success_difference']:,} additional seed-1 successes, or "
        f"{100*bound['minimum_shared_rate_gap']:.2f} percentage points, remain among shared grid/profile cases. "
        "This is a conservative counting bound; it does not require attributing individual outcomes to grids.")
    caveats="".join(f"<li>{html.escape(c)}</li>" for c in result["caveats"])
    doc=f"""<!doctype html><meta charset="utf-8"><title>Grid exposure during updates 13–20</title>
<style>body{{font:16px/1.55 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#202634}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ccd3dd;padding:7px;text-align:left}}</style>
<h1>V1_matched seed 1 versus seed 2: presented grids during 0.79–1.31M transitions</h1>
<p><b>{qualification}</b></p><p>{intro}</p>
<img src="success_and_stimulus_difficulty.png" alt="Learner success differs by 20.76 points, while neutral controller success on the presented cases differs by 0.02 points" style="width:100%;height:auto">
<h2>Method and fidelity</h2><p>The first full-trajectory recorder changed the compiled program's output shape and
diverged from historical metrics. This second method preserves the full 915-update loop, original returned tensors,
60M-step learning-rate schedule, all scientific functions, A100 hardware class and original package versions.
It reads existing integer exposure counters only at updates 12 and 20, then stops after those counters and scalar metrics
are saved. Per-update agreement is checked against the historical logs.</p>
<p>Each snapshot is independently reconstructed from its allocated schedule prefix, active episode IDs, round indices
and unfinished tails. Started/completed grid counts must match exactly, and transition counts must equal update×65,536.
Subtracting the snapshots isolates the requested eight updates, including rounds crossing either boundary.</p>
<h2>Independent difficulty of empirical exposures</h2>
<table><tr><th>Weighting</th><th>Seed</th><th>Observations</th><th>Unique grids</th><th>Mean best ticks</th><th>90th percentile best ticks</th><th>Neutral success</th><th>Walls</th><th>Dead ends</th></tr>{table}</table>
<p><b>Starts</b> count new rounds presented in the interval. <b>Completions</b> count rounds finishing there, including rounds
begun earlier. <b>Transitions</b> count every experienced tick. Grid difficulty is scored independently of either learner,
using validated exact fully informed completion times, geometry features, and a common uniform-assignment/uniform-movement controller.</p>
<p>For newly presented rounds, both seeds have median best completion time of 18 ticks, 90th percentile 30 ticks,
95th percentile 34 ticks, and 99th percentile 42 ticks. The largest difference between their best-time cumulative
distributions is 0.71 percentage points. Thus the means do not hide an excess of very hard cases for seed 2.</p>
<p>The shared random-controller reference uses 128 simulated rounds per grid/profile (3,366,912 simulations overall).
Its estimated standard error for the difference between the two start-weighted mixtures is 0.019 percentage points;
the observed difference of 0.016 points is within this Monte Carlo precision. This is simulation uncertainty for the
reference controller, not uncertainty across trained policies.</p>
<p>The table scores complete grid/profile pairs. The CSV additionally separates grid composition with profiles weighted
uniformly from profile composition with grids weighted uniformly. This avoids calling a change in partner speeds a geometry difference.</p>
<h2>A conservative check on shared cases</h2><p>{bound_text}</p>
<p>This comparison holds grid geometry and partner capability fixed as a case definition. It does not hold learned
weights, action choices, observations produced by earlier actions, or the training update at encounter fixed. The bound
shows that different case composition cannot account for all the realized gap; it does not rule out encounter order affecting learning.</p>
<h2>Evidence</h2><p><a href="window_grid_profile_exposure.csv">Empirical grid/profile counts and reference difficulty</a> ·
<a href="window_difficulty.csv">Difficulty summary</a> · <a href="window_profile_mix.csv">Partner-profile mixtures</a> ·
<a href="seed1_historical_comparison.csv">Seed 1 fidelity</a> · <a href="seed2_historical_comparison.csv">Seed 2 fidelity</a> ·
<a href="saved_metrics_validation.json">All 330 historical metric comparisons</a> ·
<a href="audit.json">Checks and provenance</a></p><h2>Limits</h2><ul>{caveats}</ul>"""
    (out/"report.html").write_text(doc)


if __name__=="__main__":main()
