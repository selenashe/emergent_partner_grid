"""Align familiar-capability single-state probes with early evaluation behavior."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import h5py
import numpy as np
import pandas as pd

import full_capability_evaluation as full
from analyze_instantaneous_decoding import bootstrap_seeds
from plot_early_decoding import timing,first_run,summarize

STEPS=np.arange(101)
CATEGORIES=("training_profile","one_delay_novel","both_delays_novel")
METRICS=("accuracy","correct_probability","fixed20_accuracy","current_assignment_faster",
         "prior_assignment_faster","round_finished","success_seen","both_active_goals",
         "any_active_goal","any_goal_switch","any_assignment_switch","rounds_finished")


def history(event):
    """At observation t include transitions 0..t-1, excluding transition t."""
    return np.c_[np.zeros(len(event)),np.cumsum(event[:,:-1],axis=1)]


def first_event(history_values):
    return np.where(history_values.any(axis=1),history_values.argmax(axis=1),-1)


def behavioral_history(data):
    valid=STEPS[None,:]<data["lengths"][:,None]
    goal=data["partner_goal"];active=valid&~data["is_t0"]&np.isin(goal,[1,2])
    red=history(active&(goal==1))>0;blue=history(active&(goal==2))>0
    both=red&blue;one=red|blue
    same_round=np.c_[np.zeros(len(goal),bool),data["round_idx"][:,1:]==data["round_idx"][:,:-1]]
    previous=np.c_[np.zeros(len(goal),int),goal[:,:-1]]
    previous_active=np.c_[np.zeros(len(goal),bool),active[:,:-1]]
    switch=active&previous_active&same_round&(previous!=goal)
    assignment=data["partner_assignment"]
    previous_assignment=np.c_[np.zeros(len(goal),int),assignment[:,:-1]]
    assignment_switch=valid&same_round&np.isin(assignment,[1,2])&np.isin(previous_assignment,[1,2])&(assignment!=previous_assignment)
    faster_assignment=np.where(data["pairs"][:,0]<data["pairs"][:,1],2,1)
    assignment_faster=assignment==faster_assignment[:,None]
    prior_assignment=np.c_[np.full(len(goal),.5),assignment_faster[:,:-1]]
    rounds=history(data["round_done"]&valid)
    return dict(current_assignment_faster=assignment_faster.astype(float),prior_assignment_faster=prior_assignment,
        round_finished=rounds>0,rounds_finished=rounds,success_seen=history(data["success"]&valid)>0,
        both_active_goals=both,any_active_goal=one,any_goal_switch=history(switch)>0,
        any_assignment_switch=history(assignment_switch)>0,
        first_round_feedback=first_event(rounds>0),first_both_active_goals=first_event(both),
        first_goal_switch=first_event(history(switch)>0))


def load_policy(record):
    archive=Path(record["probe_directory"])/"training_capability_probes.npz"
    done=full.read(Path(record["probe_directory"])/"completion.json")
    full.require(full.sha(archive)==done["archive_sha256"],"Changed fitted-probe archive")
    with np.load(archive) as probe:
        output={k:probe[k].copy() for k in ("pairs","repetitions","test","truth","lengths")}
        output["predictions"]=probe["predictions"][:101].copy()
        output["probabilities"]=probe["probabilities"][:101].copy()
        fixed_weight=probe["weight"][20].copy();fixed_bias=probe["bias"][20].copy()
    fields=("partner_assignment","ego_alloc_action","partner_goal","round_idx","round_time","round_done","success","is_t0")
    batches=[]
    for path in list(record["original_rollouts"].values())+[str(Path(record["directory"])/"additional_profiles.h5")]:
        with h5py.File(path,"r") as h:
            pairs=h["capability"][:,0,:];counts={};reps=[]
            for pair in map(tuple,pairs):reps.append(counts.get(pair,0));counts[pair]=reps[-1]+1
            item={name:h[name][:,:101] for name in fields}
            hidden=h["hidden_state"][:,:101]
            logits=hidden@fixed_weight+fixed_bias
            logits=logits-logits.max(axis=-1,keepdims=True);prob=np.exp(logits);prob/=prob.sum(axis=-1,keepdims=True)
            item.update(pairs=pairs,repetitions=np.array(reps),fixed20_probabilities=prob)
            batches.append(item)
    loaded={key:np.concatenate([b[key] for b in batches]) for key in batches[0]}
    order=np.lexsort((loaded["repetitions"],loaded["pairs"][:,1],loaded["pairs"][:,0]))
    loaded={k:v[order] for k,v in loaded.items()}
    for key in ("pairs","repetitions"):np.testing.assert_array_equal(loaded[key],output[key])
    output.update(loaded)
    full.require(output["lengths"].min()>100,"Early window includes episode termination")
    np.testing.assert_allclose(output["fixed20_probabilities"][:,20],output["probabilities"][20],atol=2e-5,rtol=1e-4)
    return output


def onset_statistics(root):
    with np.load(root/"category_averages/category_probe_per_seed.npz") as data:
        accuracy=data["accuracy"].copy();cats=data["categories"].tolist();conditions=data["conditions"].tolist()
    summary=pd.read_csv(root/"category_averages/category_probe_summary.csv")
    draws=np.random.default_rng(30000).integers(0,5,(10000,5));rows=[]
    for v,version in enumerate(full.VERSIONS):
        for c,condition in enumerate(conditions):
            for category in CATEGORIES:
                values=accuracy[v,c,:,:101,cats.index(category)];mean=values.mean(axis=0)
                point=timing(mean);boot=timing(values[draws].mean(axis=1))
                row=dict(version=version,condition=condition,category=category,
                    **{key:float(value) for key,value in point.items()})
                row["peak5_end"]=int(row["peak5_start"]+5)
                low=summary[(summary.version==version)&(summary.condition==condition)&(summary.category==category)].sort_values("env_step").accuracy_ci_low.to_numpy()[:61]
                row["first_five_steps_ci_above_chance"]=float(first_run(low>.5,5))
                for name in ("peak_step","peak5_start","half_rise_step"):
                    values_boot=boot[name][np.isfinite(boot[name])]
                    row[name+"_low"]=float(np.quantile(values_boot,.025)) if len(values_boot) else np.nan
                    row[name+"_high"]=float(np.quantile(values_boot,.975)) if len(values_boot) else np.nan
                row["accuracy_start"]=float(mean[int(row["peak5_start"])])
                row["accuracy_end"]=float(mean[int(row["peak5_end"])])
                rows.append(row)
    return pd.DataFrame(rows)


def equal_profile_mean(values,pairs):
    unique,codes=np.unique(pairs,axis=0,return_inverse=True)
    return float(np.mean([values[codes==k].mean() for k in range(len(unique))]))


def policy_tables(record,data,manifest,onsets):
    lookup={(p["red_delay"],p["blue_delay"]):p["category"] for p in manifest["profiles"]}
    category=np.array([lookup[tuple(pair)] for pair in data["pairs"]])
    truth=data["truth"];correct=data["predictions"]==truth[None,:]
    confidence=data["probabilities"][:,np.arange(len(truth)),truth]
    fixed_correct=data["fixed20_probabilities"].argmax(axis=-1)==truth[:,None]
    fixed_confidence=data["fixed20_probabilities"][np.arange(len(truth)),:,truth]
    behavior=behavioral_history(data);rows=[];paired=[];events=[];episode=[];strata=[]
    for cat in CATEGORIES:
        selected=data["test"]&(category==cat)
        for t in STEPS:
            row=dict(**full.identity(record),category=cat,env_step=int(t),accuracy=float(correct[t,selected].mean()),
                correct_probability=float(confidence[t,selected].mean()),fixed20_accuracy=float(fixed_correct[selected,t].mean()),n_episodes=int(selected.sum()))
            row.update({name:float(behavior[name][selected,t].mean()) for name in METRICS if name not in row and name not in ("accuracy","correct_probability","fixed20_accuracy")})
            rows.append(row)
        onset=onsets[(onsets.version==record["version"])&(onsets.condition==record["condition"])&(onsets.category==cat)].iloc[0]
        start,end=int(onset.peak5_start),int(onset.peak5_end)
        cohorts=dict(all=np.ones(len(truth),bool),no_round_feedback=~behavior["round_finished"][:,end],
            one_goal_no_round=(~behavior["round_finished"][:,end])&~behavior["both_active_goals"][:,end]&behavior["any_active_goal"][:,end])
        for name,cohort in cohorts.items():
            chosen=np.flatnonzero(selected&cohort)
            if len(chosen)==0:continue
            before=equal_profile_mean(correct[start,chosen],data["pairs"][chosen])
            after=equal_profile_mean(correct[end,chosen],data["pairs"][chosen])
            paired.append(dict(**full.identity(record),category=cat,cohort=name,window_start=start,window_end=end,
                n_episodes=len(chosen),n_profiles=len(np.unique(data["pairs"][chosen],axis=0)),
                accuracy_before=before,accuracy_after=after,paired_gain=after-before,
                fixed20_gain=equal_profile_mean((fixed_correct[chosen,end].astype(float)-fixed_correct[chosen,start]),data["pairs"][chosen])))
        for event in ("first_round_feedback","first_goal_switch","first_both_active_goals"):
            times=behavior[event]
            chosen=np.flatnonzero(selected&(times>=5)&(times<=95))
            if len(chosen)==0:continue
            for lag in range(-5,6):
                t=times[chosen]+lag
                events.append(dict(**full.identity(record),category=cat,event=event,lag=lag,n_episodes=len(chosen),
                    accuracy=equal_profile_mean(correct[t,chosen],data["pairs"][chosen]),
                    correct_probability=equal_profile_mean(confidence[t,chosen],data["pairs"][chosen]),
                    fixed20_probability=equal_profile_mean(fixed_confidence[chosen,t],data["pairs"][chosen])))
        for e in np.flatnonzero(selected):
            pcorrect=confidence[:,e]
            stable=first_run(np.lib.stride_tricks.sliding_window_view(pcorrect>.65,3).all(axis=-1),1)
            episode.append(dict(**full.identity(record),category=cat,d_R=int(data["pairs"][e,0]),d_B=int(data["pairs"][e,1]),repetition=int(data["repetitions"][e]),
                first_round_feedback=int(behavior["first_round_feedback"][e]),first_goal_switch=int(behavior["first_goal_switch"][e]),
                first_both_active_goals=int(behavior["first_both_active_goals"][e]),first_three_confident_steps=float(stable),
                correct_at_window_start=bool(correct[start,e]),correct_at_window_end=bool(correct[end,e])))
    # Split the fully novel category by zero-delay vs the difficult (5,6)/(6,5) pair.
    for label,mask in (("pink_zero_delay",(category=="both_delays_novel")&(data["pairs"]==0).any(axis=1)),
                       ("pink_5_vs_6",(category=="both_delays_novel")&~(data["pairs"]==0).any(axis=1))):
        chosen=data["test"]&mask
        for t in STEPS:strata.append(dict(**full.identity(record),stratum=label,env_step=int(t),accuracy=float(correct[t,chosen].mean()),n_episodes=int(chosen.sum())))
    return tuple(pd.DataFrame(x) for x in (rows,paired,events,episode,strata))


def summarize_curves(table):
    frames=[]
    for key,g in table.groupby(["version","condition","category"],sort=True):
        g=g.sort_values(["training_seed","env_step"])
        full.require(len(g)==505,"Five seeds and 101 steps required")
        values=g[list(METRICS)].to_numpy().reshape(5,101,len(METRICS))
        mean,low,high=bootstrap_seeds(values)
        rows=[]
        for t in STEPS:
            row=dict(zip(("version","condition","category"),key),env_step=int(t))
            for k,metric in enumerate(METRICS):row.update({metric+"_mean":mean[t,k],metric+"_low":low[t,k],metric+"_high":high[t,k]})
            rows.append(row)
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames,ignore_index=True)


def figures(summary,onsets,paired,strata,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(16,11),sharex=True,sharey=True)
    specifications=(("accuracy",'Frozen value estimate decoding',"#9380ff"),
        ("fixed20_accuracy","Same fixed step-20 decoder","#5d3b9c"),
        ("current_assignment_faster","Action following h_t assigns faster goal","#e67e00"),
        ("round_finished","First round already finished","#333333"),
        ("both_active_goals","Both goals already pursued","#3779bf"),
        ("any_goal_switch","Within-round goal change already observed","#168e88"))
    for r,cat in enumerate(CATEGORIES):
        for c,version in enumerate(full.VERSIONS):
            ax=axes[r,c];g=summary[(summary.version==version)&(summary.condition==full.CONDITIONS[0])&(summary.category==cat)].sort_values("env_step")
            for metric,label,color in specifications:
                ax.plot(g.env_step,g[metric+"_mean"],color=color,lw=1.6,label=label,ls="--" if metric=="fixed20_accuracy" else "-")
            p=onsets[(onsets.version==version)&(onsets.condition==full.CONDITIONS[0])&(onsets.category==cat)].iloc[0]
            ax.axvspan(p.peak5_start,p.peak5_end,color="gray",alpha=.18)
            ax.set_facecolor(full.CATEGORY_COLORS[cat]);ax.spines[["top","right"]].set_visible(False)
            ax.set(title=f"{version} · {full.CATEGORY_LABELS[cat]}\nSteepest five-step rise: {int(p.peak5_start)}–{int(p.peak5_end)}",xlim=(0,40),ylim=(0,1.02),xlabel="Environment step (0 = initial observation)",ylabel="Accuracy / episode fraction")
            ax.title.set_fontsize(10)
    fig.suptitle("Frozen value estimate · early decoding and evaluation behavior\n"
        "Prior-event curves include transitions strictly before h_t; action curve follows h_t. Gray windows are descriptive maxima.",fontsize=12)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.93))
    for ext in ("png","pdf"):fig.savefig(out/f"decoding_and_behavior.{ext}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True)
    colors=("#278148","#bd7e00","#b54a9e")
    for ax,version in zip(axes,full.VERSIONS):
        for j,(cat,color) in enumerate(zip(CATEGORIES,colors)):
            selected=paired[(paired.version==version)&(paired.condition==full.CONDITIONS[0])&(paired.category==cat)&(paired.cohort=="one_goal_no_round")]
            if selected.empty:
                ax.text(j,.02,'Insufficient cohort',rotation=90,ha='center',va='bottom',fontsize=8)
                continue
            g=selected.iloc[0]
            y=g.paired_gain_mean*100
            ax.errorbar(j,y,yerr=[[y-g.paired_gain_low*100],[g.paired_gain_high*100-y]],fmt="o",color=color,capsize=4,ms=7)
        ax.axhline(0,color="gray",lw=.8);ax.spines[["top","right"]].set_visible(False)
        ax.set(title=version,xticks=[0,1,2],xticklabels=["Green\nseen","Yellow\none novel","Pink\nboth novel"],ylabel="Accuracy gain (percentage points)")
    fig.suptitle("Matched episodes: decoding gain before any round feedback or pursuit of both goals\n"
        "Same held-out episodes at both endpoints of each category's steepest window; equal profile weights within seed")
    fig.tight_layout(rect=(0,0,1,.87))
    for ext in ("png","pdf"):fig.savefig(out/f"paired_gain_before_feedback.{ext}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.7),sharex=True,sharey=True)
    for ax,version in zip(axes,full.VERSIONS):
        for name,label,color in (("pink_zero_delay","Pink profiles involving 0","#b54a9e"),("pink_5_vs_6","Pink (5,6) and (6,5)","#765111")):
            g=strata[(strata.version==version)&(strata.condition==full.CONDITIONS[0])&(strata.stratum==name)].sort_values("env_step")
            ax.plot(g.env_step,g.accuracy_mean,label=label,color=color)
            ax.fill_between(g.env_step,g.accuracy_low,g.accuracy_high,color=color,alpha=.15)
        ax.axhline(.5,color="gray",ls="--");ax.spines[["top","right"]].set_visible(False)
        ax.set(title=version,xlim=(0,40),ylim=(0,1.02),xlabel="Environment step",ylabel="Decoding accuracy")
    fig.suptitle("Fully novel profiles: zero-vs-slow versus distinguishing two slow delays")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.07,1,.91))
    for ext in ("png","pdf"):fig.savefig(out/f"pink_profile_strata.{ext}",dpi=180)
    plt.close(fig)


def run(path,out):
    path=Path(path).resolve();root=path.parent;run=full.read(path);out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    full.require(run["status"]=="completed","Completed training-capability probes required")
    manifest=full.read(run["input_manifest"]);full.verify(manifest)
    (out/"source").mkdir(exist_ok=True)
    for name in ("analyze_decoding_behavior.py","replay_early_partner_motion.py","plot_early_decoding.py","analyze_early_decoding.py","analyze_instantaneous_decoding.py","full_capability_evaluation.py"):
        shutil.copyfile(Path(__file__).with_name(name),out/"source"/name)
    onsets=onset_statistics(root);onsets.to_csv(out/"onset_summary.csv",index=False)
    collections=[[] for _ in range(5)]
    for record in run["policies"]:
        print("[analyze saved episodes]",full.identity(record),flush=True)
        data=load_policy(record)
        tables=policy_tables(record,data,manifest,onsets)
        for bucket,table in zip(collections,tables):bucket.append(table)
    names=("curves","paired_cohorts","event_aligned","episode_events","pink_strata")
    tables={name:pd.concat(items,ignore_index=True) for name,items in zip(names,collections)}
    for name,table in tables.items():table.to_csv(out/(name+"_per_seed.csv"),index=False)
    summary=summarize_curves(tables["curves"]);summary.to_csv(out/"behavior_summary.csv",index=False)
    paired=summarize(tables["paired_cohorts"],["version","condition","category","cohort"],["accuracy_before","accuracy_after","paired_gain","fixed20_gain","n_episodes","n_profiles"])
    paired.to_csv(out/"paired_cohort_summary.csv",index=False)
    strata=summarize(tables["pink_strata"],["version","condition","stratum","env_step"],["accuracy"])
    strata.to_csv(out/"pink_strata_summary.csv",index=False)
    summarize(tables["event_aligned"],["version","condition","category","event","lag"],["accuracy","correct_probability","fixed20_probability"],min_seeds=3).to_csv(out/"event_aligned_summary.csv",index=False)
    snapshots=[]
    for _,p in onsets.iterrows():
        for t in (int(p.peak5_start),int(p.peak5_end)):
            g=summary[(summary.version==p.version)&(summary.condition==p.condition)&(summary.category==p.category)&(summary.env_step==t)].iloc[0]
            snapshots.append(dict(version=p.version,condition=p.condition,category=p.category,env_step=t,**{metric:g[metric+"_mean"] for metric in METRICS}))
    pd.DataFrame(snapshots).to_csv(out/"behavior_at_rise_windows.csv",index=False)
    figures(summary,onsets,paired,strata,out)
    full.write(out/"saved_behavior_audit.json",dict(status="passed",completed_utc=full.now(),input_manifest=str(path),input_manifest_sha256=full.sha(path),
        policies=30,primary_multi_partner_policies=15,primary_test_episodes=15*(24+42+6)*4,timepoints=101,
        probe_refitting=False,policy_retraining=False,information_timing="h_t uses observations through t and prior event transitions <t. Current-action curve is explicitly subsequent to h_t.",
        selection="Same four held-out episodes/profile as new training-capability-only probes. Same episodes at each paired cohort's window endpoints; equal profile weights within each seed.",
        onset="Largest one-step and five-step accuracy increases among t=0..50. Timing uncertainty bootstraps five policy seeds. Windows are descriptive, selected from the same decoding curves.",
        fixed_decoder="Saved step-20 probe is applied unchanged to h_t at all early timesteps; validates against original step-20 probabilities.",
        limitations="Saved behavior alone omits physical positions. Verified replay is a separate audit; observational alignment cannot establish retrieval/inference causally.",
        source_sha256={p.name:full.sha(p) for p in (out/"source").glob("*.py")}))
    print("[saved behavior analysis completed]",out,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("path",type=Path);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();run(args.path,args.out)
