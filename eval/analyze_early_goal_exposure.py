"""Supplement dense decoding with goal exposure excluding initialization STAYs."""
import argparse
from pathlib import Path
import shutil

import h5py
import numpy as np
import pandas as pd

import full_capability_evaluation as full
from analyze_early_decoding import CATEGORIES,STEPS
from plot_early_decoding import summarize


def plot_paired_gain(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=Path(out)
    data=pd.read_csv(out/"paired_gain_without_round_feedback_summary.csv")
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),sharey=True)
    colors=("#278148","#bd7e00","#b54a9e")
    for ax,version in zip(axes,full.VERSIONS):
        frame=data[(data.version==version)&(data.condition==full.CONDITIONS[0])].set_index("category").loc[list(CATEGORIES)]
        y=frame.paired_gain_mean.to_numpy()*100
        for i,color in enumerate(colors):
            ax.errorbar(i,y[i],yerr=[[y[i]-frame.paired_gain_low.iloc[i]*100],[frame.paired_gain_high.iloc[i]*100-y[i]]],
                        fmt="o",color=color,capsize=4,ms=7)
        ax.set(title=version,xticks=range(3),xticklabels=("Green\nseen profiles","Yellow\none novel delay","Pink\nboth novel"),
               ylim=(0,35),ylabel="Decoding gain (percentage points)")
        ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.15)
    fig.suptitle("Decoding improves without round completion or pursuing both goals\n"
                 "Matched held-out episodes across each category's steepest five-step window · multi-partner RNN",fontsize=12)
    fig.text(.5,.015,"Profiles averaged within each of five policy seeds; descriptive 95% bootstrap intervals. Cohorts selected at window end.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.86))
    for extension in ("png","pdf"):fig.savefig(out/f"paired_gain_without_round_feedback.{extension}",dpi=180)
    plt.close(fig)


def run(path):
    path=Path(path).resolve();analysis=full.read(path);out=Path(analysis["root"])
    manifest=full.read(analysis["input_manifest"])
    lookup={(p["red_delay"],p["blue_delay"]):p["category"] for p in manifest["profiles"]}
    rows=[];paired=[]
    onset_table=pd.read_csv(out/"onset_summary.csv")
    for record in analysis["policies"]:
        values=[];reps=[]
        paths=list(record["original_rollouts"].values())+[str(Path(record["directory"])/"additional_profiles.h5")]
        for source in paths:
            with h5py.File(source,"r") as handle:
                pairs=handle["capability"][:,0,:]
                goals=handle["partner_goal"][:,:100]
                init=handle["is_t0"][:,:100]
                dones=handle["dones"][:]
                lengths=dones.argmax(axis=1)+1
                valid=np.arange(100)[None,:]<lengths[:,None]
                active=valid&~init
                red=np.c_[np.zeros(len(pairs),bool),(np.cumsum((goals==1)&active,axis=1)>0)]
                blue=np.c_[np.zeros(len(pairs),bool),(np.cumsum((goals==2)&active,axis=1)>0)]
                both=red&blue
                finished=np.c_[np.zeros(len(pairs),bool),(np.cumsum(handle["round_done"][:,:100]&valid,axis=1)>0)]
                count={}
                for i,pair in enumerate(map(tuple,pairs)):
                    rep=count.get(pair,0);count[pair]=rep+1
                    values.append((pair,rep,both[i],finished[i],(red|blue)[i]))
        values.sort(key=lambda x:(x[0][0],x[0][1],x[1]))
        pairs=np.array([v[0] for v in values]);reps=np.array([v[1] for v in values]);both=np.array([v[2] for v in values])
        finished=np.array([v[3] for v in values]);one_or_more=np.array([v[4] for v in values])
        with np.load(Path(record["directory"])/"probe_models.npz") as prior:
            np.testing.assert_array_equal(pairs,prior["pairs"]);np.testing.assert_array_equal(reps,prior["repetitions"])
            test=prior["test"]
        with np.load(Path(record["early_directory"])/"early_probes.npz") as probes:
            predictions=probes["predictions"][:101]
            truth=probes["truth"]
        labels=np.array([lookup[tuple(p)] for p in pairs])
        for category in CATEGORIES:
            selected=test&(labels==category)
            for t in STEPS:
                rows.append(dict(**full.identity(record),category=category,env_step=int(t),
                    both_active_goals=float(both[selected,t].mean()),n_episodes=int(selected.sum())))
            onset=onset_table[(onset_table.version==record["version"])&(onset_table.condition==record["condition"])
                              &(onset_table.category==category)].iloc[0]
            start,end=int(onset.peak5_start),int(onset.peak5_end)
            chosen=np.flatnonzero(selected&~finished[:,end]&~both[:,end]&one_or_more[:,end])
            if len(chosen):
                matched=pd.DataFrame(dict(d_R=pairs[chosen,0],d_B=pairs[chosen,1],
                    before=(predictions[start,chosen]==truth[chosen]).astype(float),
                    after=(predictions[end,chosen]==truth[chosen]).astype(float)))
                grouped=matched.groupby(["d_R","d_B"])[["before","after"]].mean()
                paired.append(dict(**full.identity(record),category=category,window_start=start,window_end=end,
                    n_episodes=len(chosen),n_profiles=len(grouped),accuracy_before=float(grouped.before.mean()),
                    accuracy_after=float(grouped.after.mean()),paired_gain=float((grouped.after-grouped.before).mean())))
    table=pd.DataFrame(rows);table.to_csv(out/"active_goal_exposure_per_seed.csv",index=False)
    keys=["version","condition","category","env_step"]
    exposure=summarize(table,keys,["both_active_goals"])
    exposure.to_csv(out/"active_goal_exposure_summary.csv",index=False)
    paired=pd.DataFrame(paired);paired.to_csv(out/"paired_gain_without_round_feedback_per_seed.csv",index=False)
    summarize(paired,["version","condition","category"],["accuracy_before","accuracy_after","paired_gain"],min_seeds=3).to_csv(
        out/"paired_gain_without_round_feedback_summary.csv",index=False)
    primary=pd.read_csv(out/"early_summary.csv").merge(exposure,on=keys,validate="one_to_one",suffixes=("","_exposure"))
    onsets=pd.read_csv(out/"onset_summary.csv")
    windows=[]
    for _,onset in onsets.iterrows():
        g=primary[(primary.version==onset.version)&(primary.condition==onset.condition)&(primary.category==onset.category)].set_index("env_step")
        for t in (int(onset.peak5_start),int(onset.peak5_end)):
            windows.append(dict(version=onset.version,condition=onset.condition,category=onset.category,env_step=t,
                **{k:float(g.loc[t,k+"_mean"]) for k in ("accuracy","action_faster","round_finished","any_switch","both_active_goals")}))
    pd.DataFrame(windows).to_csv(out/"behavior_at_sharp_windows_active_goals.csv",index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(17,11),sharex=True,sharey=True)
    names=("accuracy","action_faster","round_finished","any_switch","both_active_goals")
    legend=("Decoding accuracy","Current action assigns faster goal","First round already finished","Any within-round goal change","Both goals pursued after initialization")
    colors=("#9380ff","#e67e00","#333333","#168e88","#3779bf")
    for row,category in enumerate(CATEGORIES):
        for col,version in enumerate(full.VERSIONS):
            ax=axes[row,col]
            g=primary[(primary.version==version)&(primary.condition==full.CONDITIONS[0])&(primary.category==category)].sort_values("env_step")
            for name,label,color in zip(names,legend,colors):ax.plot(g.env_step,g[name+"_mean"],c=color,lw=1.8,label=label)
            point=onsets[(onsets.version==version)&(onsets.condition==full.CONDITIONS[0])&(onsets.category==category)].iloc[0]
            ax.axvspan(point.peak5_start,point.peak5_end,color="gray",alpha=.16)
            ax.set_facecolor(full.CATEGORY_COLORS[category]);ax.spines[["top","right"]].set_visible(False)
            ax.set(title=f"{version} · {full.CATEGORY_LABELS[category]}",xlim=(0,60),ylim=(0,1.02),
                   xlabel="Completed environment transitions",ylabel="Accuracy / episode fraction")
            ax.title.set_fontsize(10)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.suptitle("Multi-partner RNN · dense decoding and behavioral events\n"
        "Prior-event curves exclude the current transition. Goal exposure excludes t=0 STAYs, but actual partner movements were not saved.",fontsize=12,y=.995)
    fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.94),h_pad=2)
    for extension in ("png","pdf"):fig.savefig(out/f"early_behavior_active_goal_correlates.{extension}",dpi=180)
    plt.close(fig)
    plot_paired_gain(out)
    shutil.copyfile(__file__,out/"source"/Path(__file__).name)
    full.write(out/"active_goal_exposure_methods.json",dict(status="passed",policies=45,timepoints=101,
        rule="At observation t, each counted goal must have occurred during a prior transition index < t with is_t0=False and before the episode terminal. Initialization STAY labels do not count.",
        limitation="Pursuing a goal for a movement-phase step does not establish an actual move or an informative delay observation: positions and movement clocks were not saved.",
        paired_cohort="Same held-out episodes at both endpoints of the category's steepest window, restricted to no completed rounds and exactly one movement-phase goal pursued by the window end. Profiles weighted equally within each seed.",
        source_sha256=full.sha(__file__)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("path",type=Path)
    run(parser.parse_args().path)
