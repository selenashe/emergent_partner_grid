"""Aggregate dense early decoding and observational event diagnostics."""
from pathlib import Path

import numpy as np
import pandas as pd

import full_capability_evaluation as full
from analyze_early_decoding import CATEGORIES,STEPS,INSTANT_STEPS


def summarize(table,keys,metrics,min_seeds=5):
    indices=np.random.default_rng(30000).integers(0,5,size=(10000,5))
    rows=[]
    for group,frame in table.groupby(keys,sort=True):
        if len(frame)<min_seeds:continue
        full.require(not frame.training_seed.duplicated().any(),"Duplicate seed observations")
        frame=frame.sort_values("training_seed")
        row=dict(zip(keys,group if isinstance(group,tuple) else (group,)))
        row["n_seeds"]=len(frame)
        for metric in metrics:
            values=frame[metric].to_numpy()
            finite=values[np.isfinite(values)]
            row[metric+"_valid_seeds"]=len(finite)
            if len(finite)<3:
                row.update({metric+"_mean":np.nan,metric+"_low":np.nan,metric+"_high":np.nan})
                continue
            resampling=indices if len(finite)==5 else np.random.default_rng(30000).integers(0,len(finite),(10000,len(finite)))
            means=finite[resampling].mean(axis=1)
            row.update({metric+"_mean":float(finite.mean()),metric+"_low":float(np.quantile(means,.025)),
                        metric+"_high":float(np.quantile(means,.975))})
        rows.append(row)
    return pd.DataFrame(rows)


def first_run(mask,length):
    windows=np.lib.stride_tricks.sliding_window_view(mask,length,axis=-1).all(axis=-1)
    return np.where(windows.any(axis=-1),windows.argmax(axis=-1),np.nan)


def timing(curves):
    """Descriptive step/5-step maxima and half-rise, restricted to steps 0..50."""
    curves=np.asarray(curves)
    first=curves[...,:51]
    step=np.diff(first,axis=-1)
    window=first[...,5:]-first[...,:-5]
    gain=first[...,40:51].mean(axis=-1)-first[...,0]
    threshold=first[...,0]+.5*gain
    half=first_run(first>=threshold[...,None],3)
    half=np.where(gain>=.1,half,np.nan)
    return dict(peak_step=step.argmax(axis=-1)+1,peak_step_gain=step.max(axis=-1),
                peak5_start=window.argmax(axis=-1),peak5_gain=window.max(axis=-1),half_rise_step=half,gain_by50=gain)


def onset_table(curves,summary):
    indices=np.random.default_rng(30000).integers(0,5,size=(10000,5));rows=[];cache={}
    for key,frame in curves.groupby(["version","condition","category"],sort=True):
        matrix=frame.pivot(index="training_seed",columns="env_step",values="accuracy").sort_index()
        full.require(matrix.shape==(5,101),"Incomplete dense timestep matrix")
        values=matrix.to_numpy();mean=values.mean(axis=0)
        point=timing(mean);resampled=timing(values[indices].mean(axis=1))
        g=summary[(summary.version==key[0])&(summary.condition==key[1])&(summary.category==key[2])].sort_values("env_step")
        point["first_5step_ci_above_chance"]=first_run(g.accuracy_low.to_numpy()>.5,5)
        row=dict(zip(["version","condition","category"],key))
        for name,value in point.items():row[name]=float(value)
        for name in ("peak_step","peak5_start","half_rise_step"):
            valid=resampled[name][np.isfinite(resampled[name])]
            row[name+"_low"]=float(np.quantile(valid,.025)) if len(valid) else np.nan
            row[name+"_high"]=float(np.quantile(valid,.975)) if len(valid) else np.nan
        row["peak5_end"]=row["peak5_start"]+5
        row["half_rise_bootstrap_coverage"]=float(np.isfinite(resampled["half_rise_step"]).mean())
        rows.append(row);cache[key]=(point,resampled)
    differences=[]
    for version in full.VERSIONS:
        for condition in full.CONDITIONS:
            base,base_boot=cache[(version,condition,"training_profile")]
            for category in CATEGORIES[1:]:
                point,boot=cache[(version,condition,category)]
                delta=boot["half_rise_step"]-base_boot["half_rise_step"]
                finite=delta[np.isfinite(delta)]
                differences.append(dict(version=version,condition=condition,category=category,
                    half_rise_difference_from_green=point["half_rise_step"]-base["half_rise_step"],
                    difference_low=float(np.quantile(finite,.025)) if len(finite) else np.nan,
                    difference_high=float(np.quantile(finite,.975)) if len(finite) else np.nan,
                    valid_bootstrap_fraction=len(finite)/10000))
    return pd.DataFrame(rows),pd.DataFrame(differences)


def figures(summary,instant,events,onsets,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def finish(fig,axes,stem,title,ncol=3):
        handles,labels=np.ravel(axes)[0].get_legend_handles_labels()
        fig.suptitle(title,fontsize=12,y=.995)
        fig.legend(handles,labels,loc="lower center",ncol=ncol,frameon=False,fontsize=9)
        fig.tight_layout(rect=(0,.055,1,.94),h_pad=2)
        for ext in ("png","pdf"):fig.savefig(out/f"{stem}.{ext}",dpi=180)
        plt.close(fig)

    fig,axes=plt.subplots(3,3,figsize=(17,11),sharex=True,sharey=True)
    for row,category in enumerate(CATEGORIES):
        for col,version in enumerate(full.VERSIONS):
            ax=axes[row,col]
            for condition,color,label in zip(full.CONDITIONS,full.COLORS,full.LABELS):
                g=summary[(summary.version==version)&(summary.condition==condition)&(summary.category==category)].sort_values("env_step")
                ax.plot(g.env_step,g.accuracy_mean,c=color,lw=1.8,label=label)
                ax.fill_between(g.env_step.to_numpy(),g.accuracy_low.to_numpy(),g.accuracy_high.to_numpy(),color=color,alpha=.16)
            p=onsets[(onsets.version==version)&(onsets.condition==full.CONDITIONS[0])&(onsets.category==category)].iloc[0]
            ax.axvspan(p.peak5_start,p.peak5_end,color="gray",alpha=.16)
            ax.axhline(.5,color="gray",ls="--",lw=.8)
            ax.set_facecolor(full.CATEGORY_COLORS[category]);ax.spines[["top","right"]].set_visible(False)
            ax.set(title=f"{version} · {full.CATEGORY_LABELS[category]}\nMulti-partner RNN steepest window: {int(p.peak5_start)}–{int(p.peak5_end)} transitions",
                   xlim=(0,60),ylim=(0,1.02),xlabel="Completed environment transitions",ylabel="Held-out decoding accuracy")
            ax.set_xticks(np.arange(0,61,10));ax.tick_params(labelsize=9);ax.title.set_fontsize(9)
    finish(fig,axes,"early_decoding_dense","Dense early decoding · same reference probes and 16/4 episode split\n"
           "One decoder per timestep; five policy seeds; 95% bootstrap CI. Gray: steepest five-transition window.")

    fig,axes=plt.subplots(3,3,figsize=(17,11),sharex=True,sharey=True)
    names=("accuracy","action_faster","round_finished","any_switch","both_goals_sampled")
    labels=("Decoding accuracy","Current action assigns faster goal","First round already finished","At least one within-round goal switch","Both goal assignments sampled")
    colors=("#9380ff","#e67e00","#333333","#168e88","#3779bf")
    for row,category in enumerate(CATEGORIES):
        for col,version in enumerate(full.VERSIONS):
            ax=axes[row,col]
            g=summary[(summary.version==version)&(summary.condition==full.CONDITIONS[0])&(summary.category==category)].sort_values("env_step")
            for name,label,color in zip(names,labels,colors):ax.plot(g.env_step,g[name+"_mean"],c=color,label=label,lw=1.8)
            p=onsets[(onsets.version==version)&(onsets.condition==full.CONDITIONS[0])&(onsets.category==category)].iloc[0]
            ax.axvspan(p.peak5_start,p.peak5_end,color="gray",alpha=.16)
            ax.set_facecolor(full.CATEGORY_COLORS[category]);ax.spines[["top","right"]].set_visible(False)
            ax.set(title=f"{version} · {full.CATEGORY_LABELS[category]}",xlim=(0,60),ylim=(0,1.02),
                   xlabel="Completed environment transitions",ylabel="Accuracy / episode fraction")
            ax.title.set_fontsize(10);ax.tick_params(labelsize=9)
    finish(fig,axes,"early_behavior_correlates","Multi-partner RNN · early decoding and saved behavioral events\n"
           "Events other than the current action have occurred before the decoded observation. Assignment exposure is not direct movement evidence.",ncol=3)

    fig,axes=plt.subplots(1,3,figsize=(16,5),sharex=True,sharey=True)
    catcolors=("#278148","#bd7e00","#b54a9e")
    for ax,version in zip(axes,full.VERSIONS):
        for category,color in zip(CATEGORIES,catcolors):
            g=summary[(summary.version==version)&(summary.condition==full.CONDITIONS[0])&(summary.category==category)].sort_values("env_step")
            j=instant[(instant.version==version)&(instant.condition==full.CONDITIONS[0])&(instant.category==category)].sort_values("env_step")
            ax.plot(g.env_step,g.accuracy_mean,c=color,label=full.CATEGORY_LABELS[category],lw=1.8)
            ax.plot(j.env_step,j.accuracy_mean,c=color,ls="--",marker=".",lw=1.3)
        ax.axhline(.5,color="gray",ls=":");ax.set(title=version,xlim=(0,60),ylim=(0,1.02),xlabel="Completed environment transitions",ylabel="Decoding accuracy")
        ax.spines[["top","right"]].set_visible(False)
    finish(fig,axes,"early_instantaneous_vs_prefix","Multi-partner RNN · solid: cumulative mean state; dashed: instantaneous state\n"
           "Same affine probe, episode split, and initialization; instantaneous probes fitted at the marked timesteps.")

    fig,axes=plt.subplots(1,3,figsize=(16,5),sharex=True,sharey=True)
    for ax,version in zip(axes,full.VERSIONS):
        for category,color in zip(CATEGORIES,catcolors):
            g=summary[(summary.version==version)&(summary.condition==full.CONDITIONS[0])&(summary.category==category)].sort_values("env_step")
            ax.plot(g.env_step,g.accuracy_mean,c=color,label=full.CATEGORY_LABELS[category],lw=1.8)
            ax.plot(g.env_step,g.frozen50_accuracy_mean,c=color,ls="--",lw=1.3)
        ax.axhline(.5,color="gray",ls=":");ax.set(title=version,xlim=(0,60),ylim=(0,1.02),xlabel="Completed environment transitions",ylabel="Decoding accuracy")
        ax.spines[["top","right"]].set_visible(False)
    finish(fig,axes,"early_fixed_decoder_transfer","Multi-partner RNN · solid: separately fitted time-specific probes; dashed: one fixed prefix-50 probe\n"
           "Fixed decoder is applied across time without refitting; changed feature distributions can affect transfer accuracy.")

    if len(events):
        fig,axes=plt.subplots(2,3,figsize=(16,8),sharex=True,sharey=True)
        for row,event in enumerate(("first_round_end","first_switch")):
            for col,version in enumerate(full.VERSIONS):
                ax=axes[row,col]
                for category,color in zip(CATEGORIES,catcolors):
                    g=events[(events.version==version)&(events.condition==full.CONDITIONS[0])&(events.category==category)&(events.event==event)].sort_values("lag")
                    if len(g):ax.plot(g.lag,g.correct_probability_mean,c=color,label=full.CATEGORY_LABELS[category],lw=1.8)
                ax.axvline(0,c="gray",ls="--");ax.set(title=f"{version} · {event}",ylim=(0,1),xlabel="Observation steps relative to event feedback",ylabel="Mean correct-class probability")
                ax.spines[["top","right"]].set_visible(False)
        finish(fig,axes,"early_event_alignment","Event-conditioned neural confidence · multi-partner RNN\n"
               "Descriptive selection: only events within the saved early window; profiles averaged within each seed. Event time 0 is the next observation.")


def aggregate(path):
    run=full.read(path);out=Path(run["root"])
    tables={name:[] for name in ("curves","event_aligned","prospective","instantaneous","episode_events")}
    for record in run["policies"]:
        directory=Path(record["early_directory"])
        full.require(full.read(directory/"completion.json")["status"]=="completed","Incomplete dense analysis")
        for name in tables:
            frame=pd.read_csv(directory/(name+".csv"))
            if name=="episode_events":
                for key,value in full.identity(record).items():frame[key]=value
            tables[name].append(frame)
    tables={k:pd.concat(v,ignore_index=True) for k,v in tables.items()}
    curves=tables["curves"]
    full.require(len(curves)==45*3*101,"Incomplete primary curves")
    for name,table in tables.items():table.to_csv(out/(name+"_per_seed.csv"),index=False)
    keys=["version","condition","category","env_step"]
    metrics=["accuracy","correct_probability","frozen50_accuracy","action_faster","request_faster","request_available","allocation_history_baseline",
             "round_finished","rounds_finished","both_goals_sampled","any_switch","switch_count","action_switch","round_start"]
    summary=summarize(curves,keys,metrics);summary.to_csv(out/"early_summary.csv",index=False)
    instant=summarize(tables["instantaneous"],keys,["accuracy"]);instant.to_csv(out/"instantaneous_summary.csv",index=False)
    events=summarize(tables["event_aligned"],["version","condition","category","event","lag"],["correct_probability","accuracy"],min_seeds=3)
    events.to_csv(out/"event_aligned_summary.csv",index=False)
    prospective=summarize(tables["prospective"],keys,["partial_r"]);prospective.to_csv(out/"prospective_summary.csv",index=False)
    onsets,differences=onset_table(curves,summary)
    onsets.to_csv(out/"onset_summary.csv",index=False);differences.to_csv(out/"category_timing_differences.csv",index=False)
    # Quantify saved events and choices at the independently defined category window.
    window_rows=[]
    for _,onset in onsets.iterrows():
        g=summary[(summary.version==onset.version)&(summary.condition==onset.condition)&(summary.category==onset.category)].set_index("env_step")
        for t in (int(onset.peak5_start),int(onset.peak5_end)):
            window_rows.append(dict(version=onset.version,condition=onset.condition,category=onset.category,env_step=t,
                **{name:float(g.loc[t,name+"_mean"]) for name in metrics}))
    pd.DataFrame(window_rows).to_csv(out/"behavior_at_sharp_windows.csv",index=False)
    figures(summary,instant,events,onsets,out)
    run.update(status="completed",completed_utc=full.now());full.write(path,run)
    full.write(out/"completion_audit.json",dict(status="passed",policies=45,seeds=5,profiles_in_probe_fitting=90,
        reported_categories=list(CATEGORIES),dense_timepoints=101,instantaneous_timepoints=12,
        original_prefix_probes_reproduced=[1,50,100],primary_rows=len(curves),summary_rows=len(summary),
        no_policy_retraining=True,no_new_rollouts=True,
        sharpness_rule="Largest one-transition gain and largest five-transition gain within 0..50 completed transitions. Half-rise is first three consecutive points reaching half the increase from step 0 to mean steps 40..50; defined only for increase >= .10.",
        uncertainty="Percentile bootstrap over five policy seeds, 10000 resamples, RNG 30000. Timing and event associations are exploratory/descriptive.",
        temporal_rule="Observation t follows t completed transitions; prior-event correlates exclude the current action/transition. Same-step action metrics are subsequent choices, not prior evidence.",
        prospective_rule="Partial correlation of current correct-class probability with faster-goal allocation fraction at subsequent actions t+1..t+10, controlling profile fixed effects, current actual/requested allocation correctness, round count, round time and both-goal exposure; held-out episodes only.",
        limitations=run["limitations"]))
    print(onsets[onsets.condition==full.CONDITIONS[0]][["version","category","peak_step","peak5_start","peak5_end","half_rise_step","half_rise_step_low","half_rise_step_high"]].to_string(index=False),flush=True)
