"""Dense early decoding and temporally ordered correlates from saved rollouts.

No policy optimization, rollouts, or interventions. Prefix length k includes
the initial observation and follows k-1 environment transitions. Outcomes of
transition t enter the information history only at observation t+1.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import shlex
import shutil
import subprocess

import h5py
import numpy as np
import pandas as pd

import full_capability_evaluation as full

CATEGORIES = ("training_profile", "one_delay_novel", "both_delays_novel")
STEPS = np.arange(101)
INSTANT_STEPS = np.array([0,2,3,4,5,8,10,15,20,30,50,100])
N_READ = 122


def load_early(record):
    """Read only early hidden states; preserve the original canonical ordering."""
    fields = ("partner_assignment", "ego_alloc_action", "partner_goal", "round_idx", "round_time",
              "round_done", "success", "rewards", "is_t0")
    batches=[]
    paths=list(record["original_rollouts"].values())+[str(Path(record["directory"])/"additional_profiles.h5")]
    for path in paths:
        with h5py.File(path,"r") as handle:
            done=handle["dones"][:]
            full.require(done.any(axis=1).all(),"Missing episode terminal")
            lengths=done.argmax(axis=1)+1
            hidden=handle["hidden_state"][:,:N_READ]
            full.require(np.isfinite(hidden).all() and hidden.shape[2]==128,"Invalid hidden states")
            pairs=handle["capability"][:,0,:]
            indices=handle["capability_index_per_ep"][:]
            np.testing.assert_array_equal(handle["capability_pool"][:][indices],pairs)
            counters={};reps=[]
            for pair in map(tuple,pairs):
                reps.append(counters.get(pair,0));counters[pair]=reps[-1]+1
            full.require(set(counters.values())=={20},"Incorrect repetition counts")
            item=dict(hidden=hidden,pairs=pairs,reps=np.array(reps),lengths=lengths)
            for name in fields:item[name]=handle[name][:,:N_READ]
            batches.append(item)
    data={key:np.concatenate([b[key] for b in batches]) for key in batches[0]}
    order=np.lexsort((data["reps"],data["pairs"][:,1],data["pairs"][:,0]))
    data={key:value[order] for key,value in data.items()}
    full.require(len(data["pairs"])==1800 and len(np.unique(data["pairs"],axis=0))==90,"Wrong early data coverage")
    with np.load(Path(record["directory"])/"probe_models.npz") as prior:
        np.testing.assert_array_equal(data["pairs"],prior["pairs"])
        np.testing.assert_array_equal(data["reps"],prior["repetitions"])
        data["train"]=prior["train"].copy();data["test"]=prior["test"].copy()
    valid_lengths=np.minimum(data["lengths"],101)
    index=np.minimum(STEPS[None,:],valid_lengths[:,None]-1)
    prefix=np.cumsum(data["hidden"][:,:101],axis=1,dtype=np.float64)
    data["prefix"]=(prefix[np.arange(1800)[:,None],index]/(index+1)[:,:,None]).transpose(1,0,2).astype(np.float32)
    data["instant"]=data["hidden"][np.arange(1800)[:,None],np.minimum(INSTANT_STEPS[None,:],valid_lengths[:,None]-1)].transpose(1,0,2)
    return data


def behavior_history(data):
    """At observation t, separate previous outcomes from the action at t."""
    E=len(data["pairs"]);valid=np.arange(N_READ)[None,:]<data["lengths"][:,None]
    assignment=data["partner_assignment"]
    faster_assignment=np.where(data["pairs"][:,0]<data["pairs"][:,1],2,1)
    correct=(assignment==faster_assignment[:,None]).astype(float)
    # V1 has reserved NONE requests during movement; retain its actual assignment
    # rather than incorrectly treating every such step as a wrong request.
    request_available=np.isin(data["ego_alloc_action"],[1,2])
    effective_request=np.where(request_available,data["ego_alloc_action"],assignment)
    request_correct=(effective_request==faster_assignment[:,None]).astype(float)
    previous=np.c_[np.zeros(E,int),data["partner_goal"][:,:-1]]
    same_round=np.c_[np.zeros(E,bool),data["round_idx"][:,1:]==data["round_idx"][:,:-1]]
    switched=(data["partner_goal"]!=previous)&np.isin(previous,[1,2])&np.isin(data["partner_goal"],[1,2])&same_round&~data["is_t0"]&valid
    round_end=data["round_done"]&valid
    history=lambda values:np.c_[np.zeros(E),np.cumsum(values,axis=1)[:,:100]].T
    red_seen=history((data["partner_goal"]==1)&valid)>0
    blue_seen=history((data["partner_goal"]==2)&valid)>0
    previous_index=np.minimum(np.maximum(STEPS[None,:]-1,0),data["lengths"][:,None]-1)
    prior_correct=correct[np.arange(E)[:,None],previous_index].T
    prior_correct[0]=.5
    prior_valid=np.isin(assignment[np.arange(E)[:,None],previous_index].T,[1,2]);prior_valid[0]=False
    prior_correct=np.where(prior_valid,prior_correct,.5)
    first=lambda event:np.where(event.any(axis=1),event.argmax(axis=1)+1,-1)
    result=dict(action_faster=correct[:,:101].T,request_faster=request_correct[:,:101].T,
        request_available=request_available[:,:101].T,
        allocation_history_baseline=prior_correct,round_finished=history(round_end)>0,
        rounds_finished=history(round_end),both_goals_sampled=red_seen&blue_seen,
        any_switch=history(switched)>0,switch_count=history(switched),
        action_switch=switched[:,:101].T,round_start=data["is_t0"][:,:101].T,
        first_round_end=first(round_end),first_switch=first(switched),
        round_time=np.vstack([np.zeros(E),data["round_time"][:,:100].T]))
    both=(np.cumsum(data["partner_goal"]==1,axis=1)>0)&(np.cumsum(data["partner_goal"]==2,axis=1)>0)&valid
    result["first_both_goals"]=first(both)
    # Prospective outcomes exclude the current action and all invalid padding.
    future=np.full((101,E),np.nan)
    for t in STEPS:
        mask=valid[:,t+1:t+11]
        denominator=mask.sum(axis=1)
        future[t]=np.divide((correct[:,t+1:t+11]*mask).sum(axis=1),denominator,
                             out=np.full(E,np.nan),where=denominator>0)
    result["future_faster_next10"]=future
    return result


def category_labels(manifest,pairs):
    lookup={(p["red_delay"],p["blue_delay"]):p["category"] for p in manifest["profiles"]}
    return np.array([lookup[tuple(p)] for p in pairs])


def partial_correlation(x,y,controls):
    keep=np.isfinite(x)&np.isfinite(y)&np.isfinite(controls).all(axis=1)
    x,y,Z=x[keep],y[keep],controls[keep]
    if len(x)<12:return np.nan,len(x)
    Z=np.c_[np.ones(len(x)),Z]
    residual_x=x-Z@np.linalg.lstsq(Z,x,rcond=None)[0]
    residual_y=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]
    if np.std(residual_x)<1e-8 or np.std(residual_y)<1e-8:return np.nan,len(x)
    return float(np.corrcoef(residual_x,residual_y)[0,1]),len(x)


def policy_tables(record,data,probabilities,predictions,transfer,instant_prob,manifest):
    truth=(data["pairs"][:,1]<data["pairs"][:,0]).astype(int)
    categories=category_labels(manifest,data["pairs"]);test=data["test"]
    behavior=behavior_history(data);rows=[];events=[];prospective=[]
    p_correct=probabilities[:,np.arange(1800),truth]
    for category in CATEGORIES:
        selected=test&(categories==category)
        full.require(selected.sum()==sum(categories==category)//5,"Wrong category test support")
        for t in STEPS:
            row=dict(**full.identity(record),category=category,env_step=int(t),prefix_length=int(t+1),
                n_episodes=int(selected.sum()),accuracy=float((predictions[t,selected]==truth[selected]).mean()),
                correct_probability=float(p_correct[t,selected].mean()),
                frozen50_accuracy=float((transfer[t,selected]==truth[selected]).mean()))
            active=selected&(data["lengths"]>t)
            row["n_active_episodes"]=int(active.sum())
            for name in ("action_faster","request_faster","request_available","allocation_history_baseline","round_finished","rounds_finished",
                         "both_goals_sampled","any_switch","switch_count","action_switch","round_start"):
                chosen=active if name in ("action_faster","request_faster","request_available","action_switch","round_start") else selected
                row[name]=float(behavior[name][t,chosen].mean()) if chosen.any() else np.nan
            rows.append(row)
        for event_name in ("first_round_end","first_switch","first_both_goals"):
            times=behavior[event_name]
            for lag in range(-10,11):
                chosen=np.flatnonzero(selected&(times>=0)&(times+lag>=0)&(times+lag<=100))
                if len(chosen)==0:continue
                time_index=times[chosen]+lag
                # Equal weights across represented profiles, then their episodes.
                values=pd.DataFrame(dict(d_R=data["pairs"][chosen,0],d_B=data["pairs"][chosen,1],
                    p=p_correct[time_index,chosen],accuracy=(predictions[time_index,chosen]==truth[chosen]).astype(float)))
                by_profile=values.groupby(["d_R","d_B"])[["p","accuracy"]].mean()
                events.append(dict(**full.identity(record),category=category,event=event_name,lag=lag,
                    correct_probability=float(by_profile.p.mean()),accuracy=float(by_profile.accuracy.mean()),
                    n_episodes=len(chosen),n_profiles=len(by_profile)))
        for t in (5,10,15,20,30,50):
            chosen=np.flatnonzero(selected)
            pairs=data["pairs"][chosen]
            pair_codes=np.unique(pairs,axis=0,return_inverse=True)[1]
            controls=np.c_[np.eye(pair_codes.max()+1)[pair_codes,1:],
                behavior["action_faster"][t,chosen],behavior["request_faster"][t,chosen],
                behavior["rounds_finished"][t,chosen],behavior["round_time"][t,chosen],behavior["both_goals_sampled"][t,chosen]]
            r,n=partial_correlation(p_correct[t,chosen],behavior["future_faster_next10"][t,chosen],controls)
            prospective.append(dict(**full.identity(record),category=category,env_step=t,partial_r=r,n_episodes=n))
    instant=[]
    for i,t in enumerate(INSTANT_STEPS):
        for category in CATEGORIES:
            selected=test&(categories==category)
            instant.append(dict(**full.identity(record),category=category,env_step=int(t),
                accuracy=float((instant_prob[i,selected].argmax(axis=-1)==truth[selected]).mean()),n_episodes=int(selected.sum())))
    episode_events=pd.DataFrame({"d_R":data["pairs"][:,0],"d_B":data["pairs"][:,1],"repetition":data["reps"],
        "category":categories,"probe_test":test,"first_round_end":behavior["first_round_end"],
        "first_switch":behavior["first_switch"],"first_both_goals":behavior["first_both_goals"]})
    return pd.DataFrame(rows),pd.DataFrame(events),pd.DataFrame(prospective),pd.DataFrame(instant),episode_events


def worker(path,index):
    import torch
    torch.set_num_threads(1)
    run=full.read(path);manifest=full.read(run["input_manifest"]);record=run["policies"][index]
    out=Path(record["early_directory"]);out.mkdir(parents=True,exist_ok=True)
    if (out/"completion.json").exists() and full.read(out/"completion.json").get("analysis_version")==2:return
    full.verify(manifest,record)
    print("[read early rollouts]",full.identity(record),flush=True)
    data=load_early(record)
    binary,_=full.references(manifest)
    truth=binary.relative_classes(data["pairs"])
    features=np.concatenate([data["prefix"],data["instant"]])
    print("[fit]",len(features),"independent reference probes",flush=True)
    cache=out/"fitted_probes.npz"
    if cache.exists():
        with np.load(cache) as saved:
            full.require(saved["probabilities"].shape==(113,1800,2),"Incompatible fitted probe cache")
            models={k:saved[k].copy() for k in ("weight","bias","class_weights","final_train_loss")}
            predictions=saved["predictions"].copy();probabilities=saved["probabilities"].copy()
    else:
        predictions,probabilities,models=binary.train_probes(features,np.broadcast_to(truth,(len(features),1800)).copy(),
            data["train"],[10000]*len(features))
        np.savez_compressed(cache,**models,predictions=predictions,probabilities=probabilities)
    with np.load(Path(record["directory"])/"probe_models.npz") as prior:
        for old_index,step in ((0,0),(1,49),(2,99)):
            np.testing.assert_allclose(models["weight"][step],prior["weight"][old_index],atol=2e-5,rtol=1e-4)
            np.testing.assert_allclose(probabilities[step],prior["probabilities"][old_index],atol=2e-5,rtol=1e-4)
            np.testing.assert_array_equal(predictions[step],prior["predictions"][old_index])
        logits=data["prefix"]@prior["weight"][1]+prior["bias"][1]
        transfer=logits.argmax(axis=-1)
    tables=policy_tables(record,data,probabilities[:101],predictions[:101],transfer,probabilities[101:],manifest)
    for name,table in zip(("curves","event_aligned","prospective","instantaneous","episode_events"),tables):
        table.to_csv(out/(name+".csv"),index=False)
    np.savez_compressed(out/"early_probes.npz",**models,probabilities=probabilities,predictions=predictions,
        pairs=data["pairs"],repetitions=data["reps"],train=data["train"],test=data["test"],
        env_steps=STEPS,instant_steps=INSTANT_STEPS,truth=truth)
    full.write(out/"completion.json",dict(status="completed",completed_utc=full.now(),**full.identity(record),
        anchor_probes_reproduced=[1,50,100],episodes=1800,primary_timepoints=101,analysis_version=2,policy_retraining=False,new_rollouts=False))
    print("[completed]",full.identity(record),flush=True)


def prepare(root,out):
    root=Path(root).resolve();out=Path(out).resolve()
    full.require(not out.exists(),"Use a fresh analysis directory")
    manifest=full.read(root/"analysis_run.json")
    full.require(manifest["status"]=="completed" and full.read(root/"completion_audit.json")["status"]=="passed",
                 "Completed source analysis required")
    out.mkdir(parents=True)
    policies=[]
    for p in manifest["policies"]:
        if not p["condition"].startswith("rnn"):continue
        p=dict(p,early_directory=str(out/"policies"/p["version"]/f"{p['condition']}_seed{p['seed']}"))
        policies.append(p)
    full.require(len(policies)==45,"Expected 45 RNN policies")
    for name in ("analyze_early_decoding.py","full_capability_evaluation.py"):
        target=out/"source"/name;target.parent.mkdir(exist_ok=True)
        shutil.copyfile(Path(__file__).with_name(name),target)
    (out/"logs").mkdir();(out/"jobs").mkdir()
    run=dict(status="prepared",root=str(out),input_manifest=str(root/"analysis_run.json"),policies=policies,
        env_steps=STEPS.tolist(),instantaneous_steps=INSTANT_STEPS.tolist(),categories=list(CATEGORIES),
        seeds=[1,2,3,4,5],analysis_python=full.ANALYSIS_PYTHON,source_sha256={},jobs={},prepared_utc=full.now(),
        method="Same 16/4 per-profile split; all 90 profiles in probe fitting; report green/yellow/pink only. Full-batch reference affine probes, 1000 Adam updates, lr=.01, init 10000.",
        timing="Dense prefix length k = observation states 0..k-1, after k-1 environment transitions. Past event features use only records 0..k-2; action metrics use the current subsequent action.",
        limitations="No saved positions or movement clocks: direct partner-motion evidence, retrieval, Bayesian updating, and predictive coding cannot be identified causally from these observational traces.",
        policy_retraining=False,new_rollouts=False)
    run["source_sha256"]={p.name:full.sha(p) for p in (out/"source").glob("*.py")}
    path=out/"analysis_run.json";full.write(path,run)
    return path


def submit(path):
    path=Path(path).resolve();run=full.read(path);out=Path(run["root"])
    for relative,digest in run["source_sha256"].items():
        full.require(full.sha(out/"source"/relative)==digest,"Analysis source changed")
    clean={k:v for k,v in os.environ.items() if not k.startswith("SLURM_")}
    for mode in ("worker","aggregate"):
        if mode in run["jobs"]:continue
        command=[run["analysis_python"],"-u",str(out/"source/analyze_early_decoding.py"),mode,str(path)]
        if mode=="worker":command+=["--index"]
        lines=["#!/bin/bash","set -euo pipefail","export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1",
            "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1",
            "export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=",'export XLA_FLAGS=""',
            "export MPLCONFIGDIR="+shlex.quote(str(out/"mpl_cache")),
            "exec "+shlex.join(command)+(' "${SLURM_ARRAY_TASK_ID}"' if mode=="worker" else "")]
        script=out/"jobs"/(mode+".sh");script.write_text("\n".join(lines)+"\n")
        subprocess.run(["bash","-n",str(script)],check=True)
        cmd=["sbatch","--parsable","--account=nlp","--partition=sphinx","--exclude=sphinx9",
             "--cpus-per-task=1","--mem=6G","--time=01:00:00","--job-name=early_decode_"+mode,
             "--chdir="+str(Path(run["input_manifest"]).parents[3]),"--kill-on-invalid-dep=yes",
             f"--output={out}/logs/{mode}_%A_%a.out",f"--error={out}/logs/{mode}_%A_%a.err"]
        if mode=="worker":cmd+=["--array=0-44%15"]
        else:cmd+=["--dependency=afterok:"+run["jobs"]["worker"]["job_id"]]
        job_id=subprocess.check_output(cmd+[str(script)],text=True,env=clean).strip().split(";")[0]
        full.require(job_id.isdigit(),"Invalid job ID")
        run["jobs"][mode]=dict(job_id=job_id,command=cmd+[str(script)])
        run["status"]="submitted";full.write(path,run);print(mode,job_id,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","worker","aggregate","submit"))
    parser.add_argument("path",type=Path);parser.add_argument("--out",type=Path);parser.add_argument("--index",type=int)
    args=parser.parse_args()
    if args.mode=="prepare":print(prepare(args.path,args.out),flush=True)
    elif args.mode=="worker":worker(args.path,args.index)
    elif args.mode=="submit":submit(args.path)
    else:
        from plot_early_decoding import aggregate
        aggregate(args.path)


if __name__=="__main__":main()
