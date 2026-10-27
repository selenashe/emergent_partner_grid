"""Train single-state faster-goal probes on familiar capabilities; test all 90.

Only saved post-training rollouts are read. Probes use 16 episodes from each of
the 24 diverse-policy training profiles, with four independent test episodes for
every profile at every environment step 0..400. All other episodes are unused.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import h5py
import numpy as np
import pandas as pd

import full_capability_evaluation as full
import analyze_instantaneous_decoding as instant

STEPS = instant.STEPS
NATIVE_CONDITIONS = ("rnn_diverse_influence", "rnn_diverse_noinfluence")


def capability_split(pairs, base_train, base_test, fit_profiles):
    """Restrict probe fitting by capability without changing the episode test set."""
    pairs = np.asarray(pairs); fit_profiles = np.asarray(fit_profiles)
    base_train = np.asarray(base_train, dtype=bool); test = np.asarray(base_test, dtype=bool).copy()
    full.require(not np.any(base_train & test) and np.all(base_train | test), "Invalid original episode split")
    fit_profile = np.isin(pairs[:,0] * 10 + pairs[:,1], fit_profiles[:,0] * 10 + fit_profiles[:,1])
    train = base_train & fit_profile
    unused = ~(train | test)
    full.require(not np.any(train & test), "Training/test overlap")
    full.require(set(((pairs[train,1] < pairs[train,0]).astype(int)).tolist()) == {0,1},
                 "Probe fitting requires both faster-goal classes")
    for pair in np.unique(pairs, axis=0):
        selected = (pairs == pair).all(axis=1)
        expected = (16,4,0) if fit_profile[selected][0] else (0,4,16)
        full.require(tuple(int(mask[selected].sum()) for mask in (train,test,unused)) == expected,
                     "Incorrect capability/episode partition")
    return train,test,unused,fit_profile


def policy_training_profiles(record):
    config = Path(record["config"])
    audit = config.with_name(config.name.removesuffix("_config.json") + "_sampling_audit.npz")
    with np.load(audit) as saved:
        return np.asarray(saved["capability_pairs"], dtype=int), audit


def verify_run(run):
    full.require(full.sha(run["input_manifest"]) == run["input_manifest_sha256"], "Source manifest changed")
    root = Path(run["root"])
    for name,digest in run["source_sha256"].items():
        full.require(full.sha(root / "source" / name) == digest, "Frozen analysis source changed")


def worker(path,index):
    import torch
    torch.set_num_threads(1)
    run = full.read(path); verify_run(run)
    manifest = full.read(run["input_manifest"]); record = run["policies"][index]
    out = Path(record["probe_directory"]); out.mkdir(parents=True,exist_ok=True)
    if (out / "completion.json").exists(): return
    full.verify(manifest,record)
    extra = full.read(Path(record["directory"]) / "evaluation_completion.json")
    full.require(full.stats(Path(record["directory"]) / "additional_profiles.h5") == extra["rollout_stats"], "Saved rollout changed")
    training_profiles,audit_path = policy_training_profiles(record)
    full.require(full.sha(audit_path) == record["sampling_audit_sha256"], "Training exposure audit changed")
    np.testing.assert_array_equal(training_profiles,record["policy_training_profiles"])
    fit_profiles = np.asarray(run["probe_fit_profiles"])
    with h5py.File(record["original_rollouts"]["train"], "r") as saved:
        np.testing.assert_array_equal(np.unique(saved["capability_pool"][:],axis=0),fit_profiles)
    if record["condition"] in NATIVE_CONDITIONS:
        np.testing.assert_array_equal(np.unique(training_profiles,axis=0),fit_profiles)
    else:
        full.require(run["single_partner_mode"] == "calibrate", "Unapproved single-partner calibration")
        np.testing.assert_array_equal(training_profiles,[[1,4]])
    print("[read saved final-policy states]",full.identity(record),flush=True)
    data = instant.load_states(record)
    train,test,unused,fit_profile = capability_split(data["pairs"],data["train"],data["test"],fit_profiles)
    full.require((int(train.sum()),int(test.sum()),int(unused.sum())) == (384,360,1056),"Wrong split totals")
    binary,_ = full.references(manifest); truth = binary.relative_classes(data["pairs"])
    np.testing.assert_array_equal(np.bincount(truth[train]),[192,192])
    np.testing.assert_array_equal(np.bincount(truth[test]),[180,180])
    full.write(out / "probe_split.json",dict(train_profiles=fit_profiles.tolist(),
        train_repetitions=np.unique(data["repetitions"][train]).tolist(),test_repetitions=np.unique(data["repetitions"][test]).tolist(),
        fit_episodes=384,test_episodes=360,unused_episodes=1056,fitting_class_counts=[192,192],test_class_counts=[180,180],
        calibration_scope=record["calibration_scope"],policy_training_profiles=training_profiles.tolist()))
    table = pd.DataFrame(dict(d_R=data["pairs"][:,0],d_B=data["pairs"][:,1],repetition=data["repetitions"],
        probe_partition=np.where(train,"fit",np.where(test,"test","unused")),probe_fit_profile=fit_profile,
        policy_training_profile=np.isin(data["pairs"][:,0]*10+data["pairs"][:,1],training_profiles[:,0]*10+training_profiles[:,1]),
        episode_length=data["lengths"]))
    table.to_csv(out / "episode_manifest.csv",index=False)
    chunks=[];keys=("weight","bias","class_weights","final_train_loss","predictions","probabilities")
    for start in range(0,len(STEPS),instant.CHUNK):
        steps = STEPS[start:start+instant.CHUNK]; target = out / f"chunk_{start:03d}.npz"
        if not target.exists():
            features = instant.instant_features(data["hidden"],data["lengths"],steps)
            print("[fit 384 familiar-profile examples]",int(steps[0]),int(steps[-1]),flush=True)
            predictions,probabilities,models = binary.train_probes(features,
                np.broadcast_to(truth,(len(steps),len(truth))).copy(),train,[10000]*len(steps))
            np.savez_compressed(target,**models,predictions=predictions,probabilities=probabilities,env_steps=steps,train=train,test=test)
            del features
        with np.load(target) as saved:
            for key,value in (("env_steps",steps),("train",train),("test",test)):
                np.testing.assert_array_equal(saved[key],value)
            chunks.append({key:saved[key].copy() for key in keys})
    output={key:np.concatenate([chunk[key] for chunk in chunks]) for key in keys}
    full.require(output["probabilities"].shape==(401,1800,2),"Missing timepoints")
    np.testing.assert_allclose(output["probabilities"].sum(axis=-1),1,atol=1e-6)
    np.testing.assert_array_equal(output["class_weights"],np.ones((401,2)))
    metrics = instant.profile_metrics(output["predictions"],output["probabilities"],data["pairs"],truth,test,data["lengths"])
    archive=out / "training_capability_probes.npz"
    np.savez_compressed(archive,**output,**metrics,truth=truth,pairs=data["pairs"],repetitions=data["repetitions"],
        train=train,test=test,unused=unused,probe_fit_profile=fit_profile,probe_fit_profiles=fit_profiles,
        policy_training_profiles=training_profiles,lengths=data["lengths"],env_steps=STEPS)
    full.write(out / "completion.json",dict(status="completed",completed_utc=full.now(),**full.identity(record),
        archive_sha256=full.sha(archive),timepoints=401,profiles=90,fit_profiles=24,fit_episodes=384,
        test_episodes=360,unused_episodes=1056,novel_profiles_in_probe_fit=0,
        calibration_scope=record["calibration_scope"],policy_retraining=False,new_rollouts=False))
    print("[completed]",full.identity(record),flush=True)


def prepare(source,out,single_partner_mode):
    source=Path(source).resolve();out=Path(out).resolve()
    full.require(not out.exists(),"Use a fresh results directory")
    manifest=full.read(source / "analysis_run.json")
    full.require(manifest["status"]=="completed" and full.read(source / "completion_audit.json")["status"]=="passed","Incomplete source evaluation")
    profiles=sorted((p["red_delay"],p["blue_delay"]) for p in manifest["profiles"] if p["seen_by_diverse_training"])
    full.require(len(profiles)==24,"Expected 24 actual diverse-training capabilities")
    conditions=full.CONDITIONS if single_partner_mode=="calibrate" else NATIVE_CONDITIONS
    policies=[];excluded=[]
    for record in manifest["policies"]:
        if not record["condition"].startswith("rnn"):continue
        training,audit_path=policy_training_profiles(record)
        if record["condition"] not in conditions:
            excluded.append(dict(**full.identity(record),reason="Only (1,4) was used in policy training; every label is red-faster. A strict-support two-class probe is not identifiable."))
            continue
        policies.append(dict(record,probe_directory=str(out / "policies" / record["version"] / f"{record['condition']}_seed{record['seed']}"),
            policy_training_profiles=training.tolist(),sampling_audit=str(audit_path),sampling_audit_sha256=full.sha(audit_path),
            calibration_scope="own_policy_training_capabilities" if record["condition"] in NATIVE_CONDITIONS else "external_diverse_training_profile_calibration"))
    full.require(len(policies)==3*len(conditions)*5,"Incorrect policy coverage")
    out.mkdir(parents=True);(out/"source").mkdir();(out/"logs").mkdir();(out/"jobs").mkdir()
    for name in ("analyze_training_capability_decoding.py","analyze_instantaneous_decoding.py","plot_instantaneous_decoding.py","full_capability_evaluation.py"):
        shutil.copyfile(Path(__file__).with_name(name),out/"source"/name)
    run=dict(status="prepared",root=str(out),input_manifest=str(source/"analysis_run.json"),input_manifest_sha256=full.sha(source/"analysis_run.json"),
        policies=policies,excluded_policies=excluded,conditions=list(conditions),single_partner_mode=single_partner_mode,
        probe_fit_profiles=profiles,env_steps=STEPS.tolist(),seeds=[1,2,3,4,5],prepared_utc=full.now(),
        analysis_python=full.ANALYSIS_PYTHON,jobs={},source_sha256={p.name:full.sha(p) for p in (out/"source").glob("*.py")},
        fit=dict(profiles=24,episodes_per_profile=16,total=384,class_counts=[192,192]),
        test=dict(profiles=90,episodes_per_profile=4,total=360,class_counts=[180,180],repetitions=[1,9,14,15]),
        unused=dict(profiles=66,episodes_per_profile=16,total=1056),
        method="Independent single-state h_t affine 128->2 probes, t=0..400; raw float32 states; full-batch class-balanced Adam1000 updates, lr .01, init10000. No explicit time averaging. The test repetition split is unchanged.",
        terminal="Carry the final valid hidden state after episode termination; exclude padding. Save active test counts.",
        uncertainty="Five policy seeds; 10000 percentile bootstrap resamples, RNG30000. Category means give profiles equal weights within each seed.",
        policy_retraining=False,new_rollouts=False)
    full.write(out/"analysis_run.json",run)
    inventory=pd.DataFrame(manifest["profiles"]);inventory["probe_fit_profile"]=inventory.seen_by_diverse_training
    inventory["fit_episodes_per_policy"]=np.where(inventory.probe_fit_profile,16,0);inventory["test_episodes_per_policy"]=4
    inventory.to_csv(out/"capability_inventory.csv",index=False)
    (out/"PLAN.md").write_text(plan_text(run))
    return out/"analysis_run.json"


def plan_text(run):
    single = ("Single-partner RNN is omitted: its own policy-training support is only (1,4), supplying red-faster labels only. No degenerate binary accuracy curve is fabricated."
              if run["single_partner_mode"]=="omit" else
              "Single-partner RNN uses explicitly labeled external calibration: hidden states from its frozen-policy evaluations on the 24 diverse-training profiles. The policy itself trained only on (1,4); these probe calibration profiles exceed its policy-training experience.")
    return ("# Training-capability probes evaluated on all 90 profiles\n\n"
        "Read existing post-training evaluation rollouts of the final frozen direct-input, fixed-t0 v1_matched/v2/v3 policies, seeds 1–5. No RL policy retraining and no new rollouts. These are final-policy states on familiar capabilities, not historical states recorded during RL optimization.\n\n"
        "Fit a separate affine 128->2 faster-goal classifier at every environment step t=0..400. The input is h_t after the current observation and before its action, with no explicit time averaging. The label is red-faster (0) or blue-faster (1), determined by comparing the two delays. Equal-delay profiles are excluded.\n\n"
        "Only the 24 diverse-policy training capability pairs enter probe fitting: 16 episodes per profile, 384 examples, with 192 per class. The frozen reference optimizer uses full-batch cross-entropy, Adam1000 updates, lr=.01, no weight decay, init10000, and raw float32 features. Test every one of the 90 unequal pairs on four separate episodes: repetition IDs 1,9,14,15, zero-based. The remaining 1056 examples (16 for each of the 66 novel profiles) never enter probe fitting or reported testing.\n\n"
        "The same held-out episode IDs are used at every timestep, as in the earlier all90-fit analysis. Profile accuracy is correct predictions / 4 for each policy seed, then mean across five seeds. Within one profile this equals recall for its fixed faster-goal class. Shading is the 95% percentile bootstrap over policy seeds (10000 draws, RNG30000). Category means first average profiles within each seed.\n\n"
        "At timesteps past termination, carry the final valid hidden state; never use padding. Active test counts are retained. Colors label diverse-policy training exposure.\n\n"+single+"\n\n"
        "Fit profiles [red delay, blue delay]:\n\n"+", ".join(f"({r},{b})" for r,b in run["probe_fit_profiles"])+"\n\n"
        "Outputs: all90 combined PNG/PDF; 90 individual profile PNGs and multipage PDF; category averages over steps 0..400 and the first60; summary CSV/NPZ; per-policy fitted readouts, probabilities, partitions, and episode manifests; a final integrity and no-leakage audit.\n")


def submit(path):
    path=Path(path).resolve();run=full.read(path);verify_run(run);out=Path(run["root"])
    full.require("worker" not in run["jobs"],"Workers already submitted")
    command=[run["analysis_python"],"-u",str(out/"source/analyze_training_capability_decoding.py"),"worker",str(path),"--index"]
    lines=["#!/bin/bash","set -euo pipefail","export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1",
        "export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=",'export XLA_FLAGS=""',
        "export MPLCONFIGDIR="+shlex.quote(str(out/"mpl_cache")),"exec "+shlex.join(command)+' "${SLURM_ARRAY_TASK_ID}"']
    script=out/"jobs/worker.sh";script.write_text("\n".join(lines)+"\n");subprocess.run(["bash","-n",str(script)],check=True)
    n=len(run["policies"])
    cmd=["sbatch","--parsable","--account=nlp","--partition=sphinx","--exclude=sphinx9","--cpus-per-task=1",
        "--mem=4G","--time=00:20:00","--job-name=traincap_decode",f"--array=0-{n-1}%{n}",
        "--chdir="+str(Path(run["input_manifest"]).parents[3]),f"--output={out}/logs/worker_%A_%a.out",f"--error={out}/logs/worker_%A_%a.err"]
    clean={k:v for k,v in os.environ.items() if not k.startswith("SLURM_")}
    job=subprocess.check_output(cmd+[str(script)],text=True,env=clean).strip().split(";")[0]
    full.require(job.isdigit(),"Invalid scheduler job")
    run["jobs"]["worker"]=dict(job_id=job,command=cmd+[str(script)]);run["status"]="submitted"
    full.write(path,run);print("worker",job,flush=True)


def aggregate(path):
    from plot_instantaneous_decoding import plot_profiles,plot_categories
    run=full.read(path);verify_run(run);manifest=full.read(run["input_manifest"]);full.verify(manifest)
    out=Path(run["root"]);profiles=manifest["profiles"];conditions=tuple(run["conditions"])
    pairs=np.array([(p["red_delay"],p["blue_delay"]) for p in profiles]);fit_profiles=np.array(run["probe_fit_profiles"])
    accuracy=np.full((3,len(conditions),5,401,90),np.nan);active=np.zeros_like(accuracy,dtype=np.uint8)
    pooled=[];policy_audits=[]
    for record in run["policies"]:
        d=Path(record["probe_directory"]);done=full.read(d/"completion.json");archive=d/"training_capability_probes.npz"
        full.require(done["status"]=="completed" and full.sha(archive)==done["archive_sha256"],"Incomplete or changed readouts")
        with np.load(archive) as data:
            for key,value in (("env_steps",STEPS),("profiles",pairs),("probe_fit_profiles",fit_profiles)):
                np.testing.assert_array_equal(data[key],value)
            with np.load(Path(record["directory"])/"probe_models.npz") as original:
                for key in ("pairs","repetitions","test","truth"):
                    np.testing.assert_array_equal(data[key],original[key])
                expected=capability_split(data["pairs"],original["train"],original["test"],fit_profiles)
            for key,value in zip(("train","test","unused","probe_fit_profile"),expected):
                np.testing.assert_array_equal(data[key],value)
            full.require((int(data["train"].sum()),int(data["test"].sum()),int(data["unused"].sum()))==(384,360,1056),"Wrong split totals")
            np.testing.assert_array_equal(np.unique(data["pairs"][data["train"]],axis=0),fit_profiles)
            np.testing.assert_array_equal(data["truth"],(data["pairs"][:,1]<data["pairs"][:,0]).astype(int))
            np.testing.assert_array_equal(data["predictions"],data["probabilities"].argmax(axis=-1))
            np.testing.assert_array_equal(data["class_weights"],np.ones((401,2)))
            recomputed=instant.profile_metrics(data["predictions"],data["probabilities"],data["pairs"],data["truth"],data["test"],data["lengths"])
            for key,value in recomputed.items():np.testing.assert_array_equal(data[key],value)
            v=full.VERSIONS.index(record["version"]);c=conditions.index(record["condition"]);s=record["seed"]-1
            full.require(np.isnan(accuracy[v,c,s]).all(),"Duplicate policy")
            accuracy[v,c,s]=data["accuracy"];active[v,c,s]=data["active_test_episodes"]
            correct=data["predictions"]==data["truth"][None,:]
            for t in STEPS:
                selected=data["test"];red=selected&(data["truth"]==0);blue=selected&(data["truth"]==1)
                pooled.append(dict(**full.identity(record),env_step=int(t),accuracy=correct[t,selected].mean(),
                    red_faster_recall=correct[t,red].mean(),blue_faster_recall=correct[t,blue].mean(),n_test_episodes=360))
        policy_audits.append(done)
    full.require(np.isfinite(accuracy).all(),"Missing policy seeds or profiles")
    means,lows,highs=instant.bootstrap_seeds(accuracy.transpose(2,0,1,3,4))
    np.savez_compressed(out/"profile_probe_summary.npz",accuracy_mean=means,accuracy_ci_low=lows,accuracy_ci_high=highs,
        accuracy_per_seed=accuracy,active_test_episodes=active,env_steps=STEPS,profiles=pairs,versions=full.VERSIONS,
        conditions=conditions,seeds=np.arange(1,6),probe_fit_profiles=fit_profiles)
    rows=[]
    for v,version in enumerate(full.VERSIONS):
        for c,condition in enumerate(conditions):
            for i,p in enumerate(profiles):
                for t in STEPS:
                    rows.append(dict(version=version,condition=condition,d_R=p["red_delay"],d_B=p["blue_delay"],category=p["category"],
                        probe_fit_profile=bool(p["seen_by_diverse_training"]),env_step=int(t),accuracy_mean=means[v,c,t,i],
                        accuracy_ci_low=lows[v,c,t,i],accuracy_ci_high=highs[v,c,t,i],seed_count=5,active_test_episodes=int(active[v,c,:,t,i].sum())))
    pd.DataFrame(rows).to_csv(out/"profile_probe_summary.csv",index=False);pd.DataFrame(pooled).to_csv(out/"pooled_probe_per_seed.csv",index=False)
    labels=[full.LABELS[full.CONDITIONS.index(c)] for c in conditions]
    if run["single_partner_mode"]=="calibrate":labels[conditions.index("rnn_single_influence")]="Single-partner RNN (24-profile calibration)"
    note="Fit: 24 training profiles × 16 episodes; test: all 90 profiles × 4 episodes; five seeds.\n"
    if run["single_partner_mode"]=="omit":note+="Single-partner control omitted: one-class training support."
    else:note+="Single-partner probe uses external calibration on the 24 diverse-training profiles."
    print("[draw all90 profiles]",flush=True)
    plot_profiles(means,lows,highs,profiles,out,conditions=conditions,labels=labels,method_note=note)
    print("[draw category averages]",flush=True)
    plot_categories(accuracy,profiles,out,conditions=conditions,labels=labels,
        method_note="Probe fit: 24 training profiles; test: all 90; equal profile weights; five seeds, 95% bootstrap CI")
    full.require(len(list((out/"per_profile").glob("*.png")))==90,"Missing profile figures")
    full.verify(manifest);verify_run(run)
    audit=dict(status="passed",completed_utc=full.now(),policies=len(policy_audits),policy_seed_count=5,versions=3,profiles=90,
        fit_profiles=24,novel_fit_profiles=0,fit_episodes_per_policy=384,fit_class_counts=[192,192],
        test_episodes_per_policy=360,test_class_counts=[180,180],unused_episodes_per_policy=1056,
        train_test_overlap=0,test_episode_split_matches_previous=True,timepoints=401,first_env_step=0,last_env_step=400,
        total_independent_probes=len(policy_audits)*401,per_profile_figures=90,metrics_recomputed=True,
        excluded_single_partner_policies=len(run["excluded_policies"]),single_partner_mode=run["single_partner_mode"],
        carried_terminal_test_episode_points=int((4-active.astype(int)).sum()),policy_retraining=False,new_rollouts=False)
    full.write(out/"completion_audit.json",audit)
    (out/"README.md").write_text(plan_text(run)+"\nFinal audit passed: all reported metrics were recomputed from saved predictions and the fit/test capability and episode masks were independently verified.\n")
    run.update(status="completed",completed_utc=full.now());full.write(path,run)
    print("[completed]",out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","submit","worker","aggregate"));parser.add_argument("path",type=Path)
    parser.add_argument("--out",type=Path);parser.add_argument("--index",type=int)
    parser.add_argument("--single-partner",choices=("omit","calibrate"),default="omit")
    args=parser.parse_args()
    if args.mode=="prepare":print(prepare(args.path,args.out,args.single_partner),flush=True)
    elif args.mode=="submit":submit(args.path)
    elif args.mode=="worker":worker(args.path,args.index)
    else:aggregate(args.path)


if __name__=="__main__":main()
