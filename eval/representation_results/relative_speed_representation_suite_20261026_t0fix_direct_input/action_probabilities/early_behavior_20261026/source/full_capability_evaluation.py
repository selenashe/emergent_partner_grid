"""Evaluate saved direct-input policies on all 90 unequal delay profiles.

Reuse the original 46 profiles, collect the other 44 with each policy's frozen
evaluator, and fit the reference faster-goal probes on held-out episodes. No
policy optimization or training entry point is invoked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
COHORT = "v1_matched_v2_v3_20261022_080155_t0fix_direct_input"
EVAL_PYTHON = "/nlp/scr/jshe/miniconda3/envs/emergent_partner_model/bin/python"
ANALYSIS_PYTHON = "/juice6/u/nlp/climblab/tmp/epg_repr_all10_20261008/bin/python"
VERSIONS = ("v1_matched", "v2", "v3")
CONDITIONS = ("rnn_diverse_influence", "rnn_single_influence", "rnn_diverse_noinfluence")
LABELS = ("Multi-partner RNN", "Single-partner RNN", "No-influence RNN")
COLORS = ("#9380ff", "#e67e00", "#dbb528")
CATEGORY_COLORS = dict(training_profile="#e3f1e5", unseen_pair_of_familiar_delays="#e8eef5",
                       one_delay_novel="#fff0d5", both_delays_novel="#f5dff0")
CATEGORY_LABELS = dict(training_profile="Seen: diverse training", unseen_pair_of_familiar_delays="Unseen pair; familiar delays",
                       one_delay_novel="Unseen: one novel delay", both_delays_novel="Unseen: both delays novel")
TIMES = np.array([1, 50, 100, 150, 200, 250, 300, 350, 400])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns)


def identity(record):
    return dict(version=record["version"], condition=record["condition"], training_seed=record["seed"])


def references(manifest):
    base = Path(manifest["root"]) / "source/reference"
    sys.path.insert(0, str(base))
    sys.path.insert(0, str(base / "eval"))
    import relative_speed_representation as binary
    return binary, binary.original


def prepare(root):
    root = Path(root).resolve()
    require(not root.exists(), "Use a new output directory")
    training_path = REPO / "train/manifests" / f"sbatch_{COHORT}.json"
    training = read(training_path)
    review_path = REPO / "eval/evaluation_plans/t0fix_direct_input_capability_review_20261022/capability_review.json"
    review = read(review_path)
    require(sha(training_path) == review["training_manifest_sha256"], "Reviewed training cohort changed")
    profiles = review["proposed_profiles"]
    expected = {(r, b) for r in range(10) for b in range(10) if r != b}
    require(len(profiles) == 90 and {(p["red_delay"], p["blue_delay"]) for p in profiles} == expected,
            "Expected the approved 90 unequal profiles")
    require(training["status"] == "completed" and training["seeds"] == [1, 2, 3, 4, 5], "Wrong training cohort")
    previous = REPO / "eval/representation_results" / f"relative_speed_{COHORT}"
    policies = []
    root.mkdir(parents=True)
    for job in training["training_jobs"]:
        data = training["versions"][job["version"]]
        prefix = Path(data["checkpoint_root"]) / job["tag"]
        config_path = Path(str(prefix) + "_config.json").resolve()
        checkpoint = Path(str(prefix) + ".safetensors").resolve()
        config = read(config_path)
        require(config["GRID_ENCODER"] == "flat" and config["INCLUDE_IS_T0"] is True
                and config["RUN_UPDATES"] == 915 and config["SEED"] == job["seed"], "Wrong saved policy settings")
        require(config["ENV_KWARGS"]["step_penalty"] == .01, "Wrong reward settings")
        variant = training["source_variants"][job["source_variant"]]
        source = Path(variant["frozen_source_root"])
        for relative, digest in variant["source_sha256"].items():
            require(sha(source / relative) == digest, "Frozen policy source changed: " + relative)
        paths = {s: str((Path(data["evaluation_root"]) / (job["tag"] + f"_{s}.h5")).resolve()) for s in ("train", "test")}
        configured = {tuple(pair) for pair in config["ENV_KWARGS"]["partner_capability_pairs"]}
        intended = {(1, 4)} if job["condition"] == "rnn_single_influence" else {tuple(p) for p in review["existing_training_profiles"]}
        require(configured == intended, "Unexpected training exposure")
        directory = root / "policies" / job["version"] / f"{job['condition']}_seed{job['seed']}"
        directory.mkdir(parents=True)
        for name, path in paths.items():
            with h5py.File(path, "r") as handle:
                require(handle["dones"].shape == ((480 if name == "train" else 440), 2000), "Original rollout dimensions changed")
                if job["condition"].startswith("rnn"):
                    require(handle["hidden_state"].shape[-1] == 128, "Missing original hidden states")
            (directory / f"original_{name}.h5").symlink_to(path)
        policies.append(dict(version=job["version"], condition=job["condition"], seed=job["seed"],
            config=str(config_path), config_sha256=sha(config_path), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
            source_root=str(source), source_sha256=variant["source_sha256"],
            layouts_dir=training["frozen_layouts_dir"], allocation_protocol=data["allocation_protocol"],
            original_rollouts=paths, original_stats={k: stats(p) for k, p in paths.items()}, directory=str(directory)))
    require(len(policies) == 60 and len({(p['version'],p['condition'],p['seed']) for p in policies}) == 60,
            "Expected all 60 saved policies")
    for relative in ("eval/relative_speed_representation.py", "eval/representation_analysis.py", "repo_paths.py",
                     "jaxmarl/environments/coordination_grid/capability_populations.py"):
        source = previous / "source" / relative
        target = root / "source/reference" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(__file__, root / "source/full_capability_evaluation.py")
    shutil.copyfile(review_path, root / "capability_review.json")
    shutil.copyfile(review_path.with_name("selected_capabilities_90.csv"), root / "capabilities.csv")
    (root / "jobs").mkdir(); (root / "logs").mkdir()
    manifest = dict(status="prepared", prepared_utc=now(), root=str(root), repo_root=str(REPO),
        training_manifest=str(training_path), training_manifest_sha256=sha(training_path),
        approval="User approved all 90 unequal profiles with 'go ahead' on 2026-10-22.",
        policy_retraining=False, profiles=profiles, n_profiles=90, episodes_per_profile=20,
        episodes_per_policy=1800, reused_profiles=46, additional_profiles=44, evaluation_seed=12345,
        versions=list(VERSIONS), conditions=list(CONDITIONS), seeds=[1,2,3,4,5], policies=policies,
        rnn_indices=[i for i,p in enumerate(policies) if p["condition"].startswith("rnn")],
        source_sha256={str(p.relative_to(root)): sha(p) for p in (root/"source").rglob("*.py")},
        eval_python=EVAL_PYTHON, analysis_python=ANALYSIS_PYTHON, jobs={}, validation={})
    write(root / "analysis_run.json", manifest)
    return root / "analysis_run.json"


def verify(manifest, record=None):
    root = Path(manifest["root"])
    require(sha(manifest["training_manifest"]) == manifest["training_manifest_sha256"], "Training manifest changed")
    for relative, digest in manifest["source_sha256"].items():
        require(sha(root / relative) == digest, "Frozen analysis source changed: " + relative)
    for item in ([record] if record else manifest["policies"]):
        for key in ("checkpoint", "config"):
            require(sha(item[key]) == item[key + "_sha256"], "Saved policy changed: " + item[key])
        for relative, digest in item["source_sha256"].items():
            require(sha(Path(item["source_root"]) / relative) == digest, "Frozen evaluator source changed")
        for name, path in item["original_rollouts"].items():
            require(stats(path) == item["original_stats"][name], "Original rollout changed")


def load_evaluator(record):
    source = Path(record["source_root"])
    sys.path.insert(0, str(source))
    path = source / "analysis/evaluate_partner_modelling.py"
    spec = importlib.util.spec_from_file_location("frozen_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(Path(module.trainer_mod.__file__).resolve().is_relative_to(source.resolve()), "Wrong trainer import")
    require(Path(sys.modules[module.CoordinationGrid.__module__].__file__).resolve().is_relative_to(source.resolve()),
            "Wrong environment import")
    return module


def collect(path, index, smoke=False):
    manifest = read(path); record = manifest["policies"][index]
    verify(manifest, record)
    directory = Path(record["directory"])
    if not smoke and (directory / "evaluation_completion.json").exists():
        require(read(directory / "evaluation_completion.json")["status"] == "completed", "Invalid previous completion")
        return
    evaluator = load_evaluator(record)
    config = read(record["config"])
    params = evaluator.load_params(record["checkpoint"])
    profiles = [(p["red_delay"],p["blue_delay"]) for p in manifest["profiles"] if p["requires_new_profile_rollout"]]
    profiles = [(5,6)] if smoke else profiles
    print("[evaluate]", identity(record), len(profiles), "new profiles", flush=True)
    result, env = evaluator.rollout_condition(params, config, profiles, 1 if smoke else 20,
        record["layouts_dir"], manifest["evaluation_seed"], record["condition"].startswith("rnn"))
    require(result["dones"].shape == (len(profiles)*(1 if smoke else 20),2000), "Wrong rollout shape")
    require(np.all(result["dones"].any(axis=1)), "Incomplete evaluation episode")
    require(np.array_equal(result["capability_pool"], profiles), "Wrong capability pool")
    require(env.influence == bool(config["INFLUENCE"]), "Wrong influence setting")
    target = directory / ("smoke_new_profile.h5" if smoke else "additional_profiles.h5")
    temporary = target.with_suffix(".partial.h5")
    with h5py.File(temporary,"w") as handle:
        handle.attrs.update(allocation_protocol=record["allocation_protocol"], seed=manifest["evaluation_seed"],
                            policy_checkpoint_sha256=record["checkpoint_sha256"], episodes_per_profile=1 if smoke else 20)
        for key, value in result.items():
            value = np.asarray(value)
            if value.ndim >= 2:
                handle.create_dataset(key,data=value,compression="gzip",compression_opts=1,shuffle=True)
            else:
                handle.create_dataset(key,data=value)
    temporary.replace(target)
    completion = dict(status="completed", completed_utc=now(), **identity(record), profiles=len(profiles),
                      episodes=len(result["dones"]), rollout=str(target), rollout_stats=stats(target),
                      backend=evaluator.jax.default_backend(), no_policy_retraining=True)
    write(directory / ("smoke_completion.json" if smoke else "evaluation_completion.json"), completion)
    print("[completed]", completion, flush=True)


def extract(path, expected_pairs, hidden_required=True, repetitions=20, protocol=None):
    """Read a slice with terminal-inclusive features and behavior, rejecting padding."""
    pairs, reps, features, behavior, episodes = [], [], [], [], []
    counts = Counter()
    with h5py.File(path,"r") as handle:
        E,T = handle["dones"].shape
        require(E == len(expected_pairs)*repetitions and T == 2000,"Wrong slice episode count")
        pool = handle["capability_pool"][:]
        indices = handle["capability_index_per_ep"][:]
        require({tuple(p) for p in pool} == set(expected_pairs),"Wrong stored capability pool")
        stored_protocol = handle.attrs.get("allocation_protocol", "fixed_allocation")
        if isinstance(stored_protocol,bytes): stored_protocol=stored_protocol.decode()
        if protocol is not None: require(stored_protocol == protocol,"Wrong rollout allocation protocol")
        require(indices.shape == (E,) and np.all((indices>=0)&(indices<len(pool))), "Invalid capability indices")
        fields = ["dones","capability","round_done","round_idx","rewards","success","partner_assignment","is_t0"]
        if hidden_required:
            require(handle["hidden_state"].shape == (E,T,128),"Expected 128-dimensional GRU states")
            fields.append("hidden_state")
        for start in range(0,E,20):
            block={k:handle[k][start:start+20] for k in fields}
            for local in range(len(block["dones"])):
                e=start+local
                dones=block["dones"][local]; ends=np.flatnonzero(dones)
                require(len(ends)>0,"No episode terminal")
                length=int(ends[0])+1
                pair=tuple(int(x) for x in block["capability"][local,0])
                require(pair in expected_pairs and np.all(block["capability"][local,:length]==pair),"Invalid episode capability")
                require(tuple(pool[indices[e]]) == pair,"Capability metadata differs")
                round_ends=np.flatnonzero(block["round_done"][local,:length])
                require(len(round_ends)==20 and round_ends[-1]==length-1,"Missing ordered round ends")
                require(np.array_equal(block["round_idx"][local,round_ends],np.arange(20)),"Wrong round indices")
                expected_rounds=np.repeat(np.arange(20),np.diff(np.r_[0,round_ends+1]))
                require(np.array_equal(block["round_idx"][local,:length],expected_rounds),"Inconsistent round boundaries")
                rep=counts[pair];counts[pair]+=1
                pairs.append(pair);reps.append(rep)
                if hidden_required:
                    hidden=block["hidden_state"][local,:length]
                    require(np.isfinite(block["hidden_state"][local]).all(),"Nonfinite hidden scan")
                    features.append(prefix_features(hidden,TIMES))
                decisions=~block["is_t0"][local,:length].astype(bool)
                # Allocations.blue means ego blue, partner red in all frozen protocols.
                assignments=block["partner_assignment"][local,:length]
                require(np.isin(assignments[decisions],[1,2]).all(),"Invalid actual partner assignment")
                actual_red=assignments==2
                correct=actual_red if pair[0]<pair[1] else ~actual_red
                behavior.append(dict(d_R=pair[0],d_B=pair[1],rollout_rep=rep,
                    episode_return=float(block["rewards"][local,:length].sum(dtype=np.float64)),
                    round_success_rate=float(block["success"][local,round_ends].mean()),
                    mean_round_steps=length/20,
                    fraction_partner_assigned_faster=float(correct[decisions].mean()) if decisions.any() else None))
                episodes.append(dict(source_file=str(Path(path).resolve()),source_episode=e,d_R=pair[0],d_B=pair[1],
                                     rollout_rep=rep,valid_length=length,scan_length=T))
    require(set(counts)==set(expected_pairs) and set(counts.values())=={repetitions},"Missing or duplicate profile repetitions")
    return dict(pairs=np.array(pairs),repetitions=np.array(reps),
                features=np.array(features).transpose(1,0,2) if hidden_required else None,
                behavior=pd.DataFrame(behavior),episodes=pd.DataFrame(episodes))


def prefix_features(hidden,cutoffs):
    require(hidden.ndim==2 and len(hidden)>0 and np.isfinite(hidden).all(),"Invalid hidden prefix")
    cutoffs=np.minimum(np.asarray(cutoffs,dtype=int),len(hidden))
    require(np.all(cutoffs>0),"Cutoffs must be positive")
    prefix=np.cumsum(hidden,axis=0,dtype=np.float64)
    return (prefix[cutoffs-1]/cutoffs[:,None]).astype(np.float32)


def load_all(manifest,record):
    hidden=record["condition"].startswith("rnn")
    inventory=manifest["profiles"]
    train={tuple(p) for p in read(Path(manifest["root"])/"capability_review.json")["existing_training_profiles"]}
    old={tuple(p) for p in read(Path(manifest["root"])/"capability_review.json")["existing_held_out_profiles"]}
    extra={(p["red_delay"],p["blue_delay"]) for p in inventory if p["requires_new_profile_rollout"]}
    results=[extract(record["original_rollouts"]["train"],train,hidden,protocol=record["allocation_protocol"]),
             extract(record["original_rollouts"]["test"],old,hidden,protocol=record["allocation_protocol"]),
             extract(Path(record["directory"])/"additional_profiles.h5",extra,hidden,protocol=record["allocation_protocol"])]
    pairs=np.concatenate([x["pairs"] for x in results]); reps=np.concatenate([x["repetitions"] for x in results])
    order=np.lexsort((reps,pairs[:,1],pairs[:,0]))
    require(len(pairs)==1800 and len(np.unique(pairs,axis=0))==90,"Incomplete expanded dataset")
    return dict(pairs=pairs[order],repetitions=reps[order],
                features=np.concatenate([x["features"] for x in results],axis=1)[:,order] if hidden else None,
                behavior=pd.concat([x["behavior"] for x in results],ignore_index=True).iloc[order].reset_index(drop=True),
                episodes=pd.concat([x["episodes"] for x in results],ignore_index=True).iloc[order].reset_index(drop=True))


def profile_metrics(predictions, probabilities, pairs, test):
    rows=[]
    require(predictions.shape==(len(TIMES),len(pairs)) and test.shape==(len(pairs),),"Invalid prediction dimensions")
    truth=(pairs[:,1]<pairs[:,0]).astype(int)
    for red,blue in sorted({tuple(p) for p in pairs}):
        selected=test & np.all(pairs==(red,blue),axis=1)
        require(selected.sum()==4,"Each profile needs four held-out probe episodes")
        label=int(blue<red)
        for i,cutoff in enumerate(TIMES):
            correct=int((predictions[i,selected]==truth[selected]).sum())
            rows.append(dict(d_R=int(red),d_B=int(blue),cutoff=int(cutoff),true_class=label,
                n_episodes=4,n_correct=correct,accuracy=correct/4,
                mean_correct_class_probability=float(probabilities[i,selected,label].mean())))
    return pd.DataFrame(rows)


def analyze(path,index):
    import torch
    torch.set_num_threads(1)
    manifest=read(path);record=manifest["policies"][index];verify(manifest,record)
    directory=Path(record["directory"])
    if (directory/"analysis_completion.json").exists():
        require(read(directory/"analysis_completion.json")["status"]=="completed","Invalid existing analysis")
        return
    require(read(directory/"evaluation_completion.json")["status"]=="completed","Evaluation incomplete")
    data=load_all(manifest,record)
    for key in identity(record): data["behavior"][key]=identity(record)[key]
    data["behavior"].to_csv(directory/"behavior_episodes.csv",index=False)
    if not record["condition"].startswith("rnn"):
        write(directory/"analysis_completion.json",dict(status="completed",**identity(record),episodes=1800,probe=False))
        return
    binary,reference=references(manifest)
    # Check the generalized loader against the exact reference feature primitive.
    np.testing.assert_array_equal(prefix_features(np.ones((11,128),np.float32),TIMES),
                                  reference.prefix_mean_hidden(np.ones((11,128),np.float32),TIMES))
    split=reference.make_probe_split(0)
    train,test=binary.masks(data["repetitions"],data["pairs"],split)
    labels=binary.relative_classes(data["pairs"])
    require((train.sum(),test.sum())==(1440,360),"Wrong expanded probe split")
    require(np.array_equal(np.bincount(labels[train]),[720,720]) and
            np.array_equal(np.bincount(labels[test]),[180,180]),"Unbalanced faster-goal labels")
    predictions,probabilities,models=binary.train_probes(data["features"],np.broadcast_to(labels,(9,1800)).copy(),train,[10000]*9)
    np.savez_compressed(directory/"probe_models.npz",**models,predictions=predictions,probabilities=probabilities,
                        truth=labels,pairs=data["pairs"],repetitions=data["repetitions"],train=train,test=test,time_cutoffs=TIMES)
    table=profile_metrics(predictions,probabilities,data["pairs"],test)
    for key,value in identity(record).items(): table[key]=value
    table.to_csv(directory/"profile_probe_per_seed.csv",index=False)
    pooled=[]
    for i,t in enumerate(TIMES):
        metrics,_=binary.class_metrics(labels[test],predictions[i,test])
        pooled.append(dict(**identity(record),cutoff=int(t),**metrics))
    pd.DataFrame(pooled).to_csv(directory/"pooled_probe_per_seed.csv",index=False)
    data["episodes"]["probe_partition"]=np.where(train,"train","test")
    data["episodes"].to_csv(directory/"episode_manifest.csv",index=False)
    write(directory/"probe_split.json",split)
    write(directory/"analysis_completion.json",dict(status="completed",completed_utc=now(),**identity(record),
        episodes=1800,profiles=90,probe=True,probe_train_episodes=1440,probe_test_episodes=360,
        fitting_class_counts=[720,720],test_class_counts=[180,180],probe_model_sha256=sha(directory/"probe_models.npz")))
    print("[analyzed]",identity(record),flush=True)


def summarize_profiles(table):
    keys=["version","condition","d_R","d_B","cutoff"]
    require(not table.duplicated(keys+["training_seed"]).any(),"Duplicate seed/profile metrics")
    rng=np.random.default_rng(30000)
    bootstrap=rng.integers(0,5,size=(10000,5))
    rows=[]
    for group,subset in table.groupby(keys,sort=True):
        require(set(subset.training_seed)=={1,2,3,4,5} and len(subset)==5,"Require all five policy seeds")
        require(np.all(subset.n_episodes==4),"Expected four held-out episodes per policy/profile")
        values=subset.sort_values("training_seed").accuracy.to_numpy()
        require(np.isfinite(values).all() and np.all((values>=0)&(values<=1)),"Invalid profile accuracy")
        means=values[bootstrap].mean(axis=1)
        rows.append(dict(zip(keys,group),accuracy_mean=float(values.mean()),
            accuracy_ci_low=float(np.quantile(means,.025)),accuracy_ci_high=float(np.quantile(means,.975)),
            seed_count=5,held_out_episodes_per_seed=4))
    return pd.DataFrame(rows)


def draw_profile(axes,summary,profile,compact=False):
    red,blue=profile["red_delay"],profile["blue_delay"]
    category=profile["category"]
    exposure=CATEGORY_LABELS[category]
    if (red,blue)==(1,4): exposure+="; also single-partner training"
    for ax,version in zip(axes,VERSIONS):
        for condition,color,label in zip(CONDITIONS,COLORS,LABELS):
            g=summary[(summary.version==version)&(summary.condition==condition)&(summary.d_R==red)&(summary.d_B==blue)].sort_values("cutoff")
            require(np.array_equal(g.cutoff.to_numpy(),TIMES),"Missing plot cutoffs")
            ax.plot(g.cutoff,g.accuracy_mean,c=color,lw=1 if compact else 2,label=label)
            ax.fill_between(g.cutoff.to_numpy(),g.accuracy_ci_low.to_numpy(),g.accuracy_ci_high.to_numpy(),color=color,alpha=.18)
        ax.axhline(.5,c="gray",ls="--",lw=.6 if compact else 1,label="Random guess (1/2)")
        ax.set_ylim(0,1.02);ax.set_xlim(0,400)
        ax.set_facecolor(CATEGORY_COLORS[category]);ax.spines[["top","right"]].set_visible(False)
        ax.set_title(f"({red},{blue}) · {version}" if compact else version,fontsize=6.4 if compact else 11,pad=3)
        ax.set_xticks([0,200,400] if compact else [0,100,200,300,400])
        ax.set_yticks([0,.5,1])
        ax.tick_params(labelsize=5 if compact else 9,length=2 if compact else 3)
        ax.set_xlabel("Timesteps",fontsize=5.5 if compact else 10,labelpad=1)
    axes[0].set_ylabel("Accuracy",fontsize=5.5 if compact else 10,labelpad=1)
    if compact:
        axes[1].text(.5,1.22,exposure,transform=axes[1].transAxes,ha="center",va="bottom",fontsize=5.4)
    return exposure


def plot_profiles(summary,profiles,root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    individuals=root/"per_profile";individuals.mkdir(exist_ok=True)
    legend=[Line2D([0],[0],color=c,lw=2,label=l) for c,l in zip(COLORS,LABELS)]
    legend+=[Line2D([0],[0],color="gray",ls="--",label="Random guess (1/2)")]
    with PdfPages(root/"relative_speed_timestep_per_profile.pdf") as pdf:
        for profile in profiles:
            fig,axes=plt.subplots(1,3,figsize=(15,4.8),sharey=True)
            exposure=draw_profile(axes,summary,profile)
            red,blue=profile["red_delay"],profile["blue_delay"]
            fig.suptitle(f"Partner ({red},{blue}) [red delay, blue delay] · {exposure}\n"
                         "Five policy seeds; shading: 95% bootstrap CI; four held-out probe episodes per seed",fontsize=11)
            axes[-1].legend(handles=legend,frameon=False,fontsize=8,loc="lower right")
            fig.tight_layout(rect=(0,0,1,.89))
            fig.savefig(individuals/f"relative_speed_timestep_dR{red}_dB{blue}.png",dpi=160)
            pdf.savefig(fig);plt.close(fig)
    rows=(len(profiles)+2)//3
    height=rows*2.3+1.8
    fig=plt.figure(figsize=(27,height))
    grid=fig.add_gridspec(rows,9,left=.025,right=.995,bottom=.3/height,top=1-2.0/height,wspace=.32,hspace=.95)
    for i,profile in enumerate(profiles):
        r,c=divmod(i,3)
        axes=[fig.add_subplot(grid[r,c*3+j]) for j in range(3)]
        draw_profile(axes,summary,profile,compact=True)
    fig.suptitle("Faster-goal decoding by partner capability · v1_matched, v2, v3 · direct input, fixed t0\n"
        f"All {len(profiles)} unequal profiles; five policy seeds; shading: 95% bootstrap CI. Policy training exposure is marked by panel color.\n"
        "Probe split: 16 fitting / 4 held-out episodes per profile. Single-partner policy training saw only (1,4).",fontsize=12,y=1-.05/height)
    legend+=[Patch(facecolor=CATEGORY_COLORS[k],label=CATEGORY_LABELS[k]) for k in CATEGORY_COLORS]
    fig.legend(handles=legend,loc="upper center",bbox_to_anchor=(.5,1-.8/height),ncol=4,frameon=False,fontsize=8)
    for extension in ("png","pdf"):
        fig.savefig(root/f"relative_speed_timestep_all_profiles_all_seeds.{extension}",dpi=140)
    plt.close(fig)


def audit_policy(manifest,record):
    directory=Path(record["directory"])
    completion=read(directory/"evaluation_completion.json")
    require(completion["profiles"]==44 and completion["episodes"]==880,"Incomplete extra evaluations")
    require(stats(directory/"additional_profiles.h5")==completion["rollout_stats"],"Additional rollout changed")
    done=read(directory/"analysis_completion.json")
    require(done["status"]=="completed" and done["episodes"]==1800,"Incomplete policy analysis")
    if record["condition"].startswith("rnn"):
        with np.load(directory/"probe_models.npz") as data:
            require(data["weight"].shape==(9,128,2),"Wrong binary probe architecture")
            require(np.all(data["class_weights"]==1),"Unexpected class weights")
            np.testing.assert_array_equal(data["predictions"],data["probabilities"].argmax(axis=-1))
            np.testing.assert_allclose(data["probabilities"].sum(axis=-1),1,atol=1e-6)
            pairs=data["pairs"];train=data["train"];test=data["test"]
            require(len(pairs)==1800 and len(np.unique(pairs,axis=0))==90,"Incomplete probe profiles")
            require(not np.any(train&test) and np.all(train|test),"Probe split overlaps")
            require((train.sum(),test.sum())==(1440,360),"Wrong split totals")
            np.testing.assert_array_equal(data["truth"],(pairs[:,1]<pairs[:,0]).astype(int))
            split=read(directory/"probe_split.json")
            np.testing.assert_array_equal(test,np.isin(data["repetitions"],split["test_reps"]))
            computed=profile_metrics(data["predictions"],data["probabilities"],pairs,test)
            saved=pd.read_csv(directory/"profile_probe_per_seed.csv")
            for column in computed:
                np.testing.assert_allclose(computed[column],saved[column],rtol=1e-6,atol=1e-7)
            require(sha(directory/"probe_models.npz")==done["probe_model_sha256"],"Probe archive changed")


def aggregate(path):
    manifest=read(path);verify(manifest);root=Path(manifest["root"])
    tables=[];behavior=[];pooled=[]
    for record in manifest["policies"]:
        audit_policy(manifest,record)
        directory=Path(record["directory"])
        behavior.append(pd.read_csv(directory/"behavior_episodes.csv"))
        if record["condition"].startswith("rnn"):
            tables.append(pd.read_csv(directory/"profile_probe_per_seed.csv"))
            pooled.append(pd.read_csv(directory/"pooled_probe_per_seed.csv"))
    table=pd.concat(tables,ignore_index=True)
    require(len(table)==45*90*9,"Incorrect per-profile decoding coverage")
    summary=summarize_profiles(table)
    require(len(summary)==3*3*90*9,"Incomplete five-seed summaries")
    inventory=pd.DataFrame(manifest["profiles"]).rename(columns={"red_delay":"d_R","blue_delay":"d_B"})
    summary=summary.merge(inventory,on=["d_R","d_B"],validate="many_to_one")
    table.to_csv(root/"profile_probe_per_seed.csv",index=False)
    summary.to_csv(root/"profile_probe_summary.csv",index=False)
    pd.concat(pooled,ignore_index=True).to_csv(root/"pooled_probe_per_seed.csv",index=False)
    episodes=pd.concat(behavior,ignore_index=True)
    require(len(episodes)==60*1800,"Incorrect behavioral evaluation coverage")
    episodes.to_csv(root/"behavior_episodes.csv",index=False)
    metrics=["episode_return","round_success_rate","mean_round_steps","fraction_partner_assigned_faster"]
    per_seed=episodes.groupby(["version","condition","training_seed","d_R","d_B"],as_index=False)[metrics].mean()
    require(len(per_seed)==60*90,"Incomplete behavioral capability coverage")
    per_seed.to_csv(root/"behavior_per_profile_per_seed.csv",index=False)
    per_seed.groupby(["version","condition","d_R","d_B"],as_index=False)[metrics].mean().to_csv(root/"behavior_per_profile_summary.csv",index=False)
    plot_profiles(summary,manifest["profiles"],root)
    require(len(list((root/"per_profile").glob("*.png")))==90,"Missing per-profile figures")
    verify(manifest)
    manifest.update(status="completed",completed_utc=now())
    write(path,manifest)
    write(root/"completion_audit.json",dict(status="passed",completed_utc=now(),policies=60,rnn_policies=45,
        profiles=90,episodes_per_policy=1800,total_evaluation_episodes=108000,new_evaluation_episodes=52800,
        original_profiles_reused=46,new_profiles=44,per_profile_figures=90,policy_retraining=False,
        policy_inputs_unchanged=True,probe_archives_and_metrics_recomputed=True,seed_count=5))
    (root/"README.md").write_text("# Complete capability evaluation\n\n"
        "All 90 unequal ordered red/blue delay pairs from 0..9. Saved direct-input, fixed-t0 "
        "v1_matched/v2/v3 policies, seeds 1–5; no policy retraining. All 60 policies receive 20 episodes per profile. "
        "The original 46-profile rollouts are reused without modification; the other 44 profiles are evaluated "
        "with each policy's original frozen evaluator, layouts, categorical action sampling, and seed 12345.\n\n"
        "The 45 RNN policies use nine cumulative mean GRU features at timesteps 1,50,...,400, including the "
        "terminal step and excluding padding. Independent binary affine 128→2 probes predict red-faster or "
        "blue-faster; the reference full-batch Adam method is unchanged (1000 steps, lr=.01, initialization 10000). "
        "The PCG64 seed-0 repetition split contributes 16 fitting / 4 held-out episodes for every profile "
        "(1440/360 per policy). All 90 profiles occur in both probe partitions. Policy training exposure and "
        "probe fitting exposure are distinct: this is held-out-episode decoding, not held-out-profile probe transfer.\n\n"
        "Per-profile accuracy equals recall for that profile's single faster-goal class. Means and 95% percentile "
        "bootstrap intervals use five policy seeds and 10000 resamples (RNG 30000). Panel colors distinguish "
        "24 diverse-training profiles, 18 new combinations of familiar delays, 42 one-novel-delay profiles, "
        "and six two-novel-delay profiles. The single-partner control saw only (1,4) in policy training.\n\n"
        "The combined PNG/PDF contains all 90 profile groups with three experiment panels each; per_profile/ "
        "contains individual PNGs, also collected in a multipage PDF. CSVs contain all seed-level decoding "
        "and behavioral results. Frozen analysis sources, input hashes, source-file references, probe weights, "
        "predictions, episode partitions, and the final audit are retained.\n")
    print("[completed]",root,flush=True)


def environment(cpu=False):
    result=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1",OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMBA_NUM_THREADS="1",XLA_PYTHON_CLIENT_PREALLOCATE="false")
    if cpu: result.update(JAX_PLATFORMS="cpu",CUDA_VISIBLE_DEVICES="",XLA_FLAGS="")
    else:
        result.pop("JAX_PLATFORMS",None)
        result["XLA_FLAGS"]="--xla_gpu_deterministic_ops=true"
    return result


def validate(path):
    manifest=read(path);verify(manifest);root=Path(manifest["root"])
    checks=[]
    binary,reference=references(manifest)
    rng=np.random.default_rng(91)
    hidden=rng.normal(size=(217,128)).astype(np.float32)
    np.testing.assert_array_equal(prefix_features(hidden,TIMES),reference.prefix_mean_hidden(hidden,TIMES))
    profiles=np.array([(p["red_delay"],p["blue_delay"]) for p in manifest["profiles"]])
    pairs=np.repeat(profiles,20,axis=0);reps=np.tile(np.arange(20),90)
    train,test=binary.masks(reps,pairs,reference.make_probe_split(0))
    require((train.sum(),test.sum())==(1440,360),"Wrong preflight split")
    np.testing.assert_array_equal(np.bincount(binary.relative_classes(pairs)[test]),[180,180])
    checks.append("90 profiles, 16/4 split, balanced 720/720 fit and 180/180 test classes; exact reference prefix features")
    cases=[("v1_matched","rnn_diverse_influence"),("v1_matched","rnn_diverse_noinfluence"),
           ("v2","rnn_single_influence"),("v3","rnn_diverse_influence"),("v3","mlp_diverse_influence")]
    core=str(min(os.sched_getaffinity(0)))
    for version,condition in cases:
        index=next(i for i,p in enumerate(manifest["policies"]) if (p["version"],p["condition"],p["seed"])==(version,condition,1))
        record=manifest["policies"][index]
        print("[preflight]",version,condition,flush=True)
        with (root/"logs"/f"preflight_{version}_{condition}.log").open("w") as log:
            subprocess.run(["taskset","-c",core,manifest["eval_python"],str(root/"source/full_capability_evaluation.py"),
                            "smoke",str(path),"--index",str(index)],env=environment(cpu=True),stdout=log,stderr=subprocess.STDOUT,check=True)
        extract(Path(record["directory"])/"smoke_new_profile.h5",{(5,6)},condition.startswith("rnn"),1,record["allocation_protocol"])
        checks.append(f"saved-policy CPU rollout: {version}/{condition}, unseen (5,6), all 20 rounds and valid terminal")
    manifest["validation"]=dict(passed=True,completed_utc=now(),checks=checks)
    write(path,manifest)
    print("[preflight passed]",len(checks),"checks",flush=True)


def submit(path):
    path=Path(path).resolve();manifest=read(path);verify(manifest)
    require(manifest["validation"].get("passed"),"Preflight validation required")
    root=Path(manifest["root"]);runner=root/"source/full_capability_evaluation.py"
    clean={k:v for k,v in environment().items() if not k.startswith("SLURM_") and k not in ("CUDA_VISIBLE_DEVICES",)}
    for stage in ("collect","analyze","aggregate"):
        if stage in manifest["jobs"]: continue
        python=manifest["eval_python"] if stage=="collect" else manifest["analysis_python"]
        command=[python,"-u",str(runner),stage,str(path)]
        if stage!="aggregate": command += ["--index"]
        lines=["#!/bin/bash","set -euo pipefail","export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1",
            "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1",
            "export XLA_PYTHON_CLIENT_PREALLOCATE=false",f"export MPLCONFIGDIR={shlex.quote(str(root/'mpl_cache'))}"]
        if stage=="collect":
            lines += ["unset JAX_PLATFORMS",'export XLA_FLAGS="--xla_gpu_deterministic_ops=true"']
        else: lines += ["export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=",'export XLA_FLAGS=""']
        lines.append("exec "+shlex.join(command)+(' "${SLURM_ARRAY_TASK_ID}"' if stage!="aggregate" else ""))
        script=root/"jobs"/(stage+".sh");script.write_text("\n".join(lines)+"\n")
        subprocess.run(["bash","-n",str(script)],check=True)
        cmd=["sbatch","--parsable","--account=nlp","--partition=sphinx","--exclude=sphinx9",
             "--cpus-per-task=2","--mem="+("24G" if stage=="collect" else "16G"),
             "--time="+("02:00:00" if stage=="collect" else "01:00:00"),
             "--job-name=cap90_"+stage,"--chdir="+manifest["repo_root"],"--kill-on-invalid-dep=yes",
             f"--output={root}/logs/{stage}_%A_%a.out",f"--error={root}/logs/{stage}_%A_%a.err"]
        if stage=="collect": cmd += ["--array=0-59%6","--gres=gpu:1","--constraint=80G"]
        elif stage=="analyze": cmd += ["--array=0-59%10","--dependency=afterok:"+manifest["jobs"]["collect"]["job_id"]]
        else: cmd += ["--dependency=afterok:"+manifest["jobs"]["analyze"]["job_id"]]
        job_id=subprocess.check_output(cmd+[str(script)],text=True,env=clean).strip().split(";")[0]
        require(job_id.isdigit(),"Invalid Slurm job ID: "+job_id)
        manifest["jobs"][stage]=dict(job_id=job_id,command=cmd+[str(script)],script_sha256=sha(script))
        manifest.update(status="submitted",submitted_utc=now());write(path,manifest)
        print("[submitted]",stage,job_id,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","validate","submit","collect","smoke","analyze","aggregate"))
    parser.add_argument("path",type=Path)
    parser.add_argument("--index",type=int)
    args=parser.parse_args()
    if args.mode=="prepare": print(prepare(args.path),flush=True)
    elif args.mode=="validate": validate(args.path)
    elif args.mode=="submit": submit(args.path)
    elif args.mode in ("collect","smoke"):
        require(args.index is not None,"Policy index is required")
        collect(args.path,args.index,args.mode=="smoke")
    elif args.mode=="analyze":
        require(args.index is not None,"Policy index is required")
        analyze(args.path,args.index)
    else: aggregate(args.path)


if __name__=="__main__":
    main()
