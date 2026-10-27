"""Audited aggregation and figures for single-state faster-goal decoding."""
from pathlib import Path

import numpy as np
import pandas as pd

import full_capability_evaluation as full
from analyze_instantaneous_decoding import STEPS, bootstrap_seeds, profile_metrics


def plot_configuration(conditions=None, labels=None):
    conditions = tuple(full.CONDITIONS if conditions is None else conditions)
    labels = tuple(full.LABELS[full.CONDITIONS.index(c)] for c in conditions) if labels is None else tuple(labels)
    colors = tuple(full.COLORS[full.CONDITIONS.index(c)] for c in conditions)
    full.require(len(labels)==len(conditions), "Mismatched plot labels")
    return conditions,labels,colors


def draw(axes, means, lows, highs, profile, compact=False, conditions=None, labels=None):
    conditions,labels,colors = plot_configuration(conditions,labels)
    exposure = full.CATEGORY_LABELS[profile["category"]]
    red, blue = profile["red_delay"], profile["blue_delay"]
    if (red, blue) == (1, 4): exposure += "; also single-partner training"
    for v, (ax, version) in enumerate(zip(axes, full.VERSIONS)):
        for c, (color, label) in enumerate(zip(colors, labels)):
            ax.plot(STEPS, means[v, c], color=color, lw=.8 if compact else 1.5, label=label)
            ax.fill_between(STEPS, lows[v, c], highs[v, c], color=color, alpha=.14, linewidth=0)
        ax.axhline(.5, color="gray", ls="--", lw=.6 if compact else 1)
        ax.set(xlim=(0,400), ylim=(0,1.02), yticks=[0,.5,1])
        ax.set_xticks([0,200,400] if compact else [0,100,200,300,400])
        ax.set_facecolor(full.CATEGORY_COLORS[profile["category"]])
        ax.spines[["top","right"]].set_visible(False)
        ax.set_title(f"({red},{blue}) · {version}" if compact else version, fontsize=6.4 if compact else 11, pad=3)
        ax.tick_params(labelsize=5 if compact else 9, length=2 if compact else 3)
        ax.set_xlabel("Environment step t", fontsize=5.5 if compact else 10, labelpad=1)
    axes[0].set_ylabel("Accuracy", fontsize=5.5 if compact else 10, labelpad=1)
    if compact:
        axes[1].text(.5,1.22,exposure,transform=axes[1].transAxes,ha="center",va="bottom",fontsize=5.4)
    return exposure


def plot_profiles(means, lows, highs, profiles, root, conditions=None, labels=None, method_note=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    conditions,labels,colors = plot_configuration(conditions,labels)
    root = Path(root); individuals = root / "per_profile"; individuals.mkdir(exist_ok=True)
    legend = [Line2D([0],[0],color=color,lw=2,label=label) for color,label in zip(colors,labels)]
    legend += [Line2D([0],[0],color="gray",ls="--",label="Random guess (1/2)")]
    with PdfPages(root / "relative_speed_timestep_per_profile.pdf") as pdf:
        for i, profile in enumerate(profiles):
            fig, axes = plt.subplots(1,3,figsize=(15,4.8),sharey=True)
            exposure = draw(axes, means[:,:,:,i], lows[:,:,:,i], highs[:,:,:,i], profile, conditions=conditions, labels=labels)
            red, blue = profile["red_delay"], profile["blue_delay"]
            fig.suptitle(f"Partner ({red},{blue}) [red delay, blue delay] · {exposure}\n"+
                ("Current observation x_t decoding at every step; five seeds; shading: 95% bootstrap CI" if method_note is None else method_note),fontsize=11)
            axes[-1].legend(handles=legend,frameon=False,fontsize=8,loc="lower right")
            fig.tight_layout(rect=(0,0,1,.89))
            fig.savefig(individuals / f"relative_speed_timestep_dR{red}_dB{blue}.png",dpi=160)
            pdf.savefig(fig); plt.close(fig)
    rows = (len(profiles)+2)//3; height = rows*2.3+1.8
    fig = plt.figure(figsize=(27,height))
    grid = fig.add_gridspec(rows,9,left=.025,right=.995,bottom=.3/height,top=1-2.0/height,wspace=.32,hspace=.95)
    for i, profile in enumerate(profiles):
        r,c = divmod(i,3); axes = [fig.add_subplot(grid[r,c*3+j]) for j in range(3)]
        draw(axes, means[:,:,:,i], lows[:,:,:,i], highs[:,:,:,i], profile, compact=True, conditions=conditions, labels=labels)
    fig.suptitle("Current observation x_t faster-goal decoding · v1_matched, v2, v3 · direct input, fixed t0\n"
        "Current observation x_t at every environment step t=0–400; no temporal averaging; five policy seeds; shading: 95% bootstrap CI.\n"+
        ("t=0: initial observation. Probe split: 16 fit / 4 test episodes per profile. Single-partner training saw only (1,4)." if method_note is None else method_note),
        fontsize=12,y=1-.05/height)
    legend += [Patch(facecolor=full.CATEGORY_COLORS[key],label=full.CATEGORY_LABELS[key]) for key in full.CATEGORY_COLORS]
    fig.legend(handles=legend,loc="upper center",bbox_to_anchor=(.5,1-.8/height),ncol=4,frameon=False,fontsize=8)
    for extension in ("png","pdf"):
        fig.savefig(root / f"relative_speed_timestep_all_profiles_all_seeds.{extension}",dpi=140)
    plt.close(fig)


def plot_categories(accuracy, profiles, root, conditions=None, labels=None, method_note=None):
    import matplotlib.pyplot as plt
    conditions,labels,colors = plot_configuration(conditions,labels)
    categories = tuple(full.CATEGORY_COLORS)
    seed_means = np.stack([accuracy[...,np.array([p["category"]==cat for p in profiles])].mean(axis=-1)
                           for cat in categories],axis=-1)
    means,lows,highs = bootstrap_seeds(seed_means.transpose(2,0,1,3,4))
    rows = []
    for v,version in enumerate(full.VERSIONS):
        for c,condition in enumerate(conditions):
            for k,category in enumerate(categories):
                for t in STEPS:
                    rows.append(dict(version=version,condition=condition,category=category,env_step=int(t),
                        accuracy_mean=means[v,c,t,k],accuracy_ci_low=lows[v,c,t,k],accuracy_ci_high=highs[v,c,t,k],seed_count=5))
    root = Path(root) / "category_averages"; root.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "category_probe_summary.csv",index=False)
    np.savez_compressed(root / "category_probe_per_seed.npz",accuracy=seed_means,env_steps=STEPS,
        versions=full.VERSIONS,conditions=conditions,seeds=np.arange(1,6),categories=categories)
    for end, suffix in ((400,""),(60,"_first60")):
        fig,axes = plt.subplots(4,3,figsize=(16,12),sharex=True,sharey=True)
        for k,category in enumerate(categories):
            for v,version in enumerate(full.VERSIONS):
                ax = axes[k,v]
                for c,(color,label) in enumerate(zip(colors,labels)):
                    ax.plot(STEPS,means[v,c,:,k],color=color,lw=1.3,label=label)
                    ax.fill_between(STEPS,lows[v,c,:,k],highs[v,c,:,k],color=color,alpha=.17,linewidth=0)
                ax.axhline(.5,color="gray",ls="--",lw=.8)
                ax.set_facecolor(full.CATEGORY_COLORS[category]); ax.spines[["top","right"]].set_visible(False)
                ax.set(xlim=(0,end),ylim=(.2,1.02))
                ax.set_title(version+" · "+full.CATEGORY_LABELS[category],fontsize=10)
                if v==0: ax.set_ylabel("Decoding accuracy")
                if k==3: ax.set_xlabel("Environment step t (0 = initial observation)")
        fig.suptitle("Current observation x_t decoding at every step · category averages\n"+
            ("Equal profile weights within each policy seed; mean and 95% bootstrap CI across five seeds" if method_note is None else method_note),fontsize=13)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.945),ncol=3,frameon=False)
        fig.tight_layout(rect=(0,0,1,.91))
        for extension in ("png","pdf"):
            fig.savefig(root / f"relative_speed_timestep_category_averages_all_seeds{suffix}.{extension}",dpi=160)
        plt.close(fig)
        for v,version in enumerate(full.VERSIONS):
            fig,axes = plt.subplots(1,4,figsize=(20,4.6),sharey=True)
            for k,category in enumerate(categories):
                for c,(color,label) in enumerate(zip(colors,labels)):
                    axes[k].plot(STEPS,means[v,c,:,k],color=color,lw=1.3,label=label)
                    axes[k].fill_between(STEPS,lows[v,c,:,k],highs[v,c,:,k],color=color,alpha=.17,linewidth=0)
                axes[k].set_facecolor(full.CATEGORY_COLORS[category]); axes[k].axhline(.5,color="gray",ls="--",lw=.8)
                axes[k].set(xlim=(0,end),ylim=(.2,1.02),title=full.CATEGORY_LABELS[category],xlabel="Environment step t")
                axes[k].spines[["top","right"]].set_visible(False)
            axes[0].set_ylabel("Decoding accuracy")
            fig.suptitle(version+" · Current observation x_t decoding; every step; five seeds; 95% bootstrap CI"+
                         ("" if method_note is None else "\n"+method_note))
            fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.92),ncol=3,frameon=False)
            fig.tight_layout(rect=(0,0,1,.83))
            fig.savefig(root / f"{version}_category_averages{suffix}.png",dpi=160); plt.close(fig)


def aggregate(path):
    run = full.read(path); root = Path(run["root"]); manifest = full.read(run["input_manifest"])
    full.verify(manifest)
    for name,digest in run["source_sha256"].items():
        full.require(full.sha(root / "source" / name)==digest,"Frozen source changed")
    profiles = manifest["profiles"]
    pairs = np.array([(p["red_delay"],p["blue_delay"]) for p in profiles])
    accuracy = np.full((3,3,5,len(STEPS),90),np.nan)
    active = np.zeros_like(accuracy,dtype=np.uint8)
    anchors = []; completions = []
    for record in run["policies"]:
        directory = Path(record["instant_directory"]); done = full.read(directory / "completion.json")
        archive = directory / "instantaneous_probes.npz"
        full.require(done["status"]=="completed" and full.sha(archive)==done["archive_sha256"],"Incomplete or changed probe archive")
        with np.load(archive) as data:
            np.testing.assert_array_equal(data["env_steps"],STEPS)
            np.testing.assert_array_equal(data["profiles"],pairs)
            np.testing.assert_array_equal(data["truth"],(data["pairs"][:,1]<data["pairs"][:,0]).astype(int))
            np.testing.assert_array_equal(data["predictions"],data["probabilities"].argmax(axis=-1))
            full.require(not np.any(data["train"]&data["test"]) and np.all(data["train"]|data["test"]),"Wrong split")
            with np.load(Path(record["directory"]) / "probe_models.npz") as original:
                for key in ("pairs","repetitions","train","test","truth"):
                    np.testing.assert_array_equal(data[key],original[key])
            recomputed = profile_metrics(data["predictions"],data["probabilities"],data["pairs"],data["truth"],data["test"],data["lengths"])
            for key,value in recomputed.items(): np.testing.assert_array_equal(data[key],value)
            v = full.VERSIONS.index(record["version"]); c = full.CONDITIONS.index(record["condition"]); s = record["seed"]-1
            full.require(np.isnan(accuracy[v,c,s]).all(),"Duplicate policy")
            accuracy[v,c,s] = data["accuracy"]; active[v,c,s] = data["active_test_episodes"]
        anchors.append(done["anchor_max_probability_error"]); completions.append(done)
    full.require(np.isfinite(accuracy).all(),"Incomplete policy coverage")
    means,lows,highs = bootstrap_seeds(accuracy.transpose(2,0,1,3,4))
    np.savez_compressed(root / "profile_probe_summary.npz",accuracy_mean=means,accuracy_ci_low=lows,accuracy_ci_high=highs,
        accuracy_per_seed=accuracy,active_test_episodes=active,env_steps=STEPS,profiles=pairs,
        versions=full.VERSIONS,conditions=full.CONDITIONS,seeds=np.arange(1,6))
    rows = []
    for v,version in enumerate(full.VERSIONS):
        for c,condition in enumerate(full.CONDITIONS):
            for i,profile in enumerate(profiles):
                for t in STEPS:
                    rows.append(dict(version=version,condition=condition,d_R=profile["red_delay"],d_B=profile["blue_delay"],
                        category=profile["category"],env_step=int(t),accuracy_mean=means[v,c,t,i],
                        accuracy_ci_low=lows[v,c,t,i],accuracy_ci_high=highs[v,c,t,i],seed_count=5,
                        active_test_episodes=int(active[v,c,:,t,i].sum())))
    pd.DataFrame(rows).to_csv(root / "profile_probe_summary.csv",index=False)
    print("[draw all 90 profiles]",flush=True)
    plot_profiles(means,lows,highs,profiles,root)
    print("[draw category means]",flush=True)
    plot_categories(accuracy,profiles,root)
    full.require(len(list((root / "per_profile").glob("*.png")))==90,"Missing figures")
    full.verify(manifest)
    audit = dict(status="passed",completed_utc=full.now(),policies=45,profiles=90,policy_seed_count=5,
        timepoints=401,first_env_step=0,last_env_step=400,total_independent_probes=45*401,
        per_profile_figures=90,original_split_reproduced=True,metrics_recomputed=True,
        previous_instantaneous_anchors=12,anchor_max_probability_error=max(anchors),
        anchor_prediction_disagreements=sum(d["anchor_prediction_disagreements"] for d in completions),
        carried_terminal_test_episode_points=int((4-active.astype(int)).sum()),
        policy_retraining=False,new_rollouts=False,feature="Single recurrent hidden state h_t; no temporal averaging")
    full.write(root / "completion_audit.json",audit)
    run.update(status="completed",completed_utc=full.now()); full.write(path,run)
    (root / "README.md").write_text("# Single-state decoding at every environment step\n\n"
        "These plots decode the single post-observation, pre-action GRU state Current observation x_t at every environment step "
        "t=0..400 inclusive (401 points). t=0 is the initial observation, before any environment transition. "
        "There is no explicit averaging over time; a recurrent state can still encode its observation history. "
        "The original parent-directory plots used cumulative averages and remain available.\n\n"
        "All 90 unequal red/blue delay pairs 0..9; v1_matched, v2, v3; three RNN conditions; five policy seeds. "
        "Only existing saved rollouts are read: no new environment evaluation or policy training. Each timestep "
        "gets an independent affine 128->2 faster-goal probe using the frozen reference procedure: full-batch "
        "class-balanced Adam, 1000 updates, lr=.01, initialization 10000. The original seed-0 repetition split "
        "is preserved: 16 fit / 4 held-out episodes per profile (1440/360 per policy). All profiles appear in "
        "probe fitting; held-out policy-training capabilities and held-out probe episodes are different.\n\n"
        "Episode padding is excluded. After termination the final valid hidden state is carried forward, "
        "retaining the same probe sample and episode partition. Active test episode counts are in the summary "
        "CSV/NPZ. Profile accuracy equals recall for that profile's single faster-goal class. Five policy-seed "
        "means and 95% percentile bootstrap intervals use 10000 resamples (RNG30000). Category averages give "
        "each profile equal weight within each policy seed, then average seeds and bootstrap seeds. No smoothing.\n\n"
        "Panel colors mark diverse-training exposure: 24 green training profiles, 18 blue unseen combinations "
        "of familiar delays, 42 yellow one-novel-delay profiles, six pink both-novel-delay profiles. Only (1,4) "
        "was seen by the single-partner policy. Frozen source, fitted readouts, probabilities, labels, split "
        "masks, and an integrity audit are retained. Twelve earlier instantaneous control probes are reproduced "
        "within numerical tolerance.\n")
    print("[completed]",root,flush=True)
