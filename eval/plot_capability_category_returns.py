"""Plot actual 20-round episode returns, averaged within capability category."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from full_capability_evaluation import (
    CATEGORY_COLORS, CATEGORY_LABELS, COLORS, CONDITIONS, LABELS, VERSIONS,
    now, read, require, sha, write,
)
from plot_capability_category_decoding import CATEGORY_COUNTS

ALL_CONDITIONS = CONDITIONS + ("mlp_diverse_influence",)
ALL_LABELS = LABELS + ("Multi-partner MLP",)
ALL_COLORS = COLORS + ("#168e88",)
CATEGORY_TICKS = ("Seen: diverse\ntraining\n24 profiles", "Unseen pair:\nfamiliar delays\n18 profiles",
                  "Unseen: one\nnovel delay\n42 profiles", "Unseen: both\ndelays novel\n6 profiles")
METRICS = ("episode_return", "round_success_rate", "mean_round_steps", "fraction_partner_assigned_faster")


def average_returns(episodes, profiles):
    inventory = pd.DataFrame(profiles)[["red_delay", "blue_delay", "category"]].rename(
        columns={"red_delay":"d_R", "blue_delay":"d_B"})
    require(len(inventory) == 90 and inventory.category.value_counts().to_dict() == CATEGORY_COUNTS,
            "Incorrect capability inventory")
    require(set(zip(inventory.d_R, inventory.d_B)) == {(r,b) for r in range(10) for b in range(10) if r != b},
            "Incomplete capability universe")
    require(len(episodes) == 60*1800, "Require all 108000 actual evaluation episodes")
    keys = ["version", "condition", "training_seed", "d_R", "d_B"]
    require(not episodes.duplicated(keys+["rollout_rep"]).any(), "Duplicate episode observations")
    require(set(episodes.version) == set(VERSIONS) and set(episodes.condition) == set(ALL_CONDITIONS)
            and set(episodes.training_seed) == {1,2,3,4,5} and set(episodes.rollout_rep) == set(range(20)),
            "Wrong evaluation cohort or repetitions")
    require(np.isfinite(episodes[list(METRICS)].to_numpy()).all(), "Nonfinite behavior metrics")
    counts = episodes.groupby(keys).size()
    require(len(counts) == 60*90 and np.all(counts == 20), "Every policy/profile requires 20 episodes")
    joined = episodes.merge(inventory, on=["d_R", "d_B"], validate="many_to_one")
    require(len(joined) == len(episodes), "Episode has no capability category")
    profile = joined.groupby(keys+["category"], as_index=False)[list(METRICS)].mean()
    seed_keys = ["version", "condition", "training_seed", "category"]
    per_seed = profile.groupby(seed_keys, as_index=False)[list(METRICS)].mean()
    per_seed["n_profiles"] = per_seed.category.map(CATEGORY_COUNTS)
    per_seed["n_episodes"] = 20*per_seed.n_profiles
    require(len(per_seed) == 240, "Missing policy/category averages")
    # Equal repetitions make equal-profile and direct episode averaging identical.
    direct = joined.groupby(seed_keys, as_index=False)[list(METRICS)].mean()
    check = per_seed.merge(direct, on=seed_keys, suffixes=("_profile", "_episode"), validate="one_to_one")
    for metric in METRICS:
        np.testing.assert_allclose(check[metric+"_profile"], check[metric+"_episode"], atol=1e-12, rtol=0)
    resampling = np.random.default_rng(30000).integers(0,5,size=(10000,5))
    rows = []
    for group, frame in per_seed.groupby(["version", "condition", "category"], sort=True):
        require(len(frame) == 5 and set(frame.training_seed) == {1,2,3,4,5}, "Require five independent policy seeds")
        frame = frame.sort_values("training_seed")
        values = frame.episode_return.to_numpy()
        bootstrap = values[resampling].mean(axis=1)
        rows.append(dict(zip(["version", "condition", "category"], group),
            episode_return_mean=float(values.mean()), episode_return_ci_low=float(np.quantile(bootstrap,.025)),
            episode_return_ci_high=float(np.quantile(bootstrap,.975)), seed_count=5,
            n_profiles=int(frame.n_profiles.iloc[0]), episodes_per_seed=int(frame.n_episodes.iloc[0]),
            **{metric+"_mean":float(frame[metric].mean()) for metric in METRICS[1:]}))
    summary = pd.DataFrame(rows)
    require(len(summary) == 48, "Missing experiment/condition/category summaries")
    return profile, per_seed, summary, joined


def plots(summary, per_seed, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = list(CATEGORY_COUNTS)

    def panel(ax, version, include_mlp):
        count = 4 if include_mlp else 3
        offsets = np.linspace(-.27,.27,count)
        x = np.arange(4)
        for i, category in enumerate(categories):
            ax.axvspan(i-.48,i+.48,color=CATEGORY_COLORS[category],zorder=0)
        for offset, condition, label, color in zip(offsets,ALL_CONDITIONS[:count],ALL_LABELS[:count],ALL_COLORS[:count]):
            frame = summary[(summary.version == version) & (summary.condition == condition)].set_index("category").loc[categories]
            y = frame.episode_return_mean.to_numpy()
            error = np.vstack([y-frame.episode_return_ci_low.to_numpy(), frame.episode_return_ci_high.to_numpy()-y])
            ax.errorbar(x+offset,y,yerr=error,fmt="o",color=color,ms=6,capsize=3,elinewidth=1.5,
                        markeredgecolor="white",markeredgewidth=.5,label=label,zorder=4)
            for index, category in enumerate(categories):
                seeds = per_seed[(per_seed.version == version) & (per_seed.condition == condition)
                                 & (per_seed.category == category)].sort_values("training_seed")
                ax.scatter(np.full(5,index+offset)+np.linspace(-.025,.025,5),seeds.episode_return,
                           s=10,color=color,alpha=.35,zorder=3)
        ax.set_title(version,fontsize=12)
        ax.set(xticks=x,xticklabels=CATEGORY_TICKS,xlim=(-.6,3.6),ylim=(0,20))
        ax.set_yticks(np.arange(0,21,2));ax.tick_params(axis="x",labelsize=9,length=0,pad=8)
        ax.tick_params(axis="y",labelsize=9);ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",color="gray",alpha=.15,linewidth=.6)
        ax.set_ylabel("Mean evaluation return per episode\n(sum across 20 rounds)")

    def finish(fig, axes, stem):
        handles, labels = np.ravel(axes)[0].get_legend_handles_labels()
        fig.suptitle("Actual evaluation episode returns averaged within capability category\n"
                     "20 episodes per profile per policy; five policy seeds; error bars: 95% bootstrap CI across seeds",
                     fontsize=12,y=.99)
        fig.legend(handles,labels,loc="lower center",ncol=4,frameon=False,fontsize=9,bbox_to_anchor=(.5,.065))
        fig.text(.5,.025,"Category membership refers to the diverse-training pool; the single-partner policy trained only on (1,4).",
                 ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.13,1,.88),w_pad=1.5)
        for extension in ("png","pdf"):fig.savefig(out/f"{stem}.{extension}",dpi=180)

    for include_mlp in (True,False):
        fig, axes = plt.subplots(1,3,figsize=(18,6.3),sharey=True)
        for ax, version in zip(axes,VERSIONS):panel(ax,version,include_mlp)
        suffix = "" if include_mlp else "_rnn_only"
        finish(fig,axes,"episode_return_category_averages_all_seeds"+suffix)
        plt.close(fig)
    for version in VERSIONS:
        fig, ax = plt.subplots(figsize=(9,6.3))
        panel(ax,version,True)
        finish(fig,[ax],f"episode_return_category_averages_{version}")
        plt.close(fig)


def diagnostics(root, joined, out, profiles):
    inventory = pd.DataFrame(profiles).rename(columns={"red_delay":"d_R","blue_delay":"d_B"})
    inventory["delay_gap"] = abs(inventory.d_R-inventory.d_B)
    inventory["movement_rate_gap"] = abs(1/(inventory.d_R+1)-1/(inventory.d_B+1))
    inventory.groupby("category")[["delay_gap","movement_rate_gap"]].agg(["mean","median"]).to_csv(out/"profile_contrasts.csv")
    familiar = joined[joined.category == "unseen_pair_of_familiar_delays"].copy()
    familiar["subgroup"] = np.where(familiar.d_R.isin([1,2,3]),"both_fast","both_slow")
    sub = familiar.groupby(["version","condition","training_seed","subgroup"],as_index=False)[list(METRICS)].mean()
    sub.to_csv(out/"familiar_delay_subgroups_per_seed.csv",index=False)
    sub.groupby(["version","condition","subgroup"],as_index=False)[list(METRICS)].mean().to_csv(out/"familiar_delay_subgroups_summary.csv",index=False)
    decode = pd.read_csv(root/"profile_probe_per_seed.csv")
    decode = decode[(decode.cutoff == 400) & (decode.condition == "rnn_diverse_influence")].merge(
        inventory[["d_R","d_B","category","delay_gap"]],on=["d_R","d_B"],validate="many_to_one")
    decode.groupby(["version","category","delay_gap"],as_index=False).agg(
        accuracy=("accuracy","mean"),profile_seed_count=("accuracy","size")).to_csv(out/"decoding_by_delay_gap.csv",index=False)


def run(root):
    root = Path(root).resolve();out = root/"category_averages/episode_returns"
    require(read(root/"completion_audit.json")["status"] == "passed","Source evaluation audit must pass")
    manifest = read(root/"analysis_run.json")
    path = root/"behavior_episodes.csv"
    profile, per_seed, summary, joined = average_returns(pd.read_csv(path),manifest["profiles"])
    out.mkdir(parents=True,exist_ok=True)
    profile.to_csv(out/"episode_return_per_profile_per_seed.csv",index=False)
    per_seed.to_csv(out/"episode_return_category_per_seed.csv",index=False)
    summary.to_csv(out/"episode_return_category_summary.csv",index=False)
    # Independently match the pre-existing per-profile evaluation summary.
    keys = ["version","condition","training_seed","d_R","d_B"]
    prior = pd.read_csv(root/"behavior_per_profile_per_seed.csv")
    comparison = profile.merge(prior,on=keys,validate="one_to_one",suffixes=("_new","_prior"))
    require(len(comparison) == 5400,"Incomplete profile-summary cross-check")
    for metric in METRICS:
        np.testing.assert_allclose(comparison[metric+"_new"],comparison[metric+"_prior"],atol=1e-12,rtol=0)
    plots(summary,per_seed,out)
    diagnostics(root,joined,out,manifest["profiles"])
    source = out/"source";source.mkdir(exist_ok=True)
    for name in (Path(__file__).name,"full_capability_evaluation.py","plot_capability_category_decoding.py"):
        src = Path(__file__).with_name(name);target = source/name
        if src.resolve() != target.resolve():shutil.copyfile(src,target)
    write(out/"methods.json",dict(status="completed",completed_utc=now(),input=str(path),input_sha256=sha(path),
        evaluation_episodes=108000,episodes_per_profile=20,profiles=90,category_counts=CATEGORY_COUNTS,
        seed_count=5,policy_count=60,summary_rows=48,profile_summary_checks=5400,
        metric="Actual episode reward sum over all 20 rounds, already terminal-inclusive and padding-excluded.",
        averaging="Mean of all 20 episodes within each profile; equal-profile mean within category and policy seed; equal mean across five policy seeds.",
        uncertainty="95% percentile bootstrap over five independent policy seed means, 10000 resamples, PCG64 seed 30000.",
        includes_all_four_conditions=True,new_evaluations=False,policy_retraining=False,
        source_sha256={p.name:sha(p) for p in source.glob("*.py")}))
    print(summary.pivot(index=["version","condition"],columns="category",values="episode_return_mean").round(3).to_string())
    print(out/"episode_return_category_averages_all_seeds.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    run(parser.parse_args().root)
