"""Compare plotted behavioral curves with instantaneous decoding, t=0..30.

All metrics use the same replay-validated held-out episodes. Report centered
within-panel associations, first differences, five-step changes, seed bootstrap
uncertainty, and descriptive positive-lag sensitivity. No iid timestep p-values.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import full_capability_evaluation as full
from analyze_decoding_behavior import CATEGORIES, history
from analyze_replayed_motion import load_motion, motion_events

METRICS = ("four_waits_after_move_seen", "second_partner_move_seen", "first_partner_move_seen",
           "round_feedback_seen", "both_active_goals", "any_goal_switch", "current_assignment_faster")
LABELS = ("Four off-goal waits after a move", "At least two partner moves", "First partner move",
          "First round finished", "Both goal colors pursued", "Within-round goal switch", "Partner assigned faster color")
MODES = (("levels", 0), ("one_step_changes", 1), ("five_step_changes", 5))


def centered_correlation(x, y, panel_axes=(-2, -1)):
    """Correlate after centering time separately in each panel.

    x,y end in [panel,time]; leading axes (e.g. bootstrap, metric) broadcast.
    Flat predictors contribute zero covariance without dropping their panels.
    """
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    numerator = np.sum(x * y, axis=panel_axes)
    denominator = np.sqrt(np.sum(x*x, axis=panel_axes) * np.sum(y*y, axis=panel_axes))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float), where=denominator > 1e-14)


def changes(values, width):
    return values if width == 0 else values[..., width:, :] - values[..., :-width, :]


def correlations(values, width=0, lag=0):
    """values: [...,version,category,time,target+metrics]. Positive lag: x leads y."""
    d = changes(values, width)
    x = np.moveaxis(d[..., 1:], -1, -4)  # [...,metric,version,category,time]
    y = d[..., 0][..., None, :, :, :]
    if lag:
        x = x[..., :-lag]
        y = y[..., lag:]
    shape = x.shape
    x = x.reshape(shape[:-3] + (shape[-3]*shape[-2], shape[-1]))
    y = y.reshape(y.shape[:-3] + (y.shape[-3]*y.shape[-2], y.shape[-1]))
    return centered_correlation(x, y)


def event_behavior(data):
    goal = data["partner_goal"]
    active = ~data["is_t0"] & np.isin(goal, [1,2])
    red = history(active & (goal == 1)) > 0
    blue = history(active & (goal == 2)) > 0
    previous_goal = np.c_[np.zeros(len(goal), int), goal[:, :-1]]
    previous_active = np.c_[np.zeros(len(goal), bool), active[:, :-1]]
    same_round = np.c_[np.zeros(len(goal), bool), data["round_idx"][:,1:] == data["round_idx"][:,:-1]]
    switch = active & previous_active & same_round & (previous_goal != goal)
    faster_assignment = np.where(data["pairs"][:,0] < data["pairs"][:,1], 2, 1)
    return dict(both_active_goals=red & blue, any_goal_switch=history(switch)>0,
                current_assignment_faster=data["partner_assignment"] == faster_assignment[:,None])


def load_matched(run, behavior_root):
    original = full.read(run["input_manifest"])
    lookup = {(p["red_delay"],p["blue_delay"]):p["category"] for p in original["profiles"]}
    values = np.full((3,5,3,31,8), np.nan, dtype=float)
    rows = []; audits = []
    for record in run["policies"]:
        if record["condition"] != "rnn_diverse_influence": continue
        print("[matched curves]", full.identity(record), flush=True)
        data, replay_audit = load_motion(record, behavior_root)
        # These additional saved behavior fields were also checked by replay.
        parts = []
        directory = behavior_root / "anchored_replay" / record["version"] / f"{record['condition']}_seed{record['seed']}"
        for name in ("train", "test", "additional"):
            with np.load(directory / (name+"_replay.npz")) as saved:
                parts.append({k:saved[k].copy() for k in ("pairs", "repetitions", "round_idx", "partner_assignment")})
        extra = {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        order = np.lexsort((extra["repetitions"], extra["pairs"][:,1], extra["pairs"][:,0]))
        extra = {k:v[order] for k,v in extra.items()}
        for k in ("pairs", "repetitions"): np.testing.assert_array_equal(data[k], extra[k])
        data.update(extra)
        archive = Path(record["probe_directory"]) / "training_capability_probes.npz"
        completion = full.read(archive.with_name("completion.json"))
        full.require(full.sha(archive) == completion["archive_sha256"], "Changed probe archive")
        with np.load(archive) as saved:
            test = saved["test"]
            np.testing.assert_array_equal(data["pairs"], saved["pairs"][test])
            np.testing.assert_array_equal(data["repetitions"], saved["repetitions"][test])
            correct = (saved["predictions"][:31,test] == saved["truth"][test][None,:]).T
        valid = data["verified_prefix"][:,30]
        event = motion_events(data); event.update(event_behavior(data))
        labels = np.array([lookup[tuple(pair)] for pair in data["pairs"]])
        v = full.VERSIONS.index(record["version"]); s = record["seed"]-1
        for c, category in enumerate(CATEGORIES):
            selected = valid & (labels == category)
            pairs, codes = np.unique(data["pairs"][selected], axis=0, return_inverse=True)
            expected_profiles = sum(p["category"] == category for p in original["profiles"])
            full.require(len(pairs) == expected_profiles, "A capability profile has no validated test episodes")
            metrics = [correct] + [event[k] for k in METRICS]
            for k, metric in enumerate(metrics):
                profile_means = np.stack([metric[selected][codes == p,:31].mean(axis=0) for p in range(len(pairs))])
                values[v,s,c,:,k] = profile_means.mean(axis=0)
            for t in range(31):
                row = dict(**full.identity(record), category=category, env_step=t,
                    n_profiles=len(pairs), n_episodes=int(selected.sum()), accuracy=values[v,s,c,t,0])
                row.update({k:values[v,s,c,t,i+1] for i,k in enumerate(METRICS)})
                rows.append(row)
        audits.append(dict(**full.identity(record), verified_test_episodes=int(valid.sum()), total_test_episodes=len(valid),
            probe_archive_sha256=completion["archive_sha256"]))
    full.require(np.isfinite(values).all(), "Missing matched curves")
    table = pd.DataFrame(rows)
    # Independently reproduce every existing motion metric on the fixed cohort.
    keys = ["version", "condition", "training_seed", "category", "env_step"]
    prior = pd.read_csv(behavior_root / "motion_per_seed.csv")
    merged = table.merge(prior, on=keys, suffixes=("", "_prior"), validate="one_to_one")
    full.require(len(merged) == len(table), "Missing original motion rows")
    for k in ("accuracy", "four_waits_after_move_seen", "second_partner_move_seen", "first_partner_move_seen", "round_feedback_seen"):
        np.testing.assert_allclose(merged[k], merged[k+"_prior"], atol=1e-14, rtol=0)
    return values, table, audits


def summarize(values, out):
    # Original estimand is correlation of the five-seed MEAN plotted curves.
    means = values.mean(axis=1)
    point = np.stack([correlations(means,width) for _,width in MODES])
    # Correlation of a noisy seed mean is nonlinear. Ordinary resampling can
    # reduce the number of unique seeds and bias its percentile distribution
    # downward. Use stratified delete-one jackknife SEs on Fisher z for the
    # displayed interval, and retain bootstrap distributions as sensitivity.
    jack = np.empty((3,5,len(MODES),len(METRICS)))
    for v in range(3):
        for s in range(5):
            leave_one = means.copy()
            leave_one[v] = np.delete(values[v],s,axis=0).mean(axis=0)
            jack[v,s] = np.stack([correlations(leave_one,width) for _,width in MODES])
    zjack = np.arctanh(np.clip(jack,-.999999,.999999))
    zvariance = (4/5 * ((zjack-zjack.mean(axis=1,keepdims=True))**2).sum(axis=1)).sum(axis=0)
    critical = float(student_t.ppf(.975,12))  # sum of within-experiment seed df: 3*(5-1)
    zpoint = np.arctanh(point)
    bounds = np.tanh(np.stack([zpoint-critical*np.sqrt(zvariance),zpoint+critical*np.sqrt(zvariance)]))
    draws = np.random.default_rng(30000).integers(0,5,size=(10000,3,5))
    boots = np.empty((10000,len(MODES),len(METRICS)))
    for start in range(0,10000,250):
        draw = draws[start:start+250]
        sample = np.stack([values[v,draw[:,v]].mean(axis=1) for v in range(3)],axis=1)
        for m,(_,width) in enumerate(MODES): boots[start:start+len(draw),m] = correlations(sample,width)
    bootstrap_bounds = np.quantile(boots,[.025,.975],axis=0)
    summary = []
    for m,(mode,width) in enumerate(MODES):
        for k,metric in enumerate(METRICS):
            rank = np.argsort(-boots[:,m],axis=1)
            summary.append(dict(mode=mode, change_steps=width, metric=metric, label=LABELS[k],
                correlation=point[m,k], ci_low=bounds[0,m,k], ci_high=bounds[1,m,k],
                interval_method="approximate Fisher-z stratified delete-one-policy-seed jackknife, t12",
                bootstrap_percentile_low=bootstrap_bounds[0,m,k], bootstrap_percentile_high=bootstrap_bounds[1,m,k],
                bootstrap_fraction_ranked_first=float((rank[:,0]==k).mean()), n_panels=9,
                timesteps_per_panel=31-width, seed_bootstrap_draws=10000))
    summary = pd.DataFrame(summary)
    summary.to_csv(out/"correlation_summary.csv",index=False)
    panel = []
    for m,(mode,width) in enumerate(MODES):
        for v,version in enumerate(full.VERSIONS):
            for c,category in enumerate(CATEGORIES):
                d = changes(means[v,c],width)
                for k,metric in enumerate(METRICS):
                    x,y = d[:,k+1],d[:,0]
                    flat = np.std(x)<1e-12
                    panel.append(dict(version=version, category=category, mode=mode, metric=metric,
                        correlation=np.nan if flat else float(np.corrcoef(x,y)[0,1]), flat_behavior_curve=flat))
    panel = pd.DataFrame(panel); panel.to_csv(out/"panel_correlations.csv",index=False)
    per_seed = []
    for v,version in enumerate(full.VERSIONS):
        for s in range(5):
            for m,(mode,width) in enumerate(MODES):
                r = correlations(values[v:v+1,s],width)
                for k,metric in enumerate(METRICS):
                    per_seed.append(dict(version=version,training_seed=s+1,mode=mode,metric=metric,correlation=r[k]))
    pd.DataFrame(per_seed).to_csv(out/"policy_seed_correlations.csv",index=False)
    differences = []
    for m,(mode,_) in enumerate(MODES):
        for k,metric in enumerate(METRICS[1:],start=1):
            diff = boots[:,m,0]-boots[:,m,k]
            jdiff = jack[:,:,m,0]-jack[:,:,m,k]
            jvariance = float((4/5 * ((jdiff-jdiff.mean(axis=1,keepdims=True))**2).sum(axis=1)).sum())
            estimate = point[m,0]-point[m,k]
            differences.append(dict(mode=mode, comparison="four_waits_minus_"+metric,
                correlation_difference=estimate,ci_low=estimate-critical*np.sqrt(jvariance),
                ci_high=estimate+critical*np.sqrt(jvariance),interval_method="approximate stratified delete-one-policy-seed jackknife, t12",
                bootstrap_percentile_low=float(np.quantile(diff,.025)),bootstrap_percentile_high=float(np.quantile(diff,.975)),
                bootstrap_fraction_above_zero=float((diff>0).mean())))
    pd.DataFrame(differences).to_csv(out/"paired_metric_differences.csv",index=False)
    lag_rows = []
    for m,(mode,width) in enumerate(MODES[1:],start=1):
        for lag in range(6):
            r = correlations(means,width,lag)
            for k,metric in enumerate(METRICS):
                lag_rows.append(dict(mode=mode,behavior_leads_decoding_steps=lag,metric=metric,correlation=r[k]))
    pd.DataFrame(lag_rows).to_csv(out/"lag_sensitivity.csv",index=False)
    np.savez_compressed(out/"correlation_bootstrap.npz", estimates=point,bootstrap=boots,jackknife=jack,
                        metrics=np.array(METRICS),modes=np.array([m for m,_ in MODES]))
    return summary,panel,pd.DataFrame(differences)


def plot(summary,panel,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = summary[summary["mode"].eq("five_step_changes")].sort_values("correlation",ascending=False).metric.tolist()
    labels = dict(zip(METRICS,LABELS))
    colors = dict(zip(METRICS,("#168e88","#b36b00","#777777","#222222","#3779bf","#60b5b0","#e67e00")))
    fig,axes = plt.subplots(1,3,figsize=(15,5.4),sharey=True)
    titles = ("Overall curve levels", "One-step changes", "Five-step changes")
    for ax,(mode,_),title in zip(axes,MODES,titles):
        g = summary[summary["mode"].eq(mode)].set_index("metric").loc[order]
        y = np.arange(len(order))
        for i,(metric,row) in enumerate(g.iterrows()):
            ax.plot([row.ci_low,row.ci_high],[i,i],color=colors[metric],lw=2)
            ax.plot(row.correlation,i,"o",color=colors[metric],ms=7)
            ax.text(.99,i,f"{row.correlation:.2f}",ha="right",va="center",fontsize=9)
        ax.axvline(0,color="gray",lw=.8)
        ax.set(title=title,xlim=(-.55,1.04),xlabel="Within-panel Pearson correlation",yticks=y)
        ax.set_yticklabels([labels[k] for k in order]); ax.invert_yaxis()
        ax.spines[["top","right"]].set_visible(False)
    # Shared y axes must be inverted exactly once.
    axes[0].set_ylim(len(order)-.5,-.5)
    fig.suptitle("Which behavioral changes track decoding?\nNine experiment/category panels · same validated episodes · steps 0–30",fontsize=13)
    fig.text(.5,.018,"Intervals: approximate 95% Fisher-z jackknife over policy seeds, stratified by experiment (t₁₂). Panels centered separately. Five-step changes = value(t+5) − value(t). Observational associations.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.055,1,.88))
    for ext in ("png","pdf"):fig.savefig(out/f"behavior_decoding_correlation_comparison.{ext}",dpi=170)
    plt.close(fig)
    fig,axes = plt.subplots(2,1,figsize=(13,9.2))
    for ax,mode,title in zip(axes,("one_step_changes","five_step_changes"),("One-step changes", "Five-step changes")):
        matrix = np.full((7,9),np.nan)
        for i,metric in enumerate(order):
            for v,version in enumerate(full.VERSIONS):
                for c,category in enumerate(CATEGORIES):
                    matrix[i,3*v+c] = panel[(panel.metric==metric)&(panel.version==version)&(panel.category==category)&(panel["mode"]==mode)].correlation.iloc[0]
        cmap = plt.get_cmap("RdBu_r").copy();cmap.set_bad("#eeeeee")
        im=ax.imshow(matrix,vmin=-1,vmax=1,cmap=cmap,aspect="auto")
        for i in range(7):
            for j in range(9):
                x=matrix[i,j];ax.text(j,i,"flat" if not np.isfinite(x) else f"{x:.2f}",ha="center",va="center",fontsize=8,color="white" if np.isfinite(x) and abs(x)>.65 else "black")
        ax.set(title=title,yticks=np.arange(7),yticklabels=[labels[k] for k in order],xticks=np.arange(9),
            xticklabels=[v+"\n"+c for v in full.VERSIONS for c in ("Green","Yellow","Pink")])
        for x in (2.5,5.5):ax.axvline(x,color="white",lw=2)
        fig.colorbar(im,ax=ax,label="Pearson r",pad=.015)
    fig.suptitle("Behavior–decoding change correlations in each panel\nFlat = behavior did not vary, so its individual-panel correlation is undefined",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93))
    for ext in ("png","pdf"):fig.savefig(out/f"behavior_decoding_correlations_by_panel.{ext}",dpi=170)
    plt.close(fig)


def run(path,behavior_root,out):
    path=Path(path).resolve();behavior_root=Path(behavior_root).resolve();out=Path(out).resolve()
    run=full.read(path)
    full.require(run["status"]=="completed", "Incomplete source decoding")
    full.require(full.read(behavior_root/"motion_audit.json")["status"]=="passed", "Unverified motion")
    out.mkdir(parents=True,exist_ok=True);(out/"source").mkdir(exist_ok=True)
    sources=(Path(__file__).name,"full_capability_evaluation.py","analyze_decoding_behavior.py","analyze_replayed_motion.py","plot_early_decoding.py","analyze_early_decoding.py","analyze_instantaneous_decoding.py")
    for name in sources:shutil.copyfile(Path(__file__).with_name(name),out/"source"/name)
    input_files=[path,behavior_root/"motion_audit.json",behavior_root/"saved_behavior_audit.json",behavior_root/"motion_per_seed.csv"]
    provenance={str(p):full.sha(p) for p in input_files}
    values,table,audits=load_matched(run,behavior_root)
    table.to_csv(out/"matched_curves_per_seed.csv",index=False)
    np.savez_compressed(out/"matched_curves.npz",values=values,versions=np.array(full.VERSIONS),categories=np.array(CATEGORIES),metrics=np.array(("accuracy",)+METRICS),env_steps=np.arange(31),seeds=np.arange(1,6))
    summary,panel,differences=summarize(values,out);plot(summary,panel,out)
    for path,digest in provenance.items():full.require(full.sha(path)==digest,"Changed input "+path)
    full.write(out/"analysis_audit.json",dict(status="passed",completed_utc=full.now(),input_sha256=provenance,
        policy_audits=audits,source_sha256={name:full.sha(out/"source"/name) for name in sources},
        steps=[0,30],panels=9,metrics=list(METRICS),conditions=["rnn_diverse_influence"],
        estimand="Pearson correlation of seed-mean curves, centering each panel separately then pooling its timepoints; level, 1-step and 5-step changes",
        uncertainty="Displayed approximate95% intervals: stratified delete-one-policy-seed jackknife SE on Fisher z, Student t12; contrasts use raw-correlation differences. Sensitivity:10,000 bootstrap draws of five seeds independently within each experiment, with all categories and metrics kept together",
        equal_profile_weights=True,identical_episode_cohort_for_all_metrics_and_timesteps=True,
        reproduces_saved_motion_curves=True,policy_retraining=False,probe_refitting=False,new_rollouts=False,
        interpretation="Observational trajectory association; no causal inference, iid timestep p-values, or out-of-sample prediction claim. Five-step differences overlap; uncertainty resamples policy seeds rather than treating windows as independent."))
    lines=["# Behavioral correlates of instantaneous decoding\n",
        "All seven distinct behavioral metrics plotted across the two figures are compared on identical replay-validated held-out episodes, steps0–30, multi-partner RNNs only, three experiments × green/yellow/pink, five policy seeds. The dashed fixed-step20 readout is another decoding measurement, not a behavioral metric, and is not ranked. Round completion appears in both figures and is counted once.\n",
        "Compute episode event indicators exactly as in the original figures; average episodes within each profile, profiles equally within each seed, then average five seeds. The same episodes validated through step30 are retained at every timestep for every metric. Actual assignment at step t follows h_t; all other plotted events precede h_t.\n",
        "For each panel separately, subtract its time mean from predictor and decoding. Pool the centered samples across nine equally represented panels and compute Pearson r. This removes between-panel offsets. Constant behavioral curves contribute zero covariance without dropping their panels. Individual-panel undefined correlations are marked flat.\n",
        "Levels correlate x(t) with accuracy(t), 31 points/panel. One-step changes correlate x(t+1)-x(t) with accuracy(t+1)-accuracy(t),30points/panel. Five-step changes use t+5 instead,26points/panel. No lag is selected for the main comparisons. These describe contemporaneous alignment of changes, not prediction of unseen data. Five-step changes overlap and are temporally dependent.\n",
        "Uncertainty: displayed approximate95% intervals use delete-one-policy-seed jackknife standard errors on Fisher-z-transformed correlations, stratified by experiment, with Student t12 (three experiments times four within-experiment seed degrees of freedom). Each deletion keeps all categories together. Correlation contrasts use the same jackknife on their raw differences. These intervals describe policy-seed uncertainty, not resampling of capabilities or evaluation episodes. No timestep-level significance tests.\n",
        "Sensitivity also independently resamples five policy seeds within each experiment, keeping all categories and metrics together; recompute seed-mean curves and correlations,10,000draws,RNG30000. Rank frequencies and paired differences are saved. Raw bootstrap percentile ranges are retained in CSV but are not the displayed confidence intervals: with only five seeds, duplicate resampling increases noise in the mean curves and biases correlation estimates downward; several percentile ranges fail to contain the observed estimate. The approximate jackknife interval avoids using that biased percentile distribution as a confidence interval.\n",
        "Positive-lag sensitivity is also saved: correlate behavior changes at t with decoding changes at t+lag, lags0..5. These comparisons are descriptive and were not used to optimize the primary ranking. Timing resolution, readout changes, mixtures of capabilities, and cumulative event definitions can affect correlations. None identifies a causal mechanism.\n",
        "| Metric | Level r | 1-step change r | 5-step change r (approx95% seed jackknife CI) |",
        "|---|---:|---:|---:|"]
    for metric in summary[summary["mode"]=="five_step_changes"].sort_values("correlation",ascending=False).metric:
        g=summary[summary.metric==metric].set_index("mode");d=g.loc["five_step_changes"]
        lines.append(f"| {d.label} | {g.loc['levels'].correlation:.3f} | {g.loc['one_step_changes'].correlation:.3f} | {d.correlation:.3f} [{d.ci_low:.3f}, {d.ci_high:.3f}] |")
    lines += ["\nThe waiting metric and second-move metric are correlated behavioral descriptions and are not independent evidence. Raw levels are especially vulnerable to a shared increasing time trend; compare the change columns when asking what changes alongside decoding.\n",
        "Files: correlation_summary.csv; panel_correlations.csv; policy_seed_correlations.csv; paired_metric_differences.csv; lag_sensitivity.csv; matched_curves_per_seed.csv/NPZ; correlation_bootstrap.npz; comparison and per-panel PNG/PDF figures; analysis_audit.json; frozen source/.\n"]
    (out/"FINDINGS.md").write_text("\n".join(lines)+"\n")
    print(summary.to_string(index=False),flush=True)
    print("[completed]",out,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("path",type=Path);p.add_argument("--behavior-root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run(a.path,a.behavior_root,a.out)
