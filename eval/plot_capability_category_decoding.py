"""Collapse the complete 90-profile decoding results into four profile categories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from full_capability_evaluation import (
    CATEGORY_COLORS, CATEGORY_LABELS, COLORS, CONDITIONS, LABELS, TIMES, VERSIONS,
    now, read, require, sha, write,
)

CATEGORY_COUNTS = dict(training_profile=24, unseen_pair_of_familiar_delays=18,
                       one_delay_novel=42, both_delays_novel=6)


def category_averages(table, profiles):
    """Average profiles within each policy seed before bootstrapping policy seeds."""
    profile_table = pd.DataFrame(profiles)[["red_delay", "blue_delay", "category"]].rename(
        columns={"red_delay": "d_R", "blue_delay": "d_B"})
    require(len(profile_table) == 90 and not profile_table.duplicated(["d_R", "d_B"]).any(),
            "Require 90 distinct capability profiles")
    require(profile_table.category.value_counts().to_dict() == CATEGORY_COUNTS, "Wrong category membership counts")
    require(set(zip(profile_table.d_R, profile_table.d_B)) ==
            {(r, b) for r in range(10) for b in range(10) if r != b}, "Incomplete capability universe")
    keys = ["version", "condition", "training_seed", "d_R", "d_B", "cutoff"]
    require(not table.duplicated(keys).any(), "Duplicate profile/seed/cutoff observations")
    require(len(table) == 3*3*5*90*9, "Incomplete decoding table")
    require(set(table.version) == set(VERSIONS) and set(table.condition) == set(CONDITIONS)
            and set(table.training_seed) == {1,2,3,4,5} and set(table.cutoff) == set(TIMES),
            "Unexpected experiments, conditions, seeds, or timesteps")
    require(np.all(table.n_episodes == 4), "Each profile must have four held-out episodes per seed")
    require(np.all((table.n_correct >= 0) & (table.n_correct <= 4))
            and np.all(table.n_correct == table.n_correct.astype(int)), "Invalid held-out correct counts")
    np.testing.assert_allclose(table.accuracy, table.n_correct/table.n_episodes, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(table.true_class, (table.d_B < table.d_R).astype(int))
    merged = table.merge(profile_table, on=["d_R", "d_B"], validate="many_to_one")
    require(len(merged) == len(table), "A decoding profile has no category")
    grouping = ["version", "condition", "training_seed", "category", "cutoff"]
    per_seed = merged.groupby(grouping, as_index=False, sort=True).agg(
        accuracy=("accuracy", "mean"), n_profiles=("d_R", "size"),
        n_episodes=("n_episodes", "sum"), n_correct=("n_correct", "sum"),
        blue_faster_profiles=("true_class", "sum"))
    require(len(per_seed) == 3*3*5*4*9, "Missing seed/category/cutoff combinations")
    require(np.all(per_seed.n_profiles == per_seed.category.map(CATEGORY_COUNTS)), "Missing profiles within a category")
    require(np.all(per_seed.blue_faster_profiles*2 == per_seed.n_profiles), "Unbalanced faster-goal classes in a category")
    np.testing.assert_allclose(per_seed.accuracy, per_seed.n_correct/per_seed.n_episodes, rtol=0, atol=1e-12)
    return per_seed, bootstrap_seeds(per_seed)


def bootstrap_seeds(per_seed):
    """CI over five independent policy seeds, with category profiles fixed."""
    keys = ["version", "condition", "category", "cutoff"]
    require(not per_seed.duplicated(keys+["training_seed"]).any(), "Duplicate category/seed observations")
    indices = np.random.default_rng(30000).integers(0, 5, size=(10000, 5))
    rows = []
    for group, frame in per_seed.groupby(keys, sort=True):
        require(len(frame) == 5 and set(frame.training_seed) == {1,2,3,4,5}, "Require all five policy seeds")
        frame = frame.sort_values("training_seed")
        values = frame.accuracy.to_numpy()
        require(np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)), "Invalid category accuracy")
        require(frame.n_profiles.nunique() == 1 and frame.n_episodes.nunique() == 1, "Unequal category coverage across seeds")
        resamples = values[indices].mean(axis=1)
        rows.append(dict(zip(keys, group), accuracy_mean=float(values.mean()),
            accuracy_ci_low=float(np.quantile(resamples, .025)), accuracy_ci_high=float(np.quantile(resamples, .975)),
            seed_count=5, n_profiles=int(frame.n_profiles.iloc[0]),
            held_out_episodes_per_seed=int(frame.n_episodes.iloc[0])))
    return pd.DataFrame(rows)


def panel(ax, summary, version, category):
    for condition, color, label in zip(CONDITIONS, COLORS, LABELS):
        frame = summary[(summary.version == version) & (summary.category == category)
                        & (summary.condition == condition)].sort_values("cutoff")
        require(np.array_equal(frame.cutoff.to_numpy(), TIMES), "Missing averaged curve cutoffs")
        ax.plot(frame.cutoff, frame.accuracy_mean, c=color, lw=2, label=label)
        ax.fill_between(frame.cutoff.to_numpy(), frame.accuracy_ci_low.to_numpy(),
                        frame.accuracy_ci_high.to_numpy(), color=color, alpha=.18)
    ax.axhline(.5, color="gray", ls="--", lw=1, label="Chance (1/2)")
    ax.set_facecolor(CATEGORY_COLORS[category])
    ax.set_title(f"{version} · {CATEGORY_LABELS[category]}\n{CATEGORY_COUNTS[category]} profiles", fontsize=10)
    ax.set(xlim=(0,400), ylim=(0,1.02), xlabel="Cumulative timesteps", ylabel="Decoding accuracy")
    ax.set_xticks([0,100,200,300,400]); ax.set_yticks([0,.2,.4,.6,.8,1])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def figures(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    categories = list(CATEGORY_COUNTS)
    subtitle = "Profiles averaged within each policy seed; mean across five seeds; shading: 95% bootstrap CI"

    def finish(fig, axes, stem, title, legend_columns=4):
        handles, labels = np.ravel(axes)[0].get_legend_handles_labels()
        fig.suptitle(title + "\n" + subtitle, fontsize=12, y=.995)
        fig.legend(handles, labels, loc="lower center", ncol=legend_columns, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0,.055,1,.94), h_pad=2.0, w_pad=1.3)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{stem}.{ext}", dpi=180)

    fig, axes = plt.subplots(4,3,figsize=(17,14),sharex=True,sharey=True)
    for row, category in enumerate(categories):
        for col, version in enumerate(VERSIONS): panel(axes[row,col], summary, version, category)
    finish(fig, axes, "relative_speed_timestep_category_averages_all_seeds",
           "Faster-goal decoding averaged over capability categories · direct input, fixed t0")
    plt.close(fig)

    for version in VERSIONS:
        fig, axes = plt.subplots(2,2,figsize=(12,8.5),sharex=True,sharey=True)
        for ax, category in zip(axes.flat, categories): panel(ax, summary, version, category)
        finish(fig, axes, f"relative_speed_timestep_category_averages_{version}", f"{version} · Capability category averages")
        plt.close(fig)

    with PdfPages(out / "relative_speed_timestep_category_comparisons.pdf") as pdf:
        for category in categories:
            fig, axes = plt.subplots(1,3,figsize=(16,5),sharex=True,sharey=True)
            for ax, version in zip(axes, VERSIONS): panel(ax, summary, version, category)
            finish(fig, axes, f"relative_speed_timestep_{category}_all_seeds", CATEGORY_LABELS[category])
            pdf.savefig(fig); plt.close(fig)

    individual = out / "per_experiment_category"; individual.mkdir(exist_ok=True)
    for version in VERSIONS:
        for category in categories:
            fig, ax = plt.subplots(figsize=(6.8,4.8))
            panel(ax, summary, version, category)
            ax.legend(frameon=False, fontsize=8, loc="lower right")
            fig.tight_layout()
            fig.savefig(individual / f"relative_speed_timestep_{version}_{category}.png", dpi=180)
            plt.close(fig)


def run(root, out=None):
    root = Path(root).resolve()
    out = Path(out).resolve() if out else root / "category_averages"
    require(read(root / "completion_audit.json")["status"] == "passed", "The source evaluation audit must have passed")
    manifest = read(root / "analysis_run.json")
    require(manifest["status"] == "completed", "The source evaluation must be complete")
    table_path = root / "profile_probe_per_seed.csv"
    table = pd.read_csv(table_path)
    per_seed, summary = category_averages(table, manifest["profiles"])
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "category_probe_per_seed.csv", index=False)
    summary.to_csv(out / "category_probe_summary.csv", index=False)
    figures(summary, out)
    source = out / "source"; source.mkdir(exist_ok=True)
    for path in (Path(__file__), Path(__file__).with_name("full_capability_evaluation.py")):
        target = source / path.name
        if target.resolve() != path.resolve(): shutil.copyfile(path, target)
    write(out / "methods.json", dict(status="completed", completed_utc=now(),
        source_results=str(root), source_table=str(table_path), source_table_sha256=sha(table_path),
        input_rows=len(table), per_seed_rows=len(per_seed), summary_rows=len(summary), category_profile_counts=CATEGORY_COUNTS,
        weighting="Equal weight per profile within each category and policy seed, then equal weight across the five policy seeds.",
        uncertainty="95% percentile bootstrap interval across five policy seeds; 10000 resamples; PCG64 seed 30000. Profiles are fixed, not independent resampling units.",
        episode_partition="Only the four held-out probe episodes per capability profile and policy seed.",
        policy_training_exposure="Seen refers to the 24-profile diverse-training pool; the single-partner control saw only (1,4).",
        policy_retraining=False, new_evaluations=False, new_probe_fitting=False,
        source_sha256={p.name:sha(p) for p in source.glob("*.py")}))
    print(out / "relative_speed_timestep_category_averages_all_seeds.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    run(arguments.root, arguments.out)
