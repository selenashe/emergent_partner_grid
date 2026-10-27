"""Measure and plot saved evaluation durations by protocol, condition, and round.

Histograms pool equally sized evaluations from trained seeds. The original
return error bars instead summarize variation among those seeds' mean returns.
Only scalar rollout fields are read; observations and hidden states stay on disk.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_path, save_numpy_archive
from eval.plot_episode_return_dots import CONDITION_STYLES


ROUND_BINS = np.arange(.5, 101, 5)
EPISODE_BINS = np.arange(0, 2001, 50)


def read_durations(path, rounds_per_episode=20, max_steps=100):
    """Recover round durations from boundaries through the first final done.

    Subtracting consecutive inclusive round endpoints counts initialization
    and terminal transitions. Cross-check the result against saved round_time.
    """
    path = Path(path)
    with h5py.File(path) as f:
        dones = np.asarray(f["dones"], dtype=bool)
        round_done = np.asarray(f["round_done"], dtype=bool)
        round_idx = np.asarray(f["round_idx"])
        round_time = np.asarray(f["round_time"])
        success = np.asarray(f["success"], dtype=bool)
        rewards = np.asarray(f["rewards"])
        initial = np.asarray(f["is_t0"], dtype=bool)
        pool = np.asarray(f["capability_pool"])
        profile_indices = np.asarray(f["capability_index_per_ep"])
        initial_capabilities = np.asarray(f["capability"][:, 0, :])
    if dones.ndim != 2 or not dones.any(axis=1).all():
        raise ValueError(f"{path}: incomplete episode")
    n_episodes = len(dones)
    if any(field.shape != dones.shape for field in
           (round_done, round_idx, round_time, success, rewards, initial)):
        raise ValueError(f"{path}: inconsistent scalar rollout shapes")
    if (pool.ndim != 2 or pool.shape[1] != 2
            or len(np.unique(pool, axis=0)) != len(pool)
            or not np.issubdtype(profile_indices.dtype, np.integer)
            or profile_indices.shape != (n_episodes,)
            or (profile_indices < 0).any() or (profile_indices >= len(pool)).any()):
        raise ValueError(f"{path}: invalid partner labels")
    if not np.array_equal(initial_capabilities, pool[profile_indices]):
        raise ValueError(f"{path}: partner capability differs from profile label")

    episode_steps = dones.argmax(axis=1) + 1
    alive = np.arange(dones.shape[1])[None, :] < episode_steps[:, None]
    completed = round_done & alive
    if not np.all(completed.sum(axis=1) == rounds_per_episode):
        raise ValueError(f"{path}: missing or repeated round boundaries")
    if not np.all(round_idx[completed].reshape(n_episodes, rounds_per_episode)
                  == np.arange(rounds_per_episode)[None, :]):
        raise ValueError(f"{path}: unexpected chronological round indices")
    endpoints = (np.nonzero(completed)[1] + 1).reshape(n_episodes, rounds_per_episode)
    round_steps = np.diff(endpoints, axis=1, prepend=0)
    if not np.array_equal(round_steps, round_time[completed].reshape(round_steps.shape)):
        raise ValueError(f"{path}: round boundary duration differs from round_time")
    if (round_steps < 1).any() or (round_steps > max_steps).any():
        raise ValueError(f"{path}: duration outside the round horizon")
    starts = np.concatenate((np.zeros((n_episodes, 1), dtype=int), endpoints[:, :-1]), axis=1)
    if not np.all(np.take_along_axis(initial, starts, axis=1)):
        raise ValueError(f"{path}: a round starts without its initialization tick")
    round_success = success[completed].reshape(round_steps.shape)
    if np.any(round_steps[~round_success] != max_steps):
        raise ValueError(f"{path}: failed round ended before the timeout")
    if not np.array_equal(round_steps.sum(axis=1), episode_steps):
        raise ValueError(f"{path}: round durations do not sum to episode length")
    returns = np.where(alive, rewards, 0).sum(axis=1, dtype=np.float64)
    expected_returns = 1.01 * round_success.sum(axis=1) - .01 * episode_steps
    if not np.allclose(returns, expected_returns, rtol=0, atol=1e-5):
        raise ValueError(f"{path}: return does not match success/step decomposition")
    return dict(round_steps=round_steps, round_success=round_success,
                episode_steps=episode_steps, episode_returns=returns,
                profile_indices=profile_indices, profiles=pool)


def stats(values):
    values = np.asarray(values)
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)),
                p05=float(np.percentile(values, 5)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), min=int(values.min()), max=int(values.max()))


def return_variance(group):
    """Decompose variance among seed means using R = 1.01 S - .01 T."""
    steps = group["episode_steps"].mean(axis=1)
    successes = group["round_success"].sum(axis=2).mean(axis=1)
    returns = group["episode_returns"].mean(axis=1)
    step_component = .01**2 * steps.var(ddof=1)
    success_component = 1.01**2 * successes.var(ddof=1)
    covariance_component = -2 * .01 * 1.01 * np.cov(steps, successes, ddof=1)[0, 1]
    if not np.isclose(step_component + success_component + covariance_component,
                      returns.var(ddof=1), rtol=1e-5, atol=1e-7):
        raise ValueError("Between-seed return variance decomposition failed")
    return dict(
        seed_mean_episode_steps_sd=float(steps.std(ddof=1)),
        seed_mean_successful_rounds_sd=float(successes.std(ddof=1)),
        seed_mean_episode_return_sd=float(returns.std(ddof=1)),
        steps_component_variance=float(step_component),
        successes_component_variance=float(success_component),
        covariance_component_variance=float(covariance_component),
        return_steps_correlation_across_seed_means=float(np.corrcoef(returns, steps)[0, 1]),
        mean_within_seed_episode_steps_sd=float(group["episode_steps"].std(axis=1, ddof=1).mean()),
    )


def load_groups(report):
    seeds = report["seeds"]
    population = report["population"]
    expected_episodes = (22 if population == "test" else 24) * 20
    aggregates = {(a["version"], a["condition"]): a for a in report["aggregates"]}
    versions = tuple(v for v in ("v1_matched", "v2", "v3") if any(k[0] == v for k in aggregates))
    indexed = {}
    for source in report["sources"]:
        if not source.endswith(f"_{population}.h5"):
            continue
        path = resolve_path(source)
        match = re.fullmatch(r"(.+)_seed(\d+)_(train|test)\.h5", path.name)
        version = next(v for v in versions if f"{v}_balanced_training" in path.parts)
        key = (version, match[1], int(match[2]))
        if key in indexed:
            raise ValueError(f"Duplicate rollout for {key}")
        indexed[key] = path
    expected = {(v, c, s) for v in versions for c, _, _ in CONDITION_STYLES for s in seeds}
    if set(indexed) != expected:
        raise ValueError("Missing, duplicate, or unexpected policy rollout")
    groups, per_seed, per_round = {}, [], []
    reference_profiles = None
    for version in versions:
        for condition, _, _ in CONDITION_STYLES:
            samples = []
            for seed in seeds:
                path = indexed[(version, condition, seed)]
                sample = read_durations(path)
                if len(sample["episode_steps"]) != expected_episodes:
                    raise ValueError(f"{path}: incorrect evaluation episode count")
                if not np.all(np.bincount(sample["profile_indices"], minlength=len(sample["profiles"])) == 20):
                    raise ValueError(f"{path}: unequal evaluation profile counts")
                if reference_profiles is None:
                    reference_profiles = sample["profiles"]
                if not np.array_equal(sample["profiles"], reference_profiles):
                    raise ValueError(f"{path}: capability pool differs across rollouts")
                samples.append(sample)
                row = dict(version=version, condition=condition, seed=seed,
                           n_episodes=expected_episodes, n_rounds=expected_episodes * 20,
                           episode_steps_mean=float(sample["episode_steps"].mean()),
                           episode_steps_sd=float(sample["episode_steps"].std(ddof=1)),
                           episode_return_mean=float(sample["episode_returns"].mean()),
                           success=float(sample["round_success"].mean()),
                           round_steps_mean=float(sample["round_steps"].mean()),
                           round_steps_sd=float(sample["round_steps"].std(ddof=1)),
                           timeout_fraction=float((~sample["round_success"]).mean()))
                per_seed.append(row)
                for index in range(20):
                    per_round.append(dict(version=version, condition=condition, seed=seed,
                                          round=index + 1, n_episodes=expected_episodes,
                                          steps_mean=float(sample["round_steps"][:, index].mean()),
                                          steps_sd=float(sample["round_steps"][:, index].std(ddof=1)),
                                          timeout_fraction=float((~sample["round_success"][:, index]).mean())))
            group = {key: np.stack([s[key] for s in samples]) for key in
                     ("round_steps", "round_success", "episode_steps", "episode_returns", "profile_indices")}
            entry = aggregates[(version, condition)]
            mean_steps = group["episode_steps"].mean(axis=1)
            mean_returns = group["episode_returns"].mean(axis=1)
            for measured, saved in ((mean_steps.mean(), entry["episode_steps_mean"]),
                                    (mean_steps.std(ddof=1), entry["episode_steps_sd"]),
                                    (mean_returns.mean(), entry["episode_return_mean"]),
                                    (mean_returns.std(ddof=1), entry["episode_return_sd"]),
                                    (group["round_success"].mean(), entry["success_mean"])):
                if not np.isclose(measured, saved, rtol=1e-6, atol=2e-5):
                    raise ValueError(f"{version}/{condition}: durations do not reproduce saved comparison")
            groups[(version, condition)] = group
        print(f"Validated duration records for {version}", flush=True)
    return versions, groups, per_seed, per_round, reference_profiles


def draw_round_histogram(ax, steps, success, color):
    steps, success = steps.ravel(), success.ravel()
    succeeded = np.histogram(steps[success], bins=ROUND_BINS)[0] * 100 / len(steps)
    failed = np.histogram(steps[~success], bins=ROUND_BINS)[0] * 100 / len(steps)
    ax.bar(ROUND_BINS[:-1], succeeded, width=np.diff(ROUND_BINS), align="edge",
           color=color, alpha=.8, linewidth=.3, edgecolor="white")
    ax.bar(ROUND_BINS[:-1], failed, bottom=succeeded, width=np.diff(ROUND_BINS), align="edge",
           color=color, alpha=.85, hatch="////", edgecolor="#444444", linewidth=.6)
    ax.set_xlim(0, 103)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.text(.97, .95, f"Mean {steps.mean():.1f}; SD {steps.std(ddof=1):.1f}\n"
            f"Timeouts {(~success).mean():.1%}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9)


def save(fig, directory, name):
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def plot_overview(groups, versions, directory, population, n_seeds):
    label = "Novel" if population == "test" else "Shared training"
    for metric in ("round", "episode"):
        fig, axes = plt.subplots(4, len(versions), figsize=(4.5 * len(versions), 11.5),
                                 sharex=True, sharey=True, squeeze=False)
        fig.subplots_adjust(left=.085, right=.985, bottom=.12, top=.89, hspace=.22, wspace=.10)
        for row, (condition, condition_label, color) in enumerate(CONDITION_STYLES):
            for col, version in enumerate(versions):
                ax = axes[row, col]
                group = groups[(version, condition)]
                if metric == "round":
                    draw_round_histogram(ax, group["round_steps"], group["round_success"], color)
                else:
                    values = group["episode_steps"].ravel()
                    counts = np.histogram(values, EPISODE_BINS)[0] * 100 / len(values)
                    ax.bar(EPISODE_BINS[:-1], counts, width=np.diff(EPISODE_BINS),
                           align="edge", color=color, alpha=.8, edgecolor="white", linewidth=.3)
                    seed_means = group["episode_steps"].mean(axis=1)
                    ax.plot(seed_means, np.full(len(seed_means), .025), "|", markersize=10,
                            color="#222222", transform=ax.get_xaxis_transform())
                    ax.set_xlim(0, 2000)
                    ax.set_xticks([0, 500, 1000, 1500, 2000])
                    ax.text(.97, .95, f"Mean {values.mean():.0f}; episode SD {values.std(ddof=1):.0f}\n"
                            f"Seed-mean SD {seed_means.std(ddof=1):.0f}",
                            transform=ax.transAxes, ha="right", va="top", fontsize=9)
                if row == 0:
                    ax.set_title(version, fontsize=16, fontweight="bold", pad=12)
                if col == 0:
                    ax.set_ylabel(condition_label.replace(" ", "\n", 1) + "\nFrequency (%)", fontsize=11)
                ax.grid(axis="y", alpha=.18)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{label} partners: environment steps per {metric}", fontsize=18, fontweight="bold", y=.98)
        if metric == "round":
            handles = [Patch(facecolor="#aaaaaa", label="Successful rounds"),
                       Patch(facecolor="#aaaaaa", hatch="////", edgecolor="#444444", label="100-step timeouts")]
            fig.supxlabel("Environment steps in one round (includes initialization and terminal transition)", y=.065)
            caption = f"Each panel pools {n_seeds} seeds × 440 episodes × 20 rounds. Bin width: 5 steps."
        else:
            handles = [Patch(facecolor="#aaaaaa", label="All evaluated episodes"),
                       Line2D([], [], color="#222222", marker="|", linestyle="none", markersize=10,
                              label="Mean of each trained seed")]
            fig.supxlabel("Total environment steps in a 20-round episode", y=.065)
            caption = f"Each panel pools {n_seeds} seeds × 440 episodes. Bin width: 50 steps. Original return error bars: SD of seed means."
        # Derive episode counts rather than assuming the novel pool when reused.
        episode_count = next(iter(groups.values()))["episode_steps"].shape[1]
        caption = caption.replace("440 episodes", f"{episode_count} episodes")
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .95), ncol=2, frameon=False)
        fig.text(.5, .025, caption, ha="center", fontsize=10, color="#555555")
        save(fig, directory, f"{metric}_steps_histograms")


def plot_each_round(groups, versions, directory, population, n_seeds):
    label = "Novel" if population == "test" else "Shared training"
    round_directory = directory / "by_round"
    round_directory.mkdir(exist_ok=True)
    for version in versions:
        for condition, condition_label, color in CONDITION_STYLES:
            group = groups[(version, condition)]
            fig, axes = plt.subplots(4, 5, figsize=(16, 11), sharex=True, sharey=True)
            for index, ax in enumerate(axes.flat):
                draw_round_histogram(ax, group["round_steps"][:, :, index],
                                     group["round_success"][:, :, index], color)
                ax.set_title(f"Round {index + 1}", fontsize=11)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", alpha=.15)
                ax.set_axisbelow(True)
            fig.suptitle(f"{label} partners · {version} · {condition_label}: steps in every round", fontsize=17)
            fig.supxlabel("Environment steps in the round; bin width 5 steps", y=.04)
            fig.supylabel("Frequency (%)")
            fig.text(.5, .005, f"Each panel: {n_seeds} seeds × {group['episode_steps'].shape[1]} episodes. "
                     "Hatched bars: 100-step timeouts. SD describes rounds pooled across episodes and seeds.",
                     ha="center", fontsize=10, color="#555555")
            fig.tight_layout(rect=(.02, .065, 1, .95))
            save(fig, round_directory, f"{version}_{condition}_round_steps")


def write_readme(summary, directory):
    labels = {condition: label for condition, label, _ in CONDITION_STYLES}
    lines = ["# Evaluation step distributions", "",
             "Columns show v1_matched/v2/v3. Rows show MLP, diverse-partner RNN, single-partner RNN, and no-influence RNN.", "",
             "- [Steps per round](round_steps_histograms.png): pool all 20 rounds; hatched bars identify 100-step timeouts.",
             "- [Steps per episode](episode_steps_histograms.png): sum all 20 rounds; black ticks mark each trained seed's mean.",
             "- [Every round separately](by_round/): each version/condition has 20 histogram panels, one per round index.",
             "- All figures also have PDF exports.", "",
             "Each round includes its stationary initialization tick and its final transition. "
             "Entries after the first episode termination are excluded. Evaluation profiles and training seeds "
             "have equal weight because they contribute the same number of episodes.", "",
             "The return plot's error bars are sample SDs of the trained seeds' mean returns. "
             "The histogram distributions and their episode SDs pool individual evaluations across seeds and profiles. "
             "A broad episode histogram can coexist with closely matched seed means.", "",
             "| Version | Condition | Episode steps: mean / pooled SD | SD of seed mean steps | SD of seed mean return | Timeouts |",
             "|---|---|---:|---:|---:|---:|"]
    for row in summary["summaries"]:
        lines.append(f"| {row['version']} | {labels[row['condition']]} | "
                     f"{row['episode_steps']['mean']:.1f} / {row['episode_steps']['sd']:.1f} | "
                     f"{row['seed_mean_episode_steps_sd']:.1f} | "
                     f"{row['seed_mean_episode_return_sd']:.3f} | {row['timeout_fraction']:.2%} |")
    lines.extend(["", "The exact reward identity is `R = 1.01 * successful_rounds - 0.01 * episode_steps`. "
                  "Longer episodes and more unsuccessful rounds both lower return. "
                  "Their covariance is included in the between-seed variance decomposition in `duration_summary.json`.", "",
                  "Reproduce with:", "", "```bash",
                  f"python eval/plot_step_distributions.py {Path(summary['comparison']).relative_to(ROOT)}",
                  "```", "",
                  "`duration_samples.npz` contains arrays indexed by training seed, episode, and (for round fields) "
                  "round. `per_seed_duration_metrics.csv` and `per_seed_per_round_metrics.csv` retain the measured summaries.", ""])
    (Path(directory) / "README.md").write_text("\n".join(lines))


def run(comparison, out_dir=None):
    comparison = Path(comparison)
    report = json.loads(comparison.read_text())
    versions, groups, per_seed, per_round, profiles = load_groups(report)
    directory = Path(out_dir) if out_dir else comparison.parent / "step_distributions"
    directory.mkdir(parents=True, exist_ok=True)
    summaries, arrays = [], {"seeds": np.array(report["seeds"]), "profiles": profiles}
    for (version, condition), group in groups.items():
        summaries.append(dict(version=version, condition=condition,
                              n_seeds=len(report["seeds"]), n_episodes=int(group["episode_steps"].size),
                              n_rounds=int(group["round_steps"].size),
                              round_steps=stats(group["round_steps"]), episode_steps=stats(group["episode_steps"]),
                              timeout_fraction=float((~group["round_success"]).mean()),
                              **return_variance(group)))
        for key, values in group.items():
            arrays[f"{version}__{condition}__{key}"] = values
    save_numpy_archive(directory / "duration_samples.npz", compressed=True, **arrays)
    for name, rows in (("per_seed_duration_metrics.csv", per_seed), ("per_seed_per_round_metrics.csv", per_round)):
        with (directory / name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(comparison=str(comparison.resolve()), population=report["population"], seeds=report["seeds"],
                   sources=[s for s in report["sources"] if s.endswith(f"_{report['population']}.h5")],
                   definitions=dict(round_steps="Inclusive transitions between consecutive completed round boundaries",
                                    episode_steps="Transitions through the first full-episode done, including that transition",
                                    original_error_bars="Sample SD of ten trained seeds' mean episode returns",
                                    histograms="Equally pooled episodes/rounds, not independent training seeds",
                                    return_identity="R = 1.01 * successful_rounds - 0.01 * episode_steps",
                                    variance_identity="Var(R) = 1.01^2 Var(S) + 0.01^2 Var(T) - 2*1.01*0.01 Cov(S,T), applied to seed means"),
                   round_bin_edges=ROUND_BINS.tolist(), episode_bin_edges=EPISODE_BINS.tolist(), summaries=summaries)
    (directory / "duration_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_readme(summary, directory)
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
        plot_overview(groups, versions, directory, report["population"], len(report["seeds"]))
        plot_each_round(groups, versions, directory, report["population"], len(report["seeds"]))
    print(f"Saved duration histograms and measurements to {directory}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path, help="Saved comparison.json from protocol comparison")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    run(args.comparison, args.out_dir)
