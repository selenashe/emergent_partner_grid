"""Match Overcooked Figure 2A: best training seed, episode-bootstrap mean CI.

Run on saved final checkpoints and scalar rollout arrays; no training required.
The primary plot includes all four grid conditions. A three-condition companion
uses exactly the MLP / diverse RNN / single-partner RNN comparison in Figure 2A.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_path, resolve_record_paths
from train.training_log_format import LOG_LINE

VERSIONS = ("v1_matched", "v2", "v3")
# The first three colors are the upstream xkcd palette, in upstream model order.
STYLES = (
    ("mlp_diverse_influence", "MLP", "#06b48b"),
    ("rnn_diverse_influence", "RNN", "#8e82fe"),
    ("rnn_single_influence", "RNN single-partner", "#e17701"),
    ("rnn_diverse_noinfluence", "RNN no influence", "#5a86ad"),
)
UPSTREAM_COMMIT = "35a8430046531bd4b62e923d7507ee9f02fd97ea"
UPSTREAM_URL = ("https://github.com/ruaridhmon/emergent_partner_modelling/blob/"
                + UPSTREAM_COMMIT + "/analysis/do_performance_plots.py")
UPSTREAM_SHA256 = "435d953ac440d573f87f4291cb63d91fb00cdec55ab5cd54aa24a8ec17104903"
DEFAULT_MANIFEST = ROOT / "train/run_snapshots/analyses_20261022_172833_t0fix_direct_input/analysis_source_t0fix_direct_input/train/manifests/sbatch_v1_matched_v2_v3_20261022_080155_t0fix_direct_input.json"
DEFAULT_OUT = ROOT / "eval/protocol_comparison/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/behavioral_analysis_t0fix_direct_input/overcooked_paper_match"


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def load_manifests(path):
    """Read linked original, extension, v3 and matched manifests once each."""
    records, visited = [], set()

    def visit(value):
        target = resolve_path(value).resolve()
        if target in visited:
            return
        visited.add(target)
        raw = target.read_bytes()
        manifest = resolve_record_paths(json.loads(raw))
        records.append((manifest, dict(path=str(target), sha256=sha256(raw))))
        for key in ("base_manifest", "comparison_manifest"):
            if manifest.get(key):
                visit(manifest[key])

    visit(path)
    return records


def training_candidates(records):
    """Score final logged rollout only; held-out returns never enter selection."""
    candidates, seen = [], set()
    for manifest, provenance in records:
        for job in manifest["training_jobs"]:
            version, condition, seed = job["version"], job["condition"], int(job["seed"])
            if version not in VERSIONS:
                continue
            key = (version, condition, seed)
            if key in seen:
                raise ValueError(f"Duplicate training policy: {key}")
            seen.add(key)
            outputs = [x.split("=", 1)[1] for x in job["command"] if x.startswith("--output=")]
            if len(outputs) > 1:
                raise ValueError(f"Ambiguous training log for {key}")
            log = resolve_path(outputs[0].replace("%j", str(job["job_id"])) if outputs
                               else ROOT / "train/slurm_logs" / f"cg_cap_{job['job_id']}.out")
            raw = log.read_bytes()
            updates = []
            for line in raw.decode().splitlines():
                if line.startswith("[u "):
                    match = LOG_LINE.match(line)
                    if match is None:
                        raise ValueError(f"Unrecognized training metric: {log}: {line}")
                    updates.append(tuple(float(v) for v in match.groups()))
            config_path = resolve_path(Path(manifest["versions"][version]["checkpoint_root"])
                                       / f"{job['tag']}_config.json")
            config_raw = config_path.read_bytes()
            config = json.loads(config_raw)
            n = int(config["NUM_UPDATES"])
            if len(updates) != n or [int(r[0]) for r in updates] != list(range(1, n + 1)):
                raise ValueError(f"Incomplete training log: {log}")
            steps = int(config["NUM_ENVS"]) * int(config["NUM_STEPS"])
            if any(int(r[1]) != int(r[0]) * steps for r in updates):
                raise ValueError(f"Inconsistent training budget: {log}")
            if config["SEED"] != seed or config["NUM_SEEDS"] != 1:
                raise ValueError(f"Configuration seed mismatch: {config_path}")
            final = updates[-1]
            if not np.isfinite(final).all() or final[5] <= 0:
                raise ValueError(f"No valid completed-episode mean in final update: {log}")
            if int(final[1]) != int(manifest["effective_timesteps_per_policy"]):
                raise ValueError(f"Final training budget differs from manifest: {log}")
            candidates.append(dict(version=version, condition=condition, seed=seed,
                training_return=final[4], final_update=int(final[0]), environment_steps=int(final[1]),
                training_log=str(log), training_log_sha256=sha256(raw),
                config=str(config_path), config_sha256=sha256(config_raw),
                manifest=provenance["path"]))
    return candidates


def select_candidates(candidates, seeds):
    selected = []
    for version in VERSIONS:
        for condition, _, _ in STYLES:
            group = [r for r in candidates if r["version"] == version
                     and r["condition"] == condition and r["seed"] in seeds]
            if sorted(r["seed"] for r in group) != sorted(seeds):
                raise ValueError(f"Missing or duplicate candidates: {version}/{condition}")
            # Printed returns have three decimals. Resolve ties before evaluation.
            selected.append(max(group, key=lambda r: (r["training_return"], -r["seed"])))
    return selected


def episode_returns(rewards, dones, round_done, round_idx, rounds=20):
    """Each independent row is one complete episode, including its first done."""
    if dones.ndim != 2 or any(a.shape != dones.shape for a in (rewards, round_done, round_idx)):
        raise ValueError("Inconsistent scalar rollout shapes")
    if not dones.any(axis=1).all():
        raise ValueError("Every episode must have a final done")
    lengths = dones.argmax(axis=1) + 1
    alive = np.arange(dones.shape[1])[None, :] < lengths[:, None]
    complete = round_done & alive
    if not np.all(complete.sum(axis=1) == rounds):
        raise ValueError("Every episode must contain exactly twenty completed rounds")
    if not np.array_equal(round_idx[complete].reshape(len(dones), rounds),
                          np.tile(np.arange(rounds), (len(dones), 1))):
        raise ValueError("Round indices must occur once each, in order")
    if not np.all(complete[np.arange(len(dones)), lengths - 1]):
        raise ValueError("The last round must end at first done")
    if not np.isfinite(rewards[alive]).all():
        raise ValueError("Nonfinite valid rewards")
    return np.where(alive, rewards, 0).sum(axis=1, dtype=np.float64)


def bootstrap_mean_ci(values, n_boot=1000, seed=2026):
    """Seaborn's ordinary row bootstrap: mean, 1000 draws, percentile 95% CI.

    Intentionally no profile stratification, training-seed bootstrap, or separate
    round sampling: none of these are used in the authors' Figure 2A source.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite episode returns")
    if n_boot < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        indices = rng.integers(0, len(values), len(values), dtype=np.intp)
        means[i] = values[indices].mean()
    return float(values.mean()), np.percentile(means, [2.5, 97.5]).tolist()


def read_evaluation(directory, condition, seed):
    path = directory / f"{condition}_seed{seed}_test.h5"
    summary_path = directory / f"{condition}_seed{seed}_summary.json"
    summary_raw = summary_path.read_bytes()
    summary = json.loads(summary_raw)["test"]
    with h5py.File(path) as f:
        keys = ("rewards", "dones", "round_done", "round_idx",
                "capability_pool", "capability_index_per_ep")
        data = {k: np.asarray(f[k]) for k in keys}
    returns = episode_returns(data["rewards"], data["dones"], data["round_done"], data["round_idx"])
    pool, profiles = data["capability_pool"], data["capability_index_per_ep"]
    if pool.shape != (22, 2) or len(np.unique(pool, axis=0)) != 22:
        raise ValueError(f"Expected 22 distinct held-out profiles: {path}")
    if (profiles.shape != (440,) or not np.issubdtype(profiles.dtype, np.integer)
            or not np.array_equal(np.sort(profiles), np.repeat(np.arange(22), 20))):
        raise ValueError(f"Expected twenty episodes per held-out profile: {path}")
    if len(returns) != summary["n_episodes"] or summary["n_rounds"] != len(returns) * 20:
        raise ValueError(f"Saved evaluation counts disagree: {summary_path}")
    if not np.isclose(returns.mean(), summary["mean_ep_return"], atol=3e-5, rtol=0):
        raise ValueError(f"Returns disagree with saved summary: {summary_path}")
    provenance = dict(evaluation_rollouts=str(path), evaluation_summary=str(summary_path),
        evaluation_summary_sha256=sha256(summary_raw),
        scalar_datasets_sha256={k: sha256(data[k].tobytes()) for k in keys},
        n_episodes=len(returns), n_held_out_profiles=22, episodes_per_profile=20,
        rounds_per_episode=20)
    return returns, profiles, pool, provenance


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(results, out, n_candidates, three_conditions=False, overlay_training=False):
    styles = STYLES[:3] if three_conditions else STYLES
    lookup = {(r["version"], r["condition"]): r for r in results}
    plotted = [lookup[(v, c)] for v in VERSIONS for c, _, _ in styles]
    bounds = [x for r in plotted for x in r["ci95"]]
    if overlay_training:
        bounds.extend(r["training_return"] for r in plotted)
    padding = max(.5, (max(bounds) - min(bounds)) * .12)
    with plt.rc_context({"font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.8), sharey=True)
        fig.subplots_adjust(left=.085, right=.98, top=.62 if overlay_training else .69,
                            bottom=.20, wspace=.14)
        for vi, (v, ax) in enumerate(zip(VERSIONS, axes)):
            for x, (c, _, color) in enumerate(styles):
                r = lookup[(v, c)]
                # Explicit endpoints also preserve asymmetric bootstrap intervals.
                ax.vlines(x, *r["ci95"], color=color, linewidth=2.8, zorder=2)
                ax.plot(x, r["mean"], "o", color=color, markersize=10,
                        markeredgewidth=0, zorder=3)
                if overlay_training:
                    ax.plot(x, r["training_return"], marker="D", linestyle="none",
                            markerfacecolor="none", markeredgecolor=color,
                            markersize=11, markeredgewidth=2, zorder=4)
                label_y = max(r["ci95"][1], r["training_return"]) if overlay_training else r["ci95"][1]
                ax.annotate(f"s{r['seed']}", (x, label_y), xytext=(0, 9),
                            textcoords="offset points", ha="center", fontsize=9, color="#555555")
            ax.set_title(v, fontstyle="italic", fontsize=15, pad=16)
            ax.set_xlim(-.5, len(styles) - .5)
            ax.set_ylim(min(bounds) - padding, max(bounds) + padding * 1.5)
            ax.set_xticks([])
            for spine in ("top", "right", "bottom"):
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_visible(vi == 0)
            ax.spines["left"].set_color("#555555")
            ax.tick_params(axis="y", left=vi == 0, labelleft=vi == 0, colors="#333333")
        axes[0].set_ylabel("Average episode return", labelpad=12)
        handles = [Line2D([], [], marker="o", linestyle="none", color=color,
                          markersize=10, markeredgewidth=0, label=label) for _, label, color in styles]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.98, .87),
                   ncol=len(styles), frameon=True, fancybox=True, framealpha=1,
                   facecolor="#eeeeee", edgecolor="none", fontsize=11, handlelength=1)
        if overlay_training:
            markers = [Line2D([], [], marker="o", linestyle="none", color="#555555",
                              markersize=9, markeredgewidth=0, label="Held-out mean (95% CI)"),
                       Line2D([], [], marker="D", linestyle="none", markeredgecolor="#555555",
                              markerfacecolor="none", markersize=9, markeredgewidth=1.7,
                              label="Final training mean")]
            fig.legend(handles=markers, loc="upper right", bbox_to_anchor=(.98, .79),
                       ncol=2, frameon=False, fontsize=10, handlelength=1)
        fig.text(.085, .96, "Best training seeds: held-out partner performance",
                 fontsize=18, fontweight="bold", va="top")
        note = "Dots: mean of 440 held-out episodes (20 rounds each). Bars: 95% episode-bootstrap CI."
        if overlay_training:
            note = ("Filled dots: held-out means; bars: 95% episode-bootstrap CI over 440 episodes.\n"
                    "Hollow diamonds: final logged training means of those same seeds; no training confidence intervals.")
        fig.text(.085, .035 if overlay_training else .045,
                 f"Best of {n_candidates} seeds per condition, selected by final logged training return. Labels: selected seed.\n" + note,
                 fontsize=10, color="#555555")
        name = "episode_return_paper_conditions" if three_conditions else "episode_return_dots"
        if overlay_training:
            name += "_with_training_returns"
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"{name}.{ext}", dpi=240, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 100:
        parser.error("Use at least 100 bootstrap repetitions")
    records = load_manifests(args.manifest)
    manifest = records[0][0]
    if manifest.get("experiment") != "V1_matched" or manifest["evaluation_seeds"] != list(range(1, 6)):
        raise ValueError("Expected the completed five-seed V1_matched evaluation manifest")
    candidates = training_candidates(records)
    # Freeze both selections before opening any held-out evaluation file.
    selections = {n: select_candidates(candidates, list(range(1, n + 1))) for n in (5,)}
    cache = {}
    for n, selected in selections.items():
        out = args.out_dir.resolve() / f"best_of_{n}"
        out.mkdir(parents=True, exist_ok=True)
        results, episode_rows = [], []
        for index, row in enumerate(selected):
            key = (row["version"], row["condition"], row["seed"])
            if key not in cache:
                directory = Path(manifest["evaluation_inputs"][row["version"]][str(row["seed"])])
                cache[key] = read_evaluation(directory, row["condition"], row["seed"])
            returns, profiles, pool, provenance = cache[key]
            rng_seed = args.bootstrap_seed + index
            mean, ci = bootstrap_mean_ci(returns, args.bootstrap_repetitions, rng_seed)
            results.append(dict(**row, **provenance, mean=mean, ci95=ci, bootstrap_seed=rng_seed))
            for ep, value in enumerate(returns):
                red, blue = pool[profiles[ep]]
                episode_rows.append(dict(version=row["version"], condition=row["condition"],
                    seed=row["seed"], episode=ep, delay_red=int(red), delay_blue=int(blue),
                    episode_return=float(value)))
            print(f"best_of_{n}: {row['version']} {row['condition']} s{row['seed']}: {mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", flush=True)
        report = dict(selection=dict(candidate_seeds=list(range(1, n + 1)),
            metric="Highest mean completed-episode return in final logged training rollout",
            per="version and condition; policies were trained across layouts",
            held_out_results_used=False, tie_break="Lowest seed number", logged_return_decimal_places=3),
            uncertainty=dict(method="Ordinary unstratified episode-row percentile bootstrap of the mean",
                confidence=95, repetitions=args.bootstrap_repetitions,
                bootstrap_seed=args.bootstrap_seed, includes_training_seed_variability=False),
            upstream=dict(source_url=UPSTREAM_URL, source_sha256=UPSTREAM_SHA256),
            manifests=[p for _, p in records], source_sha256=sha256(Path(__file__).read_bytes()),
            results=results, limitations=[
                "The paper selects independently per layout; grid policies train across 1096 layouts, so selection is per version/condition.",
                "The paper uses five training seeds. best_of_5 uses seeds 1-5; best_of_5 uses all available seeds.",
                "The final logged training rollout precedes the last gradient update; evaluation uses the saved final checkpoint.",
                "Final training scores have only three decimal places and use each condition's training partner population.",
                "The code pools episode rows without profile stratification or shared-rollout-seed clustering; this is replicated deliberately.",
                "Intervals condition on the selected policy; they omit seed-selection uncertainty and variation across training runs.",
                "Our held-out evaluation is 22 profiles x 20 episodes. The paper's Figure 2 caption says 8 profiles x 10 rollout seeds; Appendix B.4.1 says 24 x 20. The released code leaves sample counts to external artifacts.",
                "The authors do not pin Seaborn or the bootstrap RNG seed; estimator and resampling procedure match, but exact random endpoints cannot be reproduced.",
                "Grid episode return includes the step penalty and all twenty rounds, rather than soup count over a fixed 400-step Overcooked episode.",
                "Version panels replace paper layout panels; no Overall panel pools different protocol variants.",
            ])
        (out / "selection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        write_csv(out / "training_candidates.csv", [r for r in candidates if r["seed"] <= n])
        write_csv(out / "selected_seed_metrics.csv", [dict(version=r["version"], condition=r["condition"],
            seed=r["seed"], training_return=r["training_return"], mean=r["mean"],
            ci95_low=r["ci95"][0], ci95_high=r["ci95"][1]) for r in results])
        write_csv(out / "episode_returns.csv", episode_rows)
        plot(results, out, n)
        plot(results, out, n, three_conditions=True)
        plot(results, out, n, overlay_training=True)
        plot(results, out, n, three_conditions=True, overlay_training=True)
    print(args.out_dir.resolve())


if __name__ == "__main__":
    main()
