"""All-seed round-by-round assignments and effective switches for v1_matched/v2/v3.

Read scalar evaluation fields, retaining terminal transitions and excluding scan
padding. An assignment names the ego's goal; the partner gets the other color.
"""

import argparse
import json
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_record_paths
from eval.plot_overcooked_paper_match import DEFAULT_MANIFEST, STYLES, VERSIONS, sha256, write_csv

DEFAULT_OUT = ROOT / "/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/v1_matched_v2_v3_20261019_191029_step_penalty_0.03/assignment_behaviors_all_seeds"
SEEDS = tuple(range(1, 6))
FIELDS = ("dones", "round_done", "round_idx", "round_time", "is_t0", "partner_assignment")
CACHE_VERSION = 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def extract_arrays(a, version, influence, rounds=20):
    """Reconstruct within-round switches from actual, post-action assignments."""
    dones = a["dones"].astype(bool)
    shape = dones.shape
    require(dones.ndim == 2 and all(a[k].shape == shape for k in FIELDS), "Scalar shapes disagree")
    require(dones.any(axis=1).all(), "Every episode must terminate")
    lengths = dones.argmax(axis=1) + 1
    alive = np.arange(shape[1])[None, :] < lengths[:, None]
    endings = a["round_done"].astype(bool) & alive
    initial = a["is_t0"].astype(bool) & alive
    require((endings.sum(axis=1) == rounds).all(), "Every episode must have all round endings")
    require((initial.sum(axis=1) == rounds).all(), "Every episode must have all round starts")
    ordered = np.tile(np.arange(rounds), (shape[0], 1))
    rid = a["round_idx"]
    require(np.array_equal(rid[endings].reshape(shape[0], rounds), ordered), "Endings not in round order")
    require(np.array_equal(rid[initial].reshape(shape[0], rounds), ordered), "Starts not in round order")
    end_idx = np.nonzero(endings)[1].reshape(shape[0], rounds)
    start_idx = np.nonzero(initial)[1].reshape(shape[0], rounds)
    require(np.array_equal(start_idx[:, 0], np.zeros(shape[0], int)), "Episodes must begin at round start")
    require(np.array_equal(start_idx[:, 1:], end_idx[:, :-1] + 1), "Round gaps or overlap")
    require(np.array_equal(end_idx[:, -1], lengths - 1), "Final round must end at first done")
    ep, tick = np.nonzero(alive)
    round_ids = rid[alive].astype(int)
    local = tick - start_idx[ep, round_ids]
    require(np.array_equal(a["round_time"][alive], local + 1), "Round time mismatch")
    require(np.array_equal(a["is_t0"][alive], local == 0), "Initial-step mask mismatch")
    assignment = a["partner_assignment"]
    require(np.isin(assignment[alive], (1, 2)).all(), "Uncommitted or unknown effective assignment")
    changed = np.zeros(shape, bool)
    changed[:, 1:] = ((assignment[:, 1:] != assignment[:, :-1])
                     & (rid[:, 1:] == rid[:, :-1]) & ~initial[:, 1:] & alive[:, 1:])
    if "assignment_changed" in a:
        require(np.array_equal(a["assignment_changed"].astype(bool) & alive, changed),
                "Saved switches disagree with reconstructed effective switches")
    if version == "v1_matched" or not influence:
        require(not changed.any(), "Fixed/no-influence assignment switched within a round")
    switches = np.bincount(ep * rounds + round_ids, weights=changed[alive],
                           minlength=shape[0] * rounds).reshape(shape[0], rounds).astype(np.int32)
    episode = np.arange(shape[0])[:, None]
    first_red = assignment[episode, start_idx] == 1
    final_red = assignment[episode, end_idx] == 1
    require(np.array_equal(first_red != final_red, switches % 2 == 1), "Switch parity disagrees with endpoints")
    return dict(initial_ego_red=first_red, final_ego_red=final_red, switches=switches,
                round_steps=end_idx - start_idx + 1)


def read_policy(directory, version, condition, seed, population, cache_dir):
    path = directory / f"{condition}_seed{seed}_{population}.h5"
    summary_path = directory / f"{condition}_seed{seed}_summary.json"
    summary_raw = summary_path.read_bytes()
    saved = json.loads(summary_raw)[population]
    stat = path.stat()
    provenance = dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                      summary=str(summary_path), summary_sha256=sha256(summary_raw), cache_version=CACHE_VERSION)
    stem = cache_dir / f"{version}_{condition}_seed{seed}"
    meta, archive = stem.with_suffix(".json"), stem.with_suffix(".npz")
    if meta.exists() and archive.exists() and json.loads(meta.read_text())["source"] == provenance:
        with np.load(archive) as f:
            return {k: np.asarray(f[k]) for k in f}, json.loads(meta.read_text())
    with h5py.File(path) as f:
        a = {k: np.asarray(f[k]) for k in FIELDS}
        if "assignment_changed" in f:
            a["assignment_changed"] = np.asarray(f["assignment_changed"])
        pool = np.asarray(f["capability_pool"])
        profile = np.asarray(f["capability_index_per_ep"])
        data = extract_arrays(a, version, "noinfluence" not in condition)
    expected_profiles = 22 if population == "test" else 24
    require(pool.shape == (expected_profiles, 2) and len(np.unique(pool, axis=0)) == expected_profiles,
            "Unexpected evaluation partner pool")
    require(np.issubdtype(profile.dtype, np.integer) and
            np.array_equal(np.sort(profile), np.repeat(np.arange(expected_profiles), 20)),
            "Need twenty evaluation episodes per profile")
    require(len(profile) == saved["n_episodes"] and saved["n_rounds"] == len(profile) * 20,
            "Saved evaluation episode/round counts disagree")
    if version != "v1_matched":
        require(np.array_equal(data["switches"].sum(axis=0), saved["assignment_switch_count_by_round"]),
                "Switch counts disagree with existing evaluation summary")
        require(np.isclose(data["switches"].mean(), saved["assignment_switches_per_round"]),
                "Mean switches disagree with existing evaluation summary")
    data.update(pool=pool, profile=profile)
    record = dict(source=provenance, scalar_datasets_sha256={k: sha256(v.tobytes()) for k, v in a.items()},
                  saved_episode_return=saved["mean_ep_return"], saved_success=saved["round_success_rate"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, **data)
    meta.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return data, record


def round_metrics(data, mask=None):
    mask = np.ones(len(data["profile"]), bool) if mask is None else mask
    require(mask.any(), "Empty episode group")
    first, final, switches = (data[k][mask] for k in ("initial_ego_red", "final_ego_red", "switches"))
    delay = data["pool"][data["profile"][mask]]
    red_faster, blue_faster = delay[:, 0] < delay[:, 1], delay[:, 1] < delay[:, 0]
    non_tie = red_faster | blue_faster
    desired_ego_red = blue_faster[:, None]
    result = dict(initial_ego_red=first.mean(axis=0), final_ego_red=final.mean(axis=0),
                  switches_per_round=switches.mean(axis=0),
                  cumulative_switches=np.cumsum(switches, axis=1).mean(axis=0),
                  fraction_rounds_switched=(switches > 0).mean(axis=0),
                  initial_partner_faster=(first == desired_ego_red)[non_tie].mean(axis=0),
                  final_partner_faster=(final == desired_ego_red)[non_tie].mean(axis=0))
    for label, selected in (("partner_faster_red", red_faster), ("partner_faster_blue", blue_faster)):
        if selected.any():
            result["initial_ego_red_given_" + label] = first[selected].mean(axis=0)
            result["final_ego_red_given_" + label] = final[selected].mean(axis=0)
    return result


def save_figure(fig, out, name, report=None):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=180, facecolor="white")
    if report is not None:
        report.savefig(fig, facecolor="white")
    plt.close(fig)


def overview(groups, out, metric, title, ylabel, fraction=False, online_only=False, report=None):
    versions = VERSIONS[1:] if online_only else VERSIONS
    fig, axes = plt.subplots(4, len(versions), figsize=(4.4 * len(versions), 11),
                             sharex=True, sharey="row", squeeze=False)
    colors = plt.get_cmap("tab10").colors
    x = np.arange(1, 21)
    for ci, (condition, label, _) in enumerate(STYLES):
        row_max = max(float(np.max(groups[(v, condition, s)][metric])) for v in versions for s in SEEDS)
        for vi, version in enumerate(versions):
            ax = axes[ci, vi]
            values = np.stack([groups[(version, condition, s)][metric] for s in SEEDS]) * (100 if fraction else 1)
            for seed, y in zip(SEEDS, values):
                ax.plot(x, y, color=colors[seed - 1], linewidth=1, alpha=.7)
            ax.plot(x, values.mean(axis=0), color="black", linewidth=2.4, zorder=3)
            if fraction:
                ax.set_ylim(-2, 102)
                ax.axhline(50, color="#888888", linewidth=.6, linestyle=":")
            else:
                ax.set_ylim(-.02, max(.5, row_max * 1.08 + .02))
            if ci == 0:
                ax.set_title(version, fontsize=14)
            if vi == 0:
                ax.set_ylabel(label + "\n" + ylabel, fontsize=9)
            ax.grid(alpha=.15)
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.set_xlim(1, 20)
            if ci == 3:
                ax.set_xlabel("Round within partner episode")
    handles = [Line2D([], [], color=colors[s - 1], label=f"Seed {s}") for s in SEEDS]
    handles.append(Line2D([], [], color="black", linewidth=2.4, label="Mean of 5 seeds"))
    fig.suptitle(title, fontsize=17, y=.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .965), ncol=6, fontsize=9, frameon=False)
    note = "Every line averages complete rounds over all partner profiles for one training seed; black is the equal-seed mean."
    if metric in ("initial_ego_red", "final_ego_red"):
        note += "\nRed means ego takes red, partner takes blue. Blue assignment percentage = 100 − red percentage."
    else:
        note += "\nSwitches are actual within-round assignment changes; round resets and initial choices are excluded."
    fig.text(.5, .012, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .90))
    save_figure(fig, out, metric + "_all_seeds", report)


def conditioned_choices(groups, out, report=None):
    fig, axes = plt.subplots(4, 3, figsize=(13.2, 11), sharex=True, sharey=True)
    x = np.arange(1, 21)
    for ci, (condition, label, _) in enumerate(STYLES):
        for vi, version in enumerate(VERSIONS):
            ax = axes[ci, vi]
            for orientation, color, linestyle in (("red", "#c94949", "-"), ("blue", "#337ac4", "--")):
                values = np.stack([groups[(version, condition, s)]["final_ego_red_given_partner_faster_" + orientation] for s in SEEDS]) * 100
                for y in values:
                    ax.plot(x, y, color=color, linewidth=.8, alpha=.22, linestyle=linestyle)
                ax.plot(x, values.mean(axis=0), color=color, linewidth=2.3, linestyle=linestyle)
            if ci == 0:
                ax.set_title(version, fontsize=14)
            if vi == 0:
                ax.set_ylabel(label + "\nFinal ego-red assignment (%)", fontsize=9)
            if ci == 3:
                ax.set_xlabel("Round within partner episode")
            ax.set_ylim(-2, 102)
            ax.set_xlim(1, 20)
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.grid(alpha=.15)
    handles = [Line2D([], [], color="#c94949", label="Partner faster at red (speed-matching ego-red target: 0%)"),
               Line2D([], [], color="#337ac4", linestyle="--", label="Partner faster at blue (speed-matching ego-red target: 100%)")]
    fig.suptitle("Final assignments by partner ability", fontsize=17, y=.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .963), ncol=1, frameon=False, fontsize=10)
    fig.text(.5, .012, "Thin lines: all five individual training seeds; thick lines: equal-seed means.\n"
             "Partner color is complementary to ego assignment. Speed matching is a behavioral diagnostic, not a geometry-aware optimal policy.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .885))
    save_figure(fig, out, "final_choice_by_partner_ability_all_seeds", report)


def seed_dashboard(groups, out, seed, report=None):
    metrics = (("initial_ego_red", "Initial ego-red assignment (%)", 100),
               ("final_ego_red", "Final ego-red assignment (%)", 100),
               ("switches_per_round", "Mean switches per round", 1),
               ("final_partner_faster", "Final partner on faster task (%)", 100))
    fig, axes = plt.subplots(4, 3, figsize=(13.2, 11), sharex=True, sharey="row")
    x = np.arange(1, 21)
    for ri, (metric, label, scale) in enumerate(metrics):
        row_max = max(float(np.max(groups[(v, c, seed)][metric])) for v in VERSIONS for c, _, _ in STYLES)
        for vi, version in enumerate(VERSIONS):
            ax = axes[ri, vi]
            for condition, _, color in STYLES:
                ax.plot(x, groups[(version, condition, seed)][metric] * scale, color=color, linewidth=1.8)
            if ri == 0:
                ax.set_title(version, fontsize=14)
            if vi == 0:
                ax.set_ylabel(label, fontsize=10)
            if scale == 100:
                ax.set_ylim(-2, 102)
                ax.axhline(50, color="#999999", linestyle=":", linewidth=.6)
            else:
                ax.set_ylim(-.02, max(.5, row_max * 1.08 + .02))
                if version == "v1_matched":
                    ax.text(.5, .5, "Assignments fixed within each round", transform=ax.transAxes,
                            ha="center", fontsize=9, color="#666666")
            ax.set_xlim(1, 20)
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.grid(alpha=.15)
            if ri == 3:
                ax.set_xlabel("Round within partner episode")
    handles = [Line2D([], [], color=color, label=label) for _, label, color in STYLES]
    fig.suptitle(f"Assignment behavior: training seed {seed}", fontsize=17, y=.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .965), ncol=4, frameon=False, fontsize=10)
    fig.text(.5, .015, "Red assignment = ego red / partner blue. V2 initial assignments are random; V3 initial assignments are learned when influence is enabled.\n"
             "No-influence curves show realized assignments, not voluntary choices. Faster-task matching is descriptive, not a full optimality score.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .92))
    save_figure(fig, out, f"seed_{seed:02d}_assignment_behavior", report)


def analyze_population(manifest, out, population, manifest_provenance):
    out.mkdir(parents=True, exist_ok=True)
    groups, summaries, round_rows, profile_rows, sources = {}, [], [], [], []
    for version in VERSIONS:
        for condition, _, _ in STYLES:
            for seed in SEEDS:
                directory = Path(manifest["evaluation_inputs"][version][str(seed)])
                data, record = read_policy(directory, version, condition, seed, population, out / "cache")
                meta = dict(version=version, condition=condition, seed=seed, population=population)
                metrics = round_metrics(data)
                groups[(version, condition, seed)] = metrics
                sources.append(dict(**meta, **record))
                for r in range(20):
                    round_rows.append(dict(**meta, round=r + 1, n_episodes=len(data["profile"]),
                                           **{k: float(v[r]) for k, v in metrics.items()}))
                summaries.append(dict(**meta, n_episodes=len(data["profile"]),
                    episode_return=record["saved_episode_return"], round_success=record["saved_success"],
                    initial_ego_red=float(metrics["initial_ego_red"].mean()),
                    final_ego_red=float(metrics["final_ego_red"].mean()),
                    switches_per_round=float(metrics["switches_per_round"].mean()),
                    switches_per_episode=float(metrics["cumulative_switches"][-1]),
                    fraction_rounds_switched=float(metrics["fraction_rounds_switched"].mean()),
                    final_partner_faster_round1=float(metrics["final_partner_faster"][0]),
                    final_partner_faster_round20=float(metrics["final_partner_faster"][-1]),
                    final_partner_faster=float(metrics["final_partner_faster"].mean())))
                for pi, (dr, db) in enumerate(data["pool"]):
                    pm = round_metrics(data, data["profile"] == pi)
                    for r in range(20):
                        profile_rows.append(dict(**meta, delay_red=int(dr), delay_blue=int(db), round=r + 1,
                            initial_ego_red=float(pm["initial_ego_red"][r]), final_ego_red=float(pm["final_ego_red"][r]),
                            switches_per_round=float(pm["switches_per_round"][r]),
                            final_partner_faster=float(pm["final_partner_faster"][r])))
            print(f"Read {population}: {version}/{condition}, seeds 1–5", flush=True)
    write_csv(out / "per_seed_summary.csv", summaries)
    write_csv(out / "per_seed_by_round.csv", round_rows)
    write_csv(out / "per_profile_seed_by_round.csv", profile_rows)
    report = dict(manifest=manifest_provenance, population=population, seeds=list(SEEDS), versions=list(VERSIONS),
        source_sha256=sha256(Path(__file__).read_bytes()), sources=sources, summaries=summaries,
        definitions=dict(assignment="1/red assigns ego to red and partner to blue; 2/blue does the reverse",
            initial="Effective assignment on the first transition of each round; a policy choice in v1_matched and influence-enabled v3, an environment default in v2",
            final="Effective assignment on the terminal transition of each round, including timeouts",
            switch="Change in effective assignment after initialization and within the same round; initialization and changes across round resets are excluded",
            course="Rounds 1–20 within one partner episode; the partner capability stays constant across those rounds",
            weighting="Per seed, equal episode and round weights; all profiles have twenty episodes. Thick overview lines average all five seeds equally",
            faster_task="Partner assigned its lower-delay color, excluding equal-delay profiles; not a geometry-aware optimality claim"),
        caveats=["All training seeds are included; none is selected or dropped.",
                 "No-influence assignment is externally imposed; ego requests do not determine it.",
                 "V1_matched is fixed within a round but may choose differently between rounds.",
                 "V2 initial assignments are random defaults. V3 has learned initial choices only for influence-enabled policies.",
                 "Raw assignment arrays, rather than historical initial-allocation summary fields, determine initial choices.",
                 "These figures describe saved evaluation behavior, not behavior during PPO training.",
                 "No confidence intervals are shown; individual seed trajectories show training-run variability."])
    (out / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}), PdfPages(out / "assignment_behavior_all_seeds.pdf") as pdf:
        for metric, title, ylabel, fraction, online in (
                ("initial_ego_red", "Initial assignment across the partner episode — all seeds", "Ego-red assignment (%)", True, False),
                ("final_ego_red", "Final assignment across the partner episode — all seeds", "Ego-red assignment (%)", True, False),
                ("switches_per_round", "Within-round assignment switches — all seeds", "Mean switches / round", False, True),
                ("cumulative_switches", "Assignment switches accumulated over the episode — all seeds", "Mean cumulative switches", False, True),
                ("fraction_rounds_switched", "Rounds with at least one assignment switch — all seeds", "Rounds with a switch (%)", True, True),
                ("final_partner_faster", "Final assignment matches partner's faster task — all seeds", "Partner on faster task (%)", True, False)):
            overview(groups, out, metric, title, ylabel, fraction, online, report=pdf)
        conditioned_choices(groups, out, report=pdf)
        dashboards = out / "per_seed"
        dashboards.mkdir(exist_ok=True)
        for seed in SEEDS:
            seed_dashboard(groups, dashboards, seed, report=pdf)
    print(f"Saved plots and data: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--populations", nargs="+", choices=("test", "train"), default=["test"])
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    manifest = resolve_record_paths(json.loads(raw))
    require(manifest["evaluation_seeds"] == list(SEEDS), "Expected all five training seeds")
    for population in args.populations:
        folder = "novel" if population == "test" else "familiar"
        analyze_population(manifest, args.out_dir.resolve() / folder, population,
                           dict(path=str(args.manifest.resolve()), sha256=sha256(raw)))


if __name__ == "__main__":
    main()
