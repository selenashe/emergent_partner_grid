"""Assignment diagnostics for the policies in the best-of-five return figure.

Reuse its saved training-only selections and the all-seed metric definitions.
Recompute the selection from training logs before reading held-out behavior.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_record_paths
from eval.analyze_assignment_behaviors import read_policy, require, round_metrics, save_figure
from eval.plot_overcooked_paper_match import (
    DEFAULT_MANIFEST, DEFAULT_OUT as RETURN_OUT, STYLES, VERSIONS,
    load_manifests, select_candidates, sha256, training_candidates, write_csv,
)

DEFAULT_SELECTION = RETURN_OUT / "best_of_5/selection.json"
DEFAULT_OUT = RETURN_OUT.parent / "assignment_behaviors_best_training_seeds/novel"
METRICS = (
    ("final_ego_red", "Final assignment across the partner episode", "Final ego-red assignment (%)", 100),
    ("final_ego_blue", "Final blue assignment across the partner episode", "Final ego-blue assignment (%)", 100),
    ("final_partner_faster", "Final assignment matches partner's faster task", "Partner on faster task (%)", 100),
    ("switches_per_round", "Within-round assignment switches", "Mean switches / round", 1),
    ("initial_ego_red", "Initial assignment across the partner episode", "Initial ego-red assignment (%)", 100),
    ("cumulative_switches", "Assignment switches accumulated over the episode", "Mean cumulative switches", 1),
    ("fraction_rounds_switched", "Rounds with at least one assignment switch", "Rounds with a switch (%)", 100),
)
DEFINITIONS = dict(
    assignment="RED assigns ego to red and partner to blue; BLUE does the reverse",
    initial="Effective assignment on the first transition of each round",
    final="Effective assignment on the terminal transition of each round, including timeouts",
    blue="Final ego-blue fraction equals one minus the final ego-red fraction; BLUE assigns ego to blue and partner to red",
    switch="Effective assignment changes within a round, excluding initialization and round resets; terminal transitions included",
    course="Rounds 1-20 within one partner episode; capability is constant across those rounds",
    weighting="Equal evaluation-episode weights at each round; 20 episodes per partner profile; one selected training seed per version/condition",
    faster_task="Partner assigned its lower-delay color; equal-delay profiles excluded; descriptive rather than a dynamic optimality score",
)


def verified_selection(path, records):
    raw = path.read_bytes()
    saved = resolve_record_paths(json.loads(raw))
    rule = saved["selection"]
    require(rule["candidate_seeds"] == list(range(1, 6)), "Expected best-of-five selection")
    require(rule["held_out_results_used"] is False, "Selection must use training only")
    actual = {(r["version"], r["condition"]): r for r in saved["results"]}
    expected = select_candidates(training_candidates(records), list(range(1, 6)))
    require(len(actual) == len(saved["results"]) == len(expected) == 12,
            "Expected exactly one selected policy per version and condition")
    require(set(actual) == {(r["version"], r["condition"]) for r in expected},
            "Selection version/condition coverage differs")
    for row in expected:
        previous = actual[(row["version"], row["condition"])]
        for field in ("seed", "training_return", "final_update", "environment_steps",
                      "training_log_sha256", "config_sha256"):
            require(row[field] == previous[field],
                    f"Saved selection differs from training logs: {row['version']}/{row['condition']}/{field}")
    return saved, dict(path=str(path.resolve()), sha256=sha256(raw),
                       recomputed_from_training_logs=True)


def format_axis(ax, scale, maximum, reference=False):
    ax.set(xlim=(1, 20), xticks=[1, 5, 10, 15, 20])
    ax.grid(alpha=.16)
    if scale == 100:
        ax.set_ylim(-2, 102)
        if reference:
            ax.axhline(50, color="#999999", linewidth=.7, linestyle=":")
    else:
        ax.set_ylim(-.02, max(.5, maximum * 1.08 + .02))


def overview(groups, selected, out, spec, report, styles=STYLES):
    metric, title, ylabel, scale = spec
    versions = VERSIONS[1:] if metric in ("switches_per_round", "cumulative_switches", "fraction_rounds_switched") else VERSIONS
    fig, axes = plt.subplots(len(styles), len(versions), figsize=(4.4 * len(versions), 2.3 * len(styles) + 1.3), sharex=True, sharey="row")
    for ci, (condition, label, color) in enumerate(styles):
        maximum = max(float(np.max(groups[v, condition][metric])) for v in versions)
        for vi, version in enumerate(versions):
            ax = axes[ci, vi]
            ax.plot(np.arange(1, 21), groups[version, condition][metric] * scale,
                    color=color, linewidth=2)
            format_axis(ax, scale, maximum, metric in ("initial_ego_red", "final_ego_red", "final_ego_blue", "final_partner_faster"))
            ax.text(.99, 1.02, f"Seed {selected[version, condition]['seed']}",
                    transform=ax.transAxes, fontsize=10, va="bottom", ha="right")
            if ci == 0:
                ax.set_title(version, fontsize=14, fontstyle="italic")
            if vi == 0:
                ax.set_ylabel(label + "\n" + ylabel, fontsize=9)
            if ci == len(styles) - 1:
                ax.set_xlabel("Round within partner episode")
    fig.suptitle(title + " — selected training seeds", fontsize=16, y=.99)
    fig.text(.5, .015, "One policy per experiment and condition, selected by final logged training return among seeds 1–5.\n"
             "Each curve averages 440 held-out episodes. Seed choices match the best-of-five return figure.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .955))
    save_figure(fig, out, metric + "_best_training_seeds", report)


def dashboard(groups, selected, out, report, styles=STYLES, metrics=METRICS[:4],
              name="assignment_behavior_best_training_seeds",
              title="Assignment behavior of the selected training seeds"):
    fig, axes = plt.subplots(len(metrics), 3, figsize=(13.2, 2.3 * len(metrics) + 2.2), sharex=True, sharey="row")
    for ri, (metric, _, ylabel, scale) in enumerate(metrics):
        maximum = max(float(np.max(g[metric])) for g in groups.values())
        for vi, version in enumerate(VERSIONS):
            ax = axes[ri, vi]
            for condition, label, color in styles:
                ax.plot(np.arange(1, 21), groups[version, condition][metric] * scale,
                        color=color, linewidth=1.9,
                        label=f"{label} · s{selected[version, condition]['seed']}")
            format_axis(ax, scale, maximum, scale == 100)
            if ri == 0:
                ax.set_title(version, fontsize=14, fontstyle="italic", pad=50)
                ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.015),
                          ncol=2, fontsize=8, frameon=False, handlelength=1.5, columnspacing=.9)
            if vi == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if ri == len(metrics) - 1:
                ax.set_xlabel("Round within partner episode")
            if metric == "switches_per_round" and version == "v1_matched":
                ax.text(.5, .5, "Assignments fixed within each round", transform=ax.transAxes,
                        ha="center", fontsize=9, color="#666666")
    fig.suptitle(title, fontsize=17, y=.99)
    assignment_note = "Red = ego red / partner blue; blue = ego blue / partner red. Final assignments average 440 episodes at each round."
    if any(c == "rnn_diverse_noinfluence" for c, _, _ in styles):
        assignment_note += " No-influence assignments are imposed by the environment."
    fig.text(.5, .015, "Same policy selections as the best-of-five held-out return figure; selected by final logged training return.\n"
             + assignment_note,
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .96))
    save_figure(fig, out, name, report)


def conditioned_choices(groups, selected, out, report, styles=STYLES):
    fig, axes = plt.subplots(len(styles), 3, figsize=(13.2, 2.3 * len(styles) + 1.3), sharex=True, sharey=True)
    orientations = (("red", "#c94949", "-"), ("blue", "#337ac4", "--"))
    for ci, (condition, label, _) in enumerate(styles):
        for vi, version in enumerate(VERSIONS):
            ax = axes[ci, vi]
            for orientation, color, linestyle in orientations:
                ax.plot(np.arange(1, 21), groups[version, condition]["final_ego_red_given_partner_faster_" + orientation] * 100,
                        color=color, linestyle=linestyle, linewidth=2)
            format_axis(ax, 100, 1)
            ax.text(.99, 1.02, f"Seed {selected[version, condition]['seed']}", transform=ax.transAxes,
                    fontsize=10, va="bottom", ha="right")
            if ci == 0:
                ax.set_title(version, fontsize=14, fontstyle="italic")
            if vi == 0:
                ax.set_ylabel(label + "\nFinal ego-red assignment (%)", fontsize=9)
            if ci == len(styles) - 1:
                ax.set_xlabel("Round within partner episode")
    fig.suptitle("Final assignments by partner ability — selected training seeds", fontsize=16, y=.99)
    handles = [Line2D([], [], color=color, linestyle=ls,
                      label=f"Partner faster at {orientation} (ego-red target: {0 if orientation == 'red' else 100}%)")
               for orientation, color, ls in orientations]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .96), ncol=2, frameon=False, fontsize=9)
    fig.text(.5, .015, "Partner color is complementary to ego assignment. Faster-task matching is descriptive; it does not account for geometry or switching costs.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .915))
    save_figure(fig, out, "final_choice_by_partner_ability_best_training_seeds", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--exclude-no-influence", action="store_true",
                        help="Show the three influence-enabled conditions in a separate output directory")
    args = parser.parse_args()
    records = load_manifests(args.manifest)
    saved, selection_provenance = verified_selection(args.selection, records)
    manifest = records[0][0]
    styles = tuple(s for s in STYLES if not (args.exclude_no_influence and s[0] == "rnn_diverse_noinfluence"))
    conditions = {c for c, _, _ in styles}
    default_out = DEFAULT_OUT / "three_conditions" if args.exclude_no_influence else DEFAULT_OUT
    out = (args.out_dir or default_out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    groups, selected, sources, summaries, round_rows, profile_rows = {}, {}, [], [], [], []
    for row in saved["results"]:
        version, condition, seed = row["version"], row["condition"], row["seed"]
        if condition not in conditions:
            continue
        directory = Path(manifest["evaluation_inputs"][version][str(seed)])
        data, provenance = read_policy(directory, version, condition, seed, "test", out / "cache")
        require(Path(provenance["source"]["path"]).resolve() == Path(row["evaluation_rollouts"]).resolve(),
                "Selected rollout path differs")
        require(provenance["source"]["summary_sha256"] == row["evaluation_summary_sha256"], "Evaluation summary changed")
        for field in ("dones", "round_done", "round_idx"):
            require(provenance["scalar_datasets_sha256"][field] == row["scalar_datasets_sha256"][field],
                    f"Selected rollout data changed: {field}")
        require(np.isclose(provenance["saved_episode_return"], row["mean"], atol=3e-3, rtol=0),
                "Held-out mean differs from the return figure")
        metrics = round_metrics(data)
        metrics["final_ego_blue"] = 1 - metrics["final_ego_red"]
        groups[version, condition], selected[version, condition] = metrics, row
        meta = dict(version=version, condition=condition, seed=seed, population="test")
        sources.append(dict(**meta, **provenance))
        for r in range(20):
            round_rows.append(dict(**meta, round=r + 1, n_episodes=len(data["profile"]),
                                   **{k: float(v[r]) for k, v in metrics.items()}))
        summaries.append(dict(**meta, training_return=row["training_return"],
            episode_return=provenance["saved_episode_return"], n_episodes=len(data["profile"]),
            round_success=provenance["saved_success"],
            **{k: float(metrics[k].mean()) for k in ("initial_ego_red", "final_ego_red", "final_ego_blue", "switches_per_round", "fraction_rounds_switched", "final_partner_faster")},
            switches_per_episode=float(metrics["cumulative_switches"][-1]),
            final_partner_faster_round1=float(metrics["final_partner_faster"][0]),
            final_partner_faster_round20=float(metrics["final_partner_faster"][-1])))
        for pi, (dr, db) in enumerate(data["pool"]):
            pm = round_metrics(data, data["profile"] == pi)
            pm["final_ego_blue"] = 1 - pm["final_ego_red"]
            for r in range(20):
                profile_rows.append(dict(**meta, delay_red=int(dr), delay_blue=int(db), round=r + 1,
                    **{k: float(pm[k][r]) for k in ("initial_ego_red", "final_ego_red", "final_ego_blue", "switches_per_round", "final_partner_faster")}))
        print(f"Read {version}/{condition}: seed {seed}", flush=True)
    write_csv(out / "selected_seed_summary.csv", summaries)
    write_csv(out / "selected_seed_by_round.csv", round_rows)
    write_csv(out / "per_profile_selected_seed_by_round.csv", profile_rows)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}), PdfPages(out / "assignment_behavior_report.pdf") as pdf:
        dashboard(groups, selected, out, pdf, styles)
        dashboard(groups, selected, out, pdf, styles, metrics=METRICS[:2],
                  name="final_assignments_red_blue_best_training_seeds",
                  title="Final red and blue assignments of the selected training seeds")
        for spec in METRICS:
            overview(groups, selected, out, spec, pdf, styles)
        conditioned_choices(groups, selected, out, pdf, styles)
    n_episodes = sum(r["n_episodes"] for r in summaries)
    report = dict(selection_source=selection_provenance, selection=saved["selection"],
        manifests=[p for _, p in records], population="test", selected_policies=summaries,
        included_conditions=[c for c, _, _ in styles],
        sources=sources, definitions=DEFINITIONS,
        source_sha256=sha256(Path(__file__).read_bytes()),
        metric_source_sha256=sha256((ROOT / "eval/analyze_assignment_behaviors.py").read_bytes()),
        validation=dict(policies=len(summaries), episodes=n_episodes, rounds=n_episodes * 20,
                        training_selection_recomputed=True, return_figure_sources_match=True,
                        saved_switch_summaries_verified=True), report_pages=len(METRICS) + 3,
        caveats=["Only the previously selected policy is shown in each version/condition.",
                 "Final training rollout precedes the last gradient update; saved evaluation uses the final checkpoint.",
                 "V1_matched assignments cannot switch within a round.",
                 "No confidence bands are shown, matching the original assignment plots; curves condition on the selected policy."])
    (out / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    seed_table = "| Experiment | " + " | ".join(label for _, label, _ in styles) + " |\n"
    seed_table += "| " + " | ".join(["---"] * (len(styles) + 1)) + " |\n"
    seed_table += "\n".join(f"| {v} | " + " | ".join(str(selected[v, c]["seed"]) for c, _, _ in styles) + " |" for v in VERSIONS)
    condition_note = ("Shown: the three influence-enabled conditions. RNN no influence is excluded.\n\n"
                      if args.exclude_no_influence else
                      "No-influence assignments are imposed by the environment.\n\n")
    reproduction_flag = " --exclude-no-influence" if args.exclude_no_influence else ""
    (out / "METHODS.txt").write_text(
        "Assignment behavior of the best training seeds\n\n"
        "Selection reuses overcooked_paper_match/best_of_5/selection.json, verified\n"
        "against all 120 training logs. Each experiment/condition maximizes mean\n"
        "completed-episode return in the final logged training rollout among seeds\n"
        "1-10 (lowest seed breaks ties). Held-out behavior never enters selection.\n\n"
        + condition_note + seed_table + "\n\n"
        "Population: 22 held-out profiles x 20 episodes x 20 rounds per policy.\n"
        f"{len(summaries)} selected policies, {n_episodes:,} episodes, {n_episodes * 20:,} rounds. These are the same\n"
        "final-checkpoint evaluations used in the best-of-five return figure.\n\n"
        + "\n".join(f"{key}: {value}." for key, value in DEFINITIONS.items()) + "\n\n"
        "V1_matched is fixed within each round. V2 initializes randomly. Influence-enabled V3\n"
        "chooses its initial assignment. Scan padding after termination is excluded.\n"
        "No confidence bands are shown, as in the original assignment figures.\n\n"
        f"Report: assignment_behavior_report.pdf ({len(METRICS) + 3} pages). Page 1 combines final red\n"
        "and blue assignment rows, faster-task matching, and switches. Page 2\n"
        "shows final red and blue assignments alone; pages 3-9 contain individual\n"
        "metric grids. Page 10 conditions final assignments on the partner's\n"
        "faster color. Switch grids show V2/V3, matching the reference layout;\n"
        "V1_matched has zero within-round switches and remains in the dashboard.\n"
        "At each round number, final assignments average all 440 evaluation\n"
        "episodes (22 profiles x 20 episodes). Blue percentage = 100 - red.\n"
        "Each figure is available as PNG, PDF and SVG.\n\n"
        "Data: selected_seed_summary.csv, selected_seed_by_round.csv, and\n"
        "per_profile_selected_seed_by_round.csv. analysis.json records selection,\n"
        "training-log verification, dataset hashes, definitions and source hashes.\n\n"
        "Reproduce from repository root:\n"
        "MPLCONFIGDIR=/tmp/grid-best-assignment-mpl /nlp/scr/jshe/miniconda3/envs/emergent_partner_model/bin/python eval/plot_best_seed_assignment_behaviors.py"
        + reproduction_flag + "\n")
    print(f"Saved report, figures, methods and data: {out}", flush=True)


if __name__ == "__main__":
    main()
