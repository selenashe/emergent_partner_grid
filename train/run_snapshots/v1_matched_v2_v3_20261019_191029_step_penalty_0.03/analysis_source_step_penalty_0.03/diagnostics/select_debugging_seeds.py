"""Rank final training updates and propose a behavior-diverse debugging cohort.

Read existing logs and held-out partner rollouts. Do not launch training, select
on a smoothed endpoint, or include the RNN-without-influence condition.
"""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_path

VERSIONS = ("v1_matched", "v2", "v3")
CONDITIONS = ("mlp_diverse_influence", "rnn_diverse_influence", "rnn_single_influence")
LABELS = dict(zip(CONDITIONS, ("MLP", "RNN-multi", "RNN-single")))
HISTORY = ROOT / "/juice6/u/jshe/emergent_partner_grid/train/training_curves/v1_matched_v2_v3_20261019_191029_step_penalty_0.03"
BEHAVIOR = ROOT / "/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/v1_matched_v2_v3_20261019_191029_step_penalty_0.03/assignment_behaviors_all_seeds/novel"
DEFAULT_OUT = ROOT / "train/seed_diagnostics/v1_matched_v2_v3/seed_selection"

# Purposive mechanistic examples, not population estimates or a performance
# comparison selected independently of the held-out data.
PROTOTYPES = (
    ("red_commitment", "Red commitment", "v1_matched", "mlp_diverse_influence", 2,
     "Chooses ego red across partner capabilities and rounds.",
     "When does color commitment emerge, and why does the opposite commitment also learn?"),
    ("blue_commitment", "Blue commitment", "v1_matched", "mlp_diverse_influence", 10,
     "Strong ego-blue preference, with little matching to partner ability.",
     "Does blue preference arise through the same training mechanism as red preference?"),
    ("across_round_adaptation", "Adaptation across rounds", "v1_matched", "rnn_diverse_influence", 1,
     "Partner-faster matching increases across the episode; assignments are fixed within each round.",
     "Which updates produce useful use of partner history?"),
    ("mixed_without_matching", "Mixed choices without partner matching", "v1_matched", "rnn_diverse_influence", 2,
     "Mixed red/blue choices, chance-level partner-faster matching, and poorer completion.",
     "Why does apparently balanced assignment fail to become partner-dependent?"),
    ("online_adaptation", "Adaptation with online correction", "v2", "rnn_diverse_influence", 10,
     "High partner-faster matching despite random initial assignments, with within-round switches.",
     "How are useful corrections learned instead of repeated switching?"),
    ("repeated_switching", "Repeated switching with poor completion", "v2", "rnn_single_influence", 8,
     "Many within-round switches, poor held-out completion, and negative mean return.",
     "At which updates does switching become persistent, and does it cause the return loss?"),
    ("fragile_transfer", "Intermittent switching bursts", "v3", "rnn_single_influence", 7,
     "Mostly stable blue assignment, but switching bursts occur in a minority of held-out rounds despite near-top training reward.",
     "When does an otherwise stable policy enter a switching burst on held-out partners?"),
)
ADDITIONAL = (
    ("required_rnn_blue_outlier", "v1_matched", "rnn_diverse_influence", 4,
     "Explicit user-requested RNN outlier: compare blue commitment with the MLP blue prototype and adaptive RNN seed 1."),
    ("lowest_training_endpoint", "v1_matched", "mlp_diverse_influence", 8,
     "Lowest final training return in the included 90 policies; poor completion without any within-round switching."),
    ("learned_initialization_switching", "v3", "rnn_diverse_influence", 4,
     "Worst v3 RNN-multi endpoint: contrast repeated switching under learned initialization with v2's random initialization."),
    ("transfer_control", "v3", "rnn_single_influence", 6,
     "Control for fragile-transfer seed 7: nearly identical final training return, better held-out completion, and few switches."),
    ("best_v3_rnn_without_matching", "v3", "rnn_diverse_influence", 10,
     "Highest final v3 RNN-multi training return, but strong red commitment and low partner-faster matching; control for seed 4."),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return row["version"], row["condition"], int(row["seed"])


def load_training():
    expected = {(v, c, s) for v in VERSIONS for c in CONDITIONS for s in range(1, 6)}
    histories = {k: [] for k in expected}
    for row in read_csv(HISTORY / "updates.csv"):
        if key(row) in expected:
            histories[key(row)].append(row)
    provenance = {key(r): r for r in json.loads((HISTORY / "summary.json").read_text())["runs"] if key(r) in expected}
    assert set(provenance) == expected
    records = {}
    for identity, rows in histories.items():
        rows.sort(key=lambda r: int(r["update"]))
        assert [int(r["update"]) for r in rows] == list(range(1, 916)), identity
        last = rows[-1]
        assert int(last["env_steps"]) == 59965440
        # Validate rankings against original stdout, not only derived tables.
        log = resolve_path(provenance[identity]["log"])
        raw = log.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == provenance[identity]["log_sha256"], log
        config = resolve_path(provenance[identity]["config"])
        assert sha(config) == provenance[identity]["config_sha256"], config
        lines = [line for line in raw.decode().splitlines() if re.match(r"^\[u\s+915\]", line)]
        assert len(lines) == 1
        for field, value in (("ret", last["episode_return"]), ("rsucc", last["round_success"])):
            match = re.search(rf"\b{field}=\s*([+\-\d.]+)", lines[0])
            assert match and float(match[1]) == float(value), identity
        records[identity] = dict(version=identity[0], condition=identity[1], seed=identity[2],
            final_update=915, final_transitions=59965440,
            final_training_return=float(last["episode_return"]),
            final_training_success=float(last["round_success"]),
            trailing46_training_return=float(np.mean([float(r["episode_return"]) for r in rows[-46:]])),
            training_log=str(log), training_log_sha256=provenance[identity]["log_sha256"],
            training_config=str(config), training_config_sha256=provenance[identity]["config_sha256"])
    return records


def load_behavior(training):
    analysis = json.loads((BEHAVIOR / "analysis.json").read_text())
    sources = {key(r): r for r in analysis["sources"] if key(r) in training}
    summaries = {key(r): r for r in read_csv(BEHAVIOR / "per_seed_summary.csv") if key(r) in training}
    assert set(sources) == set(summaries) == set(training)
    curves, orientation_rows, provenance = {}, [], []
    for identity, result in training.items():
        version, condition, seed = identity
        stem = BEHAVIOR / "cache" / f"{version}_{condition}_seed{seed}"
        metadata = json.loads(stem.with_suffix(".json").read_text())
        source = metadata["source"]
        assert source == sources[identity]["source"]
        raw_path = resolve_path(source["path"])
        stat = raw_path.stat()
        assert (stat.st_size, stat.st_mtime_ns) == (source["size"], source["mtime_ns"])
        assert sha(resolve_path(source["summary"])) == source["summary_sha256"]
        saved = json.loads(resolve_path(source["summary"]).read_text())["test"]
        with np.load(stem.with_suffix(".npz")) as f:
            a = {name: np.asarray(f[name]) for name in f}
        first, final, switches, steps = (a[name] for name in ("initial_ego_red", "final_ego_red", "switches", "round_steps"))
        assert first.shape == final.shape == switches.shape == steps.shape == (440, 20)
        assert a["pool"].shape == (22, 2)
        np.testing.assert_array_equal(np.sort(a["profile"]), np.repeat(np.arange(22), 20))
        delays = a["pool"][a["profile"]]
        red_faster = delays[:, 0] < delays[:, 1]
        blue_faster = delays[:, 1] < delays[:, 0]
        assert red_faster.sum() == blue_faster.sum() == 220
        matching = final == blue_faster[:, None]
        ends = np.cumsum(steps, axis=1) - 1
        # Independently read successes at the cached terminal transition indices.
        with h5py.File(raw_path) as f:
            success_raw = np.asarray(f["success"])
        success = success_raw[np.arange(440)[:, None], ends].astype(bool)
        observed_success = float(success.mean())
        observed_return = float((1.01 * success.sum(axis=1) - .01 * steps.sum(axis=1)).mean())
        assert np.isclose(observed_success, saved["round_success_rate"], atol=1e-10)
        assert np.isclose(observed_return, saved["mean_ep_return"], rtol=0, atol=3e-5)
        if version == "v1_matched":
            assert not switches.any()
        metrics = dict(heldout_return=float(saved["mean_ep_return"]), heldout_success=observed_success,
            heldout_mean_round_steps=float(steps.mean()), initial_ego_red=float(first.mean()),
            final_ego_red=float(final.mean()), final_ego_blue=float(1-final.mean()),
            partner_faster=float(matching.mean()), partner_faster_round1=float(matching[:, 0].mean()),
            partner_faster_round20=float(matching[:, -1].mean()),
            partner_faster_late5=float(matching[:, -5:].mean()),
            switches_per_round=float(switches.mean()), fraction_rounds_switched=float((switches > 0).mean()),
            ego_red_when_partner_red_faster=float(final[red_faster].mean()),
            ego_red_when_partner_blue_faster=float(final[blue_faster].mean()))
        switched = switches > 0
        metrics.update(mean_switches_given_switched=float(switches[switched].mean()) if switched.any() else 0.,
            heldout_success_switched=float(success[switched].mean()) if switched.any() else None,
            heldout_success_unswitched=float(success[~switched].mean()) if (~switched).any() else None,
            fraction_rounds_10plus_switches=float((switches >= 10).mean()))
        for field, summary_field in (("final_ego_red", "final_ego_red"), ("partner_faster", "final_partner_faster"),
            ("switches_per_round", "switches_per_round"), ("fraction_rounds_switched", "fraction_rounds_switched")):
            assert np.isclose(metrics[field], float(summaries[identity][summary_field]), atol=1e-12)
        metrics["ability_conditioned_red_difference"] = metrics["ego_red_when_partner_blue_faster"] - metrics["ego_red_when_partner_red_faster"]
        red = metrics["final_ego_red"]
        metrics["color_profile"] = "red-biased" if red >= .8 else "blue-biased" if red <= .2 else "mixed"
        score = metrics["partner_faster"]
        metrics["partner_matching_profile"] = "high" if score >= .8 else "partial" if score > .6 else "low"
        n = metrics["switches_per_round"]
        metrics["switch_profile"] = "none" if n == 0 else "occasional" if n < 2 else "moderate" if n < 5 else "frequent"
        result.update(metrics)
        curves[identity] = dict(red=final.mean(axis=0), matching=matching.mean(axis=0), switches=switches.mean(axis=0))
        for label, mask in (("partner_red_faster", red_faster), ("partner_blue_faster", blue_faster)):
            orientation_rows.append(dict(version=version, condition=condition, seed=seed, orientation=label,
                heldout_return=float((1.01*success[mask].sum(axis=1)-.01*steps[mask].sum(axis=1)).mean()),
                heldout_success=float(success[mask].mean()), switches_per_round=float(switches[mask].mean()),
                final_ego_red=float(final[mask].mean()), partner_faster=float(matching[mask].mean())))
        provenance.append(dict(version=version, condition=condition, seed=seed, source=source,
            cache_sha256=sha(stem.with_suffix(".npz")), cache_metadata_sha256=sha(stem.with_suffix(".json"))))
    return curves, orientation_rows, provenance


def select(records):
    extrema = []
    for version in VERSIONS:
        for condition in CONDITIONS:
            group = [r for k, r in records.items() if k[:2] == (version, condition)]
            ranked = sorted(group, key=lambda r: (-r["final_training_return"], r["seed"]))
            for rank, row in enumerate(ranked, 1):
                row["final_training_return_rank"] = rank
            high, low = ranked[0], ranked[-1]
            extrema.append(dict(version=version, condition=condition,
                highest_seed=high["seed"], highest_return=high["final_training_return"],
                lowest_seed=low["seed"], lowest_return=low["final_training_return"]))
    selected = []
    for category, label, version, condition, seed, description, question in PROTOTYPES:
        selected.append(dict(role="prototype", category=category, label=label, description=description,
            debugging_question=question, **records[(version, condition, seed)]))
    for category, version, condition, seed, reason in ADDITIONAL:
        selected.append(dict(role="additional contrast", category=category, label=category.replace("_", " "),
            description=reason, debugging_question=reason, **records[(version, condition, seed)]))
    assert len({key(r) for r in selected}) == 12
    return extrema, selected


def plot(out, selected):
    y = np.arange(len(selected))
    fig, axes = plt.subplots(1, 4, figsize=(15, 8), sharey=True, gridspec_kw={"width_ratios": [1, 1.25, 1, 1]})
    axes[0].barh(y, [100*r["final_ego_red"] for r in selected], color="#ba5656", height=.55)
    axes[0].set_yticks(y, [f"{r['version']} · {LABELS[r['condition']]} · s{r['seed']}" for r in selected], fontsize=9)
    axes[0].set_title("Ego-red assignment (%)");axes[0].set_xlim(0, 100)
    for i, r in enumerate(selected):
        early, late = 100*r["partner_faster_round1"], 100*r["partner_faster_round20"]
        axes[1].plot([early, late], [i, i], color="#8d929c", linewidth=1.5)
        axes[1].scatter(early, i, color="#999999", s=25, label="Round 1" if i == 0 else None)
        axes[1].scatter(late, i, color="#3469b1", s=30, label="Round 20" if i == 0 else None)
    axes[1].set_title("Partner on faster task (%)");axes[1].set_xlim(0, 100);axes[1].axvline(50, color="#aaaaaa", linestyle=":")
    axes[1].legend(loc="lower left", frameon=False, fontsize=8)
    axes[2].barh(y, [r["switches_per_round"] for r in selected], color="#8c6bb1", height=.55)
    axes[2].set_title("Switches per round");axes[2].set_xlim(0, 29)
    axes[3].barh(y, [r["heldout_return"] for r in selected], color="#369680", height=.55)
    axes[3].set_title("Held-out episode return");axes[3].set_xlim(-5, 20);axes[3].axvline(0, color="#aaaaaa", linewidth=.6)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=.15);ax.set_axisbelow(True);ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Proposed debugging cohort · held-out partner behavior", fontsize=16)
    fig.text(.5, .01, "22 held-out partner profiles × 20 episodes × 20 rounds per policy. Faster-task matching is descriptive; it is not a full task-optimality score.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(out/"representative_behaviors.png", dpi=180)
    fig.savefig(out/"representative_behaviors.pdf")
    plt.close(fig)


def report(out, extrema, selected):
    endpoints = "".join(f"<tr><td>{r['version']}</td><td>{LABELS[r['condition']]}</td><td>{r['highest_seed']}</td>"
        f"<td>{r['highest_return']:.3f}</td><td>{r['lowest_seed']}</td><td>{r['lowest_return']:.3f}</td></tr>" for r in extrema)
    profiles = "".join(f"<tr><td>{html.escape(r['label'])}</td><td>{r['version']} / {LABELS[r['condition']]} / seed {r['seed']}</td>"
        f"<td>{html.escape(r['description'])}</td><td>{100*r['final_ego_red']:.2f}%</td><td>{100*r['partner_faster']:.2f}%</td>"
        f"<td>{r['switches_per_round']:.2f}</td><td>{100*r['heldout_success']:.2f}%</td><td>{r['heldout_return']:.3f}</td></tr>" for r in selected)
    doc=f'''<!doctype html><meta charset="utf-8"><title>Debugging seed selection</title>
<style>body{{font:16px/1.55 system-ui;max-width:1400px;margin:36px auto;padding:0 24px;color:#202634}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ccd3dd;padding:7px;text-align:left}}img{{width:100%;height:auto}}</style>
<h1>Behavior-diverse debugging seeds</h1>
<p>All 90 policies from v1_matched/v2/v3 × MLP/RNN-multi/RNN-single × seeds 1–5. RNN without influence is excluded.</p>
<p>Color direction: the plot is ego-red assignment. V1_matched MLP seed 2 chooses red 100% of the time; seed 10 chooses blue 83.61% of the time.</p>
<h2>Highest and lowest final training return</h2>
<p>Rank by the unsmoothed mean completed-episode return in the final logged rollout, update 915 at 59,965,440 transitions.
This rollout precedes the last optimizer update. Rankings are not final-checkpoint evaluations and are not rankings by success alone.
Single-partner training uses one capability profile; diverse training uses 24, so compare ranks within a version/condition.
Very close endpoint scores are not evidence of a meaningful difference in expected performance. The all-policy CSV also reports the trailing-46-update mean.</p>
<table><tr><th>Version</th><th>Condition</th><th>Highest seed</th><th>Return</th><th>Lowest seed</th><th>Return</th></tr>{endpoints}</table>
<h2>Seven descriptive profiles and five additional contrasts</h2>
<p>Categories are descriptive, can overlap, and do not assert a training cause. Mixed red/blue frequency does not establish random action choice;
50% aggregate red does not establish adaptation. Partner-conditioned assignment and its course across rounds distinguish those possibilities.</p>
<table><tr><th>Profile / role</th><th>Example</th><th>Description</th><th>Ego red</th><th>Partner-faster matching</th><th>Switches/round</th><th>Success</th><th>Return</th></tr>{profiles}</table>
<img src="representative_behaviors.png" alt="Assignment bias, adaptation across rounds, switches, and return for the twelve proposed debugging policies">
<h2>Explicit outlier and transfer contrasts</h2>
<p>V1_matched RNN-multi seed 4 is 99.84% ego-blue and 50.11% partner-faster matching, with 98.86% success. It completes rounds reliably,
but its assignment is nearly independent of partner capability. Seeds 6 and 7 in this condition have a similar near-all-blue phenotype;
seed 4 is not the only member. Compare it with adaptive seed 1 and mixed-choice seed 2.</p>
<p>No v3 RNN-multi seed has high partner-faster matching: all five lie between 42.50% and 49.82% overall.
The best v3 RNN-multi training seed, seed 10, chooses ego red 96.25% of the time and matches partner speed 46.55% of the time.
Include it alongside seed 4 to distinguish efficient color commitment from repeated switching under the same protocol.</p>
<p>V3 RNN-single seeds 6 and 7 have final training returns 17.446 and 17.445. On held-out partners, seed 6 has return 14.020,
99.10% success, and 0.165 switches/round; seed 7 has return 11.591, 91.91% success, and 2.532 switches/round.
This controls for a strong training endpoint while exposing transfer differences. Seed 7 success is 99.20% when the partner is red-faster
and 84.61% when it is blue-faster; these are descriptive observed differences, not causal effects of orientation.</p>
<p>Seed 7 switches in only 21.41% of rounds, but averages 11.83 switches when it does switch. Its observed success is
99.77% on rounds without switches and 63.06% on switched rounds. In contrast, v2 RNN-single seed 8 switches in 99.67%
of rounds. This distinguishes intermittent bursts from persistent switching. These conditional associations do not prove
switches cause failures: difficulty may also provoke switching.</p>
<h2>Evidence and scope</h2><p>Held-out evaluation uses 22 novel partner profiles on familiar corpus layouts: 440 episodes and 8,800 rounds per policy.
All final training returns are checked against original stdout hashes and lines. Held-out success is independently reread from terminal transitions;
return is checked against the task reward identity and saved summaries. Cache provenance and assignment/switch summary agreement are checked.</p>
<p>This is a purposively selected mechanistic debugging cohort. It is not an unbiased performance estimate for all seeds, and no new training was launched.</p>
<p><a href="all_seed_profiles.csv">All 90 profiles and final-step ranks</a> · <a href="endpoint_extremes.csv">Nine highest/lowest pairs</a> ·
<a href="representative_seeds.csv">Twelve proposed targets and debugging questions</a> · <a href="partner_orientation.csv">Orientation splits</a> ·
<a href="audit.json">Validation and provenance</a></p>'''
    (out/"report.html").write_text(doc)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args=parser.parse_args();out=args.out_dir.resolve();out.mkdir(parents=True, exist_ok=True)
    records=load_training()
    curves, orientations, provenance=load_behavior(records)
    extrema, selected=select(records)
    rows=[records[k] for k in sorted(records)]
    write_csv(out/"all_seed_profiles.csv", rows)
    write_csv(out/"endpoint_extremes.csv", extrema)
    write_csv(out/"representative_seeds.csv", selected)
    write_csv(out/"partner_orientation.csv", orientations)
    plot(out, selected);report(out, extrema, selected)
    audit=dict(policies=45, conditions=list(CONDITIONS), excluded_condition="rnn_diverse_noinfluence",
        final_training_update=915, final_training_transitions=59965440, training_metric="unsmoothed final logged completed-episode mean return",
        heldout_episodes_per_policy=440, heldout_rounds_per_policy=8800,
        all_training_log_and_config_hashes_and_final_scores_checked=True, all_heldout_successes_and_reward_identities_checked=True,
        all_cache_source_provenance_checked=True, assignment_and_switch_summaries_checked=True,
        categories_are_descriptive=True, new_training_launched=False,
        selected=[dict(role=r["role"], category=r["category"], version=r["version"], condition=r["condition"], seed=r["seed"]) for r in selected],
        tagging_rules=dict(color="red >=80%, blue <=20%, otherwise mixed", partner_matching="high >=80%, partial >60%, otherwise low",
            switches="none=0, occasional <2, moderate <5, frequent >=5 switches/round; heuristic descriptors only"),
        source_sha256={str(p):sha(p) for p in (Path(__file__), HISTORY/"updates.csv", HISTORY/"summary.json", BEHAVIOR/"analysis.json", BEHAVIOR/"per_seed_summary.csv")},
        rollout_provenance=provenance)
    (out/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(dict(endpoint_extremes=extrema, proposed_targets=audit["selected"], report=str(out/"report.html")), indent=2))


if __name__=="__main__":
    main()
