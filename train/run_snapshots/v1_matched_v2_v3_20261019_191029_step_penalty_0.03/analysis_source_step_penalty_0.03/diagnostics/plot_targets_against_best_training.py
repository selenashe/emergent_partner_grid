"""Plot each approved debugging target against its previously selected best seed.

References come from the original best-of-five artifact, selected by unsmoothed
return in the final training rollout. No peer medians or held-out selection.
Targets that are the reference remain self-comparisons in the primary figure;
a separately labeled supplementary figure compares those targets to runner-up
seeds so an informative other-seed contrast is also available.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.scan_return_divergence import HISTORY, LABELS, STEPS, VERSIONS, CONDITIONS, rolling, segments

COHORT = ROOT / "train/seed_diagnostics/v1_matched_v2_v3/seed_selection/representative_seeds.csv"
SELECTION = ROOT / "/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/v1_matched_v2_v3_20261019_191029_step_penalty_0.03/overcooked_paper_match/best_of_5/selection.json"
EXTREMES = COHORT.parent / "endpoint_extremes.csv"
DEFAULT_OUT = COHORT.parent.parent / "return_divergence/best_training_references"
TARGET_COLOR = "#2879b9"
REFERENCE_COLOR = "#d77b26"


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return row["version"], row["condition"], int(row["seed"])


def compare(histories, target, reference, role):
    version, condition, seed = target
    raw_target = histories[target]
    raw_reference = histories[(version, condition, reference)]
    smoothed_target, smoothed_reference = rolling(raw_target, 8), rolling(raw_reference, 8)
    gap = smoothed_target-smoothed_reference
    events = segments(gap, 1., persistence=8)
    lasting = next((event for event in events if event["last_window_end_update"] == 915), None)
    onset = lasting["first_window_end_update"] if lasting else None
    self_comparison = seed == reference
    if self_comparison:
        np.testing.assert_array_equal(raw_target, raw_reference)
        assert not events and np.nanmax(np.abs(gap)) == 0
    if onset is not None:
        sign = 1 if lasting["direction"] == "A higher" else -1
        assert np.all(sign*gap[onset-1:] >= 1.)
        assert onset == 1 or not np.isfinite(gap[onset-2]) or sign*gap[onset-2] < 1.
    row = dict(version=version, condition=condition, target_seed=seed, reference_seed=reference,
        reference_role=role, self_comparison=self_comparison,
        target_final_logged_return=float(raw_target[-1]), reference_final_logged_return=float(raw_reference[-1]),
        target_last32_mean_return=float(np.nanmean(raw_target[-32:])),
        reference_last32_mean_return=float(np.nanmean(raw_reference[-32:])),
        last32_mean_signed_gap=float(np.nanmean(raw_target[-32:])-np.nanmean(raw_reference[-32:])),
        lasting_gap_onset_update=onset, lasting_gap_onset_transitions=onset*STEPS if onset else None,
        gap_direction=("target higher" if lasting["direction"] == "A higher" else "reference higher") if lasting else "none",
        smoothing_updates=8, threshold=1., persistence=8)
    return row, smoothed_target, smoothed_reference


def plot(out, comparisons, name, limit=60, supplemental=False, different_seed=False):
    rows, cols, size = (1, 3, (16, 4.9)) if supplemental else (4, 3, (16, 13))
    fig, axes = plt.subplots(rows, cols, figsize=size, sharex=True, sharey=True, squeeze=False)
    x = np.arange(1, 916)*STEPS/1e6
    for ax, (row, target, reference) in zip(axes.ravel(), comparisons):
        ax.plot(x, target, color=TARGET_COLOR, linewidth=2, zorder=3)
        ax.plot(x, reference, color=REFERENCE_COLOR, linewidth=2, linestyle="--", zorder=4)
        label = "runner-up" if "runner_up" in row["reference_role"] else "best"
        ax.set_title(f"{row['version']} · {LABELS[row['condition']]}\n"
            f"Target seed {row['target_seed']} vs {label} seed {row['reference_seed']}", fontsize=11, pad=8)
        onset = row["lasting_gap_onset_update"]
        if onset is not None and onset*STEPS/1e6 <= limit:
            mark = onset*STEPS/1e6
            ax.axvline(mark, color="#202020", linewidth=1.3, linestyle=":", zorder=5)
            ax.scatter([mark, mark], [target[onset-1], reference[onset-1]],
                color=[TARGET_COLOR, REFERENCE_COLOR], s=20, zorder=6)
            note = f"Lasting gap: u{onset}\n{onset*STEPS:,} transitions"
        elif onset is not None:
            note = f"Lasting gap outside view:\nu{onset} · {onset*STEPS:,} transitions"
        elif row["self_comparison"]:
            note = "Target is the selected best seed\nIdentical curves"
        else:
            note = "No lasting ≥1-unit gap"
        ax.text(.97, .045, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color="#333333", bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=3))
        ax.set_xlim(0, limit);ax.set_ylim(-17, 20);ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:ax.set_xlabel("Training transitions (millions)", fontsize=10)
    for ax in axes[:, 0]:ax.set_ylabel("Completed-episode return", fontsize=10)
    title = ("Best-seed targets versus the next-best training seed" if supplemental else
        "Each debugging target versus the best different training seed" if different_seed else
        "Each debugging target versus its selected best training seed")
    fig.suptitle(title, fontsize=17, y=.995)
    handles = [Line2D([], [], color=TARGET_COLOR, linewidth=2, label="Target seed"),
        Line2D([], [], color=REFERENCE_COLOR, linewidth=2, linestyle="--",
            label="Runner-up reference" if supplemental else "Best different seed" if different_seed else "Selected best reference"),
        Line2D([], [], color="#202020", linewidth=1.3, linestyle=":", label="Lasting-gap onset, when present")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .965 if not supplemental else .92), ncol=3, frameon=False, fontsize=10)
    caption = ("Supplement only: runner-up references are explicitly separate from the primary best-seed comparisons.\n" if supplemental else
        "Reference: original best-of-five seed; the three best-seed targets use the runner-up, explicitly labeled. No self-comparisons or medians.\n" if different_seed else
        "References are the previously selected best-of-five seeds by raw final training return (update 915); no medians.\n")
    caption += "8-update trailing means (≥6 finite returns) · marker: ≥1-unit gap continuing through update 915 · detection time, not a proven causal breakpoint"
    fig.text(.5, .012, caption, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06 if not supplemental else .13, 1, .94 if not supplemental else .87), h_pad=1.6, w_pad=1.8)
    for ext in ("png", "pdf"):
        fig.savefig(out/f"{name}.{ext}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args=parser.parse_args();out=args.out_dir.resolve();out.mkdir(parents=True, exist_ok=True)
    targets=read_csv(COHORT)
    assert len(targets)==12 and len({key(r) for r in targets})==12
    assert all(r["condition"] in CONDITIONS for r in targets)
    targets.sort(key=lambda r: (VERSIONS.index(r["version"]), CONDITIONS.index(r["condition"]), int(r["seed"])))
    selection=json.loads(SELECTION.read_text())
    best={(r["version"],r["condition"]):r for r in selection["results"] if r["condition"] in CONDITIONS}
    recorded={(r["version"],r["condition"]):r for r in read_csv(EXTREMES)}
    source=HISTORY/"updates.csv"
    audit=json.loads((COHORT.parent/"audit.json").read_text())
    assert sha(source)==audit["source_sha256"][str(source)]
    families={key(r)[:2] for r in targets}
    histories={(v,c,s):np.full(915,np.nan) for v,c in families for s in range(1,6)}
    seen={k:[] for k in histories}
    for r in read_csv(source):
        identity=key(r)
        if identity not in histories:continue
        u=int(r["update"]);assert int(r["env_steps"])==u*STEPS
        histories[identity][u-1]=float(r["episode_return"]);seen[identity].append(u)
    assert all(sorted(u)==list(range(1,916)) for u in seen.values())
    rankings={}
    for family in families:
        ranked=sorted(range(1,6), key=lambda seed: (-histories[(*family,seed)][-1], seed))
        rankings[family]=ranked
        assert ranked[0]==int(best[family]["seed"])==int(recorded[family]["highest_seed"])
        assert histories[(*family,ranked[0])][-1]==float(best[family]["training_return"])==float(recorded[family]["highest_return"])
    comparisons=[];supplement=[];curves=[]
    for target in targets:
        identity=key(target);family=identity[:2];reference=int(best[family]["seed"])
        result=compare(histories,identity,reference,"previously_selected_best_of_5")
        comparisons.append(result)
        row,curve_target,curve_reference=result
        for u in range(915):
            curves.append(dict(version=identity[0],condition=identity[1],target_seed=identity[2],reference_seed=reference,
                update=u+1,transitions=(u+1)*STEPS,raw_target=histories[identity][u],raw_reference=histories[(*family,reference)][u],
                smoothed_target=curve_target[u],smoothed_reference=curve_reference[u],signed_gap=curve_target[u]-curve_reference[u]))
        if row["self_comparison"]:
            supplement.append(compare(histories,identity,rankings[family][1],"supplementary_runner_up_by_final_training_return"))
    assert len(comparisons)==12 and len(supplement)==3
    write_csv(out/"target_reference_pairs.csv",[r[0] for r in comparisons])
    write_csv(out/"runner_up_comparisons.csv",[r[0] for r in supplement])
    write_csv(out/"curve_values.csv",curves)
    markers=[r[0] for r in comparisons if r[0]["lasting_gap_onset_update"] is not None]
    write_csv(out/"onset_markers.csv",markers)
    plot(out,comparisons,"targets_vs_best_full")
    plot(out,comparisons,"targets_vs_best_first_10M",limit=10)
    plot(out,supplement,"best_targets_vs_runner_up",supplemental=True)
    replacements={(r[0]["version"],r[0]["condition"],r[0]["target_seed"]):r for r in supplement}
    different=[]
    for comparison in comparisons:
        row=comparison[0]
        identity=(row["version"],row["condition"],row["target_seed"])
        if row["self_comparison"]:
            replacement=replacements[identity]
            different.append(({**replacement[0],"reference_role":"runner_up_because_target_is_selected_best"},replacement[1],replacement[2]))
        else:
            different.append(comparison)
    assert len(different)==12 and all(not r[0]["self_comparison"] for r in different)
    write_csv(out/"target_reference_other_seeds.csv",[r[0] for r in different])
    other_curves=[]
    for row, target_curve, reference_curve in different:
        identity=(row["version"],row["condition"],row["target_seed"])
        reference_identity=(*identity[:2],row["reference_seed"])
        for u in range(915):
            other_curves.append(dict(version=identity[0],condition=identity[1],target_seed=identity[2],reference_seed=row["reference_seed"],
                update=u+1,transitions=(u+1)*STEPS,raw_target=histories[identity][u],raw_reference=histories[reference_identity][u],
                smoothed_target=target_curve[u],smoothed_reference=reference_curve[u],signed_gap=target_curve[u]-reference_curve[u]))
    write_csv(out/"curve_values_other_seeds.csv",other_curves)
    plot(out,different,"targets_vs_best_other_full",different_seed=True)
    plot(out,different,"targets_vs_best_other_first_10M",limit=10,different_seed=True)
    result=dict(targets=12,primary_comparisons=12,self_comparisons=3,supplementary_runner_up_comparisons=3,
        selection_rule="Use the original best-of-five selection by raw completed-episode return at update 915 within the same version and condition.",
        original_selection_verified_against_all_10_final_scores=True,all_915_updates_checked=True,
        no_peer_medians=True,no_training_launched=True,
        marker_rule="8-update trailing mean, >=6 finite returns; signed gap >=1 unit in one direction continuously through update 915; at least 8 endpoints.",
        self_comparison_rule="Keep the selected best seed as primary reference; show identical curves. A separately labeled supplement uses the second-ranked seed.",
        source_sha256={str(p):sha(p) for p in (Path(__file__),COHORT,SELECTION,EXTREMES,source,ROOT/"diagnostics/scan_return_divergence.py")},
        pairs=[r[0] for r in comparisons],runner_up_pairs=[r[0] for r in supplement],
        different_seed_pairs=[r[0] for r in different])
    (out/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(pairs=result["pairs"],runner_up_pairs=result["runner_up_pairs"],output=str(out)),indent=2))


if __name__=="__main__":
    main()
