"""Compare training or held-out partner returns with validated oracle ceilings."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.plot_episode_return_dots import CONDITION_STYLES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", choices=("train", "test"), default="train")
    args = parser.parse_args()
    heldout = args.population == "test"
    directory = "heldout_return_ceiling" if heldout else "training_return_ceiling"
    out = Path('/juice6/u/jshe/emergent_partner_grid') / "eval" / directory / 'v1_matched_v2_v3_20261019_191029_step_penalty_0.03'
    summary = json.loads((out / "summary.json").read_text())
    validation = json.loads((out / "validation.json").read_text())
    if not summary["complete_corpus"] or not validation["passed"]:
        raise ValueError("Ceilings must cover the entire corpus and pass environment validation")
    if summary.get("population", "train") != args.population or validation.get("population", "train") != args.population:
        raise ValueError("Oracle results and validation must use the plotted population")
    if heldout:
        source = ROOT / "/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/v1_matched_v2_v3_20261019_191029_step_penalty_0.03/all_5_seeds_step_penalty_0.03/novel/comparison.json"
        report = json.loads(source.read_text())
        if report["population"] != "test" or report["seeds"] != list(range(1, 6)):
            raise ValueError("Expected the completed ten-seed held-out evaluation")
        observed = {(r["version"], r["condition"]): r for r in report["aggregates"]}
    else:
        source = ROOT / "/juice6/u/jshe/emergent_partner_grid/train/training_curves/v1_matched_v2_v3_20261019_191029_step_penalty_0.03/final_update_dots/aggregates.csv"
        with source.open() as file:
            observed = {(r["version"], r["condition"]): r for r in csv.DictReader(file)}
    ceilings = {(r["version"], r["condition"]): r for r in summary["summaries"]}
    lower = min(float(observed[key]["episode_return_mean"])-float(observed[key]["episode_return_sd"])
                for key in ceilings)
    rows = []
    with plt.rc_context({"font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(12.3, 5.3), sharey=True)
        fig.subplots_adjust(left=.075, right=.985, bottom=.18, top=.70, wspace=.16)
        for col, (version, ax) in enumerate(zip(("v1_matched", "v2", "v3"), axes)):
            for x, (condition, _, color) in enumerate(CONDITION_STYLES):
                empirical = observed[(version, condition)]
                mean, sd = float(empirical["episode_return_mean"]), float(empirical["episode_return_sd"])
                ceiling = ceilings[(version, condition)]["oracle_mean_episode_return"]
                ax.errorbar(x, mean, yerr=sd, fmt="o", markersize=11, color=color,
                            markeredgewidth=0, elinewidth=2.8, capsize=0, zorder=3)
                ax.plot(x, ceiling, marker="D", markersize=9, markerfacecolor="white",
                        markeredgecolor=color, markeredgewidth=2, zorder=4)
                information_bound = (summary["mlp_blind_initialization_upper_bounds"][version]
                                     if condition == "mlp_diverse_influence" and version in ("v1_matched", "v3")
                                     else None)
                if information_bound is not None and version == "v1_matched":
                    ax.plot(x, information_bound, marker="v", markersize=10, color=color, zorder=4)
                prefix = "heldout" if heldout else "final_update"
                rows.append({"version": version, "condition": condition,
                             prefix + "_return_mean": mean, prefix + "_return_sd": sd,
                             "full_information_corpus_ceiling": ceiling,
                             "corpus_ceiling_minus_" + prefix + "_mean": ceiling-mean,
                             "mlp_initial_information_upper_bound": information_bound})
            ax.axhline(20, color="#999999", lw=1, ls="--")
            ax.set_title(version, fontsize=16, fontweight="bold", pad=15)
            ax.set_xlim(-.5, 3.5)
            ax.set_yticks([6, 9, 12, 15, 18, 20])
            ax.set_ylim(min(5.5, lower-.6), 20.6)
            ax.set_xticks([])
            for spine in ("top", "right", "bottom"):
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_visible(col == 0)
            ax.tick_params(axis="y", left=col == 0, labelleft=col == 0, colors="#333333")
            if col == 0:
                ax.set_ylabel("Average episode return", labelpad=10)
        legend = [Line2D([], [], marker="o", linestyle="none", markersize=10,
                         markeredgewidth=0, color=color, label=label)
                  for _, label, color in CONDITION_STYLES]
        fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(.985, .855), ncol=4,
                   frameon=True, facecolor="#eeeeee", edgecolor="none", fontsize=11,
                   handlelength=1, handletextpad=.5, columnspacing=1.2)
        title = "Held-out partner returns and oracle ceilings" if heldout else "Final training return and oracle ceilings"
        subtitle = ("Oracle: same 1,096 grid layouts · all 22 held-out profiles · knows capabilities" if heldout else
                    "Oracle: uniform training corpus and each condition's partner population; knows capabilities")
        fig.text(.075, .95, title, fontsize=18, va="top", weight="bold")
        fig.text(.075, .905, subtitle,
                 fontsize=10.5, va="top", color="#555555")
        caption = ("Circles: held-out episode-return mean ± seed SD. Open diamonds: full-information corpus ceiling. Dashed line: 20."
                   if heldout else
                   "Circles: final-update mean ± seed SD. Open diamonds: full-information corpus ceiling. Dashed line: 20.")
        note = ("Triangle: V1_matched MLP initial-information bound. Oracle averages the corpus; evaluations sample layout sequences."
                if heldout else
                "Triangle: V1_matched MLP initial-information bound. The final update's episode mix was not saved.")
        fig.text(.075, .085, caption,
                 fontsize=9.5, color="#555555")
        fig.text(.075, .045, note,
                 fontsize=9.5, color="#555555")
        name = "heldout_return_and_ceiling" if heldout else "training_return_and_ceiling"
        for extension in ("png", "pdf", "svg"):
            fig.savefig(out / f"{name}.{extension}", dpi=240, facecolor="white")
        plt.close(fig)
    with (out / "observed_and_ceiling.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "plot_provenance.json").write_text(json.dumps({
        "population": args.population, "observed_source": str(source),
        "oracle_summary": str(out / "summary.json"), "validation": str(out / "validation.json"),
        "oracle_weighting": summary["averaging"], "observed_seeds": list(range(1, 6)),
    }, indent=2) + "\n")
    print(out / f"{name}.png")


if __name__ == "__main__":
    main()
