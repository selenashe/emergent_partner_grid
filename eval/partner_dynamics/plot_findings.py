"""Readable figures from completed summaries; does not rerun or refit analyses.

Run from the repo root with the experiment Python:
    python -m eval.partner_dynamics.plot_findings --results PATH
Every figure is exported as PNG/PDF alongside its plotted numerical values.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .numeric import grouped_ci


PROTOCOLS = ["fixed_v1", "online_v2"]
PROTOCOL_LABELS = ["Fixed allocation (v1)", "Online allocation (v2)"]
MAIN = "rnn_diverse_influence"
CONDITIONS = [MAIN, "rnn_single_influence", "rnn_diverse_noinfluence"]
LABELS = ["Diverse partners\n+ influence", "One training partner\n+ influence", "Diverse partners\nwithout influence"]
COLORS = ["#1976A3", "#6B7280", "#C67830"]
PROTOCOL_COLORS = ["#1976A3", "#8753A8"]


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#CBD5E1",
        "text.color": "#182A3B", "axes.labelcolor": "#182A3B",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def finish(fig, out, name, title, subtitle, footer):
    fig.suptitle(title, x=.04, y=.975, ha="left", fontsize=19, weight="bold")
    fig.text(.04, .91, subtitle, fontsize=11, color="#475569", va="top")
    fig.text(.04, .025, footer, fontsize=9.2, color="#475569", va="bottom", linespacing=1.6)
    fig.savefig(out / f"{name}.png", dpi=180)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def errorbar(ax, ci, y, color, scale=1):
    mean, low, high = [scale * ci[k] for k in ("mean", "low", "high")]
    ax.errorbar(mean, y, xerr=np.array([[mean-low], [high-mean]]), fmt="o",
                color=color, ms=8, capsize=4, lw=2, zorder=5)


def capability(root, out):
    table = pd.read_csv(root / "policy_summary.csv")
    novel = table[table.scope == "familiar_to_novel"].copy()
    assert len(novel) == 30
    summary = json.loads((root / "aggregate_summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.19, right=.96, top=.78, bottom=.24, wspace=.12)
    rows = []
    for ax, protocol, label in zip(axes, PROTOCOLS, PROTOCOL_LABELS):
        baseline = float(novel[novel.protocol == protocol].prior_mae.iloc[0])
        ax.axvspan(.65, baseline, color="#EFF8F1", zorder=0)
        ax.axvline(baseline, color="#748275", ls="--", lw=1.5)
        for index, (condition, color) in enumerate(zip(CONDITIONS, COLORS)):
            y = 2-index
            part = novel[(novel.protocol == protocol) & (novel.condition == condition)].sort_values("seed")
            assert len(part) == 5 and np.allclose(part.prior_mae, baseline)
            record = next(r for r in summary if r["protocol"] == protocol and r["condition"] == condition and r["scope"] == "familiar_to_novel")
            improvement = record["improvement"]
            ci = dict(mean=record["mae"], low=baseline-improvement["high"], high=baseline-improvement["low"])
            ax.scatter(part.mae, y+np.linspace(-.13, .13, 5), s=27, color=color, alpha=.48, zorder=3)
            errorbar(ax, ci, y, color)
            ax.text(ci["mean"], y+.24, f'{ci["mean"]:.2f}', ha="center", color=color, weight="bold", fontsize=11)
            rows.append(dict(protocol=protocol, condition=condition, **ci, binary_prior_mae=baseline, n_seeds=5))
        ax.set(title=label, xlim=(.65, 3.03), ylim=(-.4, 2.7), yticks=[2, 1, 0], yticklabels=LABELS,
               xlabel="Mean absolute delay error (lower is better)")
        ax.text(baseline+.04, 2.49, "Binary rule\n1.318", fontsize=9, color="#596B5B", va="top")
        ax.grid(axis="x", alpha=.12)
        ax.tick_params(axis="y", length=0, pad=10)
    pd.DataFrame(rows).to_csv(out / "01_capability_summary.csv", index=False)
    novel.to_csv(out / "01_capability_seed_values.csv", index=False)
    finish(fig, out, "01_partner_speed_information",
           "Can memory reveal partner speed beyond which goal is faster?",
           "A readout trained on familiar partners predicts both goal delays for unseen profiles, at the start of round 20.",
           "Delay counts waiting steps between eligible moves. Green: better than the true faster-goal baseline with training-derived typical delays.\n"
           "Small dots: 5 learner seeds per condition. Large dots/bars: mean and 95% seed bootstrap interval. Readability does not establish causal use.")


def temporal(root, out, analysis_seed):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    fig.subplots_adjust(left=.08, right=.97, top=.78, bottom=.25, wspace=.14)
    ticklabels = ["R1\nstart", "R1\nend", "R5\nstart", "R5\nend", "R10\nstart", "R10\nend", "R20\nstart", "R20\nend"]
    rows, transfers, seed_rows = [], [], []
    for ax, protocol, label, color in zip(axes, PROTOCOLS, PROTOCOL_LABELS, PROTOCOL_COLORS):
        matrices = [pd.read_csv(root / protocol / f"{MAIN}_seed{s}" / "cross_temporal_accuracy.csv", index_col=0) for s in range(1, 6)]
        assert all(m.shape == (8, 8) and list(m.columns) == list(m.index) for m in matrices)
        diagonal = np.stack([np.diag(m) for m in matrices]) * 100
        ci = grouped_ci(diagonal, np.arange(5), seed=analysis_seed, n=1000)
        ax.fill_between(np.arange(8), ci["low"], ci["high"], color=color, alpha=.13)
        ax.plot(np.arange(8), ci["mean"], "o-", color=color, lw=2, ms=6)
        for seed in range(5):
            ax.scatter(np.arange(8)+np.linspace(-.08, .08, 5)[seed], diagonal[seed], s=13, color=color, alpha=.35)
        ax.axhline(50, ls="--", color="#8E99A4", lw=1.2)
        ax.text(.97, .04, "Chance = 50%", transform=ax.transAxes, ha="right", fontsize=9, color="#677786")
        transfer = np.array([m.loc["r20 start", "r20 end"] for m in matrices])*100
        tci = grouped_ci(transfer, np.arange(5), seed=analysis_seed, n=1000)
        ax.text(.27, .16, f'Same readout, round 20 start → end:\n{tci["mean"]:.1f}% accuracy [{tci["low"]:.1f}, {tci["high"]:.1f}]',
                transform=ax.transAxes, fontsize=10.5, color=color,
                bbox=dict(facecolor="white", edgecolor="#E2E8F0", boxstyle="round,pad=.6"))
        ax.set(title=label, xticks=np.arange(8), xticklabels=ticklabels, ylim=(43, 104), xlim=(-.35, 7.35), xlabel="Experience checkpoint (R = round)")
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=.14)
        for phase, source in enumerate(matrices[0].columns):
            rows.append(dict(protocol=protocol, phase=source, mean=ci["mean"][phase], low=ci["low"][phase], high=ci["high"][phase], n_seeds=5))
            for seed in range(5):
                seed_rows.append(dict(protocol=protocol, phase=source, seed=seed+1, accuracy_percent=diagonal[seed, phase]))
        transfers.append(dict(protocol=protocol, train_phase="r20 start", test_phase="r20 end", **tci))
    axes[0].set_ylabel("Accuracy reading the partner’s faster goal (%)")
    pd.DataFrame(rows).to_csv(out / "02_temporal_accuracy.csv", index=False)
    pd.DataFrame(transfers).to_csv(out / "02_late_phase_transfer.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out / "02_temporal_seed_values.csv", index=False)
    finish(fig, out, "02_memory_over_experience",
           "Partner information becomes readable with experience",
           "Diverse partners + influence: faster-goal classification begins near chance and becomes highly accurate.",
           "Each plotted phase has its own fitted readout; evaluation uses held-out episodes. Band: 95% bootstrap across 5 learner seeds.\n"
           "Box: one readout fitted at round 20 start, tested at its end. Checkpoints are categorical, not equally spaced in elapsed time.")


def interventions(root, out):
    effects = pd.read_csv(root / "intervention_policy_episode_summary.csv")
    effects = effects[(effects.condition == MAIN) & (effects.donor_kind == "opposite")]
    summary = json.loads((root / "intervention_aggregate.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3), gridspec_kw={"width_ratios": [1, 1.22]})
    fig.subplots_adjust(left=.16, right=.97, top=.76, bottom=.28, wspace=.9)
    rows = []
    def draw(ax, protocol, kind, y, color):
        part = effects[(effects.protocol == protocol) & (effects.kind == kind)]
        seeds = part.groupby("seed").toward_donor_allocation_delta.mean()
        assert len(seeds) == 5 and len(part) == 30
        entry = next(r for r in summary if r["protocol"] == protocol and r["condition"] == MAIN and r["donor_kind"] == "opposite" and r["kind"] == kind)
        ci = entry["allocation"]
        assert np.isclose(ci["mean"], seeds.mean())
        ax.scatter(seeds*100, y+np.linspace(-.055, .055, 5), s=24, color=color, alpha=.4)
        errorbar(ax, ci, y, color, scale=100)
        rows.append(dict(protocol=protocol, intervention=kind,
                         mean_pp=ci["mean"]*100, low_pp=ci["low"]*100, high_pp=ci["high"]*100, n_seeds=5, histories_per_seed=6))
        return ci
    for ax in axes:
        ax.axvline(0, color="#AAB4BE", ls="--", lw=1.3)
        ax.grid(axis="x", alpha=.12)
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.set_xlabel("Change toward donor’s assignment\n(percentage points)")
    for index, (protocol, label, color) in enumerate(zip(PROTOCOLS, PROTOCOL_LABELS, PROTOCOL_COLORS)):
        ci = draw(axes[0], protocol, "whole", 1-index, color)
        axes[0].text(ci["mean"]*100, 1-index+.22, f'+{ci["mean"]*100:.1f} pp', ha="center", color=color, weight="bold")
        for j, kind in enumerate(["partner", "random", "rank_matched_nonpartner"]):
            draw(axes[1], protocol, kind, 2-j+(.1 if index == 0 else -.1), color)
    axes[0].set(title="Swap the entire state", yticks=[1, 0], yticklabels=["Fixed (v1)", "Online (v2)"], xlim=(-5, 108), ylim=(-.45, 1.6))
    axes[1].set(title="Small edits: magnified scale", yticks=[2, 1, 0],
                yticklabels=["Candidate partner\ndirections", "Random direction\n(same size)", "Other directions\n(same rank and size)"], xlim=(-1, 6.5), ylim=(-.45, 2.6))
    fig.legend([Line2D([], [], color=c, marker="o", lw=0) for c in PROTOCOL_COLORS], PROTOCOL_LABELS,
               loc="lower center", bbox_to_anchor=(.58, .135), ncol=2, frameon=False, fontsize=10)
    pd.DataFrame(rows).to_csv(out / "03_intervention_summary.csv", index=False)
    effects[effects.kind.isin(["whole", "partner", "random", "rank_matched_nonpartner"])].to_csv(out / "03_intervention_recipient_values.csv", index=False)
    finish(fig, out, "03_memory_and_decisions",
           "Changing remembered history can change the next decision",
           "Diverse partners + influence: transplant from a donor whose faster goal is opposite to the recipient’s.",
           "Same physical situation and input; first learned allocation (v1: t=0, v2: t=1). Positive = closer to donor’s faster-goal assignment.\n"
           "Dots/bars: mean and 95% seed bootstrap; small dots: 5 seed means (6 held-out histories each). Full-state swaps also change other memories.")


def dynamics(root, out):
    table = pd.read_csv(root / "dynamics_summary.csv")
    assert len(table) == 30
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.1), sharey=True)
    fig.subplots_adjust(left=.09, right=.97, top=.78, bottom=.29, wspace=.17)
    rows = []
    for ax, protocol, label in zip(axes, PROTOCOLS, PROTOCOL_LABELS):
        part = table[table.protocol == protocol].copy()
        assert len(part) == 15 and (part.test_mse < part.input_omitting_dmd_mse).all()
        without = part.input_omitting_dmd_mse / part.persistence_mse
        driven = part.test_mse / part.persistence_mse
        for (_, policy), old, new in zip(part.iterrows(), without, driven):
            color = COLORS[CONDITIONS.index(policy.condition)]
            ax.plot([0, 1, 2], [1, old, new], "o-", color=color, alpha=.4, ms=4, lw=1)
            rows.append(dict(protocol=protocol, policy=policy.policy, condition=policy.condition, seed=policy.seed,
                             persistence=1., state_history_only=old, state_and_input=new))
        medians = [1., float(without.median()), float(driven.median())]
        ax.plot(np.arange(3), medians, "D-", color="#182A3B", lw=2.3, ms=7, zorder=5)
        for x, value in enumerate(medians):
            ax.text(x, value+.06, f"{value:.2f}", ha="center", weight="bold", fontsize=11)
        ax.axhline(1., ls="--", color="#AAB4BE", lw=1)
        ax.text(.04, .04, f'{100*(1-medians[2]):.1f}% lower median error\nthan holding the state unchanged',
                transform=ax.transAxes, fontsize=10.5, color="#1976A3")
        ax.set(title=label, xticks=[0, 1, 2], xticklabels=["Hold state\nunchanged", "State history\nonly", "State history +\nnew observation"],
               xlim=(-.25, 2.25), ylim=(0, 1.17))
        ax.grid(axis="y", alpha=.13)
    axes[0].set_ylabel("Next-state prediction error\n(relative to unchanged state; lower is better)")
    handles = [Line2D([], [], color=c, marker="o", lw=1) for c in COLORS]
    handles.append(Line2D([], [], color="#182A3B", marker="D", lw=2))
    fig.legend(handles, ["Diverse + influence", "Single + influence", "Diverse, no influence", "Median across 15 policies"],
               loc="lower center", bbox_to_anchor=(.52, .14), ncol=4, frameon=False, fontsize=9.5)
    pd.DataFrame(rows).to_csv(out / "04_prediction_error_ratios.csv", index=False)
    finish(fig, out, "04_observation_driven_dynamics",
           "New observations improve prediction of recurrent updates",
           "State history + observation beats the state-history-only fit in all 15 policies per protocol.",
           "Each colored line: one policy; 3 training conditions × 5 learner seeds. Models use held-out trajectories and training-only state PCs.\n"
           "Errors score only newly predicted current-state coordinates. Targeted collection covers the first 4 rounds; predictive fit is not causal identification.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = args.results.resolve()
    out = root / "findings"
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "configuration.json").read_text())
    style()
    capability(root, out)
    temporal(root, out, config["analysis_seed"])
    interventions(root, out)
    dynamics(root, out)
    (out / "README.md").write_text(
        "# Interpretable findings\n\n"
        "Generated from completed analysis summaries, without rerunning or refitting models. "
        "Each PNG has a vector PDF and its plotted values in CSV.\n\n"
        "1. **Partner speed:** the diverse + influence fixed-allocation policies improve novel-profile delay prediction beyond an oracle faster-goal baseline; online evidence is weaker.\n"
        "2. **Experience:** faster-goal information is initially near chance, then readable across phases in the diverse + influence condition.\n"
        "3. **Decisions:** whole-state swaps change allocations; candidate partner edits are small and exploratory. The panels use different horizontal scales.\n"
        "4. **Dynamics:** adding observation input improves next-state prediction in all 30 policies, within the early four-round capture.\n\n"
        "Bootstrap intervals have only five learner seeds per condition. Subspace validation is limited; no selective partner-memory claim follows. "
        "See the parent report for coverage, controls and limitations.\n\n"
        "![Partner speed information](01_partner_speed_information.png)\n\n"
        "![Memory over experience](02_memory_over_experience.png)\n\n"
        "![Memory and decisions](03_memory_and_decisions.png)\n\n"
        "![Observation-driven dynamics](04_observation_driven_dynamics.png)\n\n"
        "Reproduce from the repository root with the experiment Python:\n\n"
        f"```bash\npython -m eval.partner_dynamics.plot_findings --results {root}\n```\n"
    )
    print(f"Saved four PNG/PDF figures and their numerical values to {out}")


if __name__ == "__main__":
    main()
