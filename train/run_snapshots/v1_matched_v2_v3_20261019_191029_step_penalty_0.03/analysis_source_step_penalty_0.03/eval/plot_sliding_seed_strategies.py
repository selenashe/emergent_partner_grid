"""Plot the complete sliding seed-strategy audit and matched-layout diagnostics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

LABELS = {"rnn_diverse_influence": "Diverse RNN + influence", "mlp_diverse_influence": "Diverse MLP + influence",
          "rnn_single_influence": "Single-partner RNN + influence", "rnn_diverse_noinfluence": "Diverse RNN, no influence"}


def export(fig, out, name):
    for extension in ("png", "pdf"):
        fig.savefig(out / (name + "." + extension), dpi=180, facecolor="white")


def switching_diagnostics(root, seeds):
    rows, profile_rows, matched = [], [], []
    for keys, group in seeds.groupby(["variant", "condition", "population"], sort=False):
        reference = None
        match = True
        for _, seed in group.sort_values("seed").iterrows():
            path = root / "cache" / seed.variant / seed.condition / f"seed{seed.seed}_{seed.population}.npz"
            with np.load(path, allow_pickle=False) as data:
                signature = (data["pool"].copy(), data["profile"].copy(), data["layout"].copy())
                if reference is None:
                    reference = signature
                else:
                    match &= all(np.array_equal(a, b) for a, b in zip(signature, reference))
                opportunities = np.maximum(data["steps"] - int(seed.t) - 1, 0)
                rows.append(dict(variant=seed.variant, condition=seed.condition, population=seed.population, seed=int(seed.seed),
                                 switches_per_postcommit_transition=float(data["switches"].sum() / opportunities.sum())))
                for index, (dr, db) in enumerate(data["pool"]):
                    selected = data["profile"] == index
                    profile_rows.append(dict(variant=seed.variant, condition=seed.condition, population=seed.population, seed=int(seed.seed),
                                             delay_red=int(dr), delay_blue=int(db),
                                             switches_per_postcommit_transition=float(data["switches"][selected].sum() / opportunities[selected].sum())))
        matched.append(dict(variant=keys[0], condition=keys[1], population=keys[2], same_episode_profiles_and_layouts_across_five_seeds=bool(match)))
    pd.DataFrame(rows).to_csv(root / "seed_switching_rates.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(root / "profile_switching_rates.csv", index=False)
    pd.DataFrame(matched).to_csv(root / "matched_evaluation_layouts.csv", index=False)
    print("Matched evaluation profile/layout sequences:", sum(r["same_episode_profiles_and_layouts_across_five_seeds"] for r in matched), "/", len(matched), flush=True)


def seed_overview(seeds, condition, out):
    selected = seeds[(seeds.condition == condition) & (seeds.population == "test")]
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 7.8), layout="constrained")
    metrics = (("initial_late_adjusted_gap", "Initial role specialization", "RdBu_r", -1, 1),
               ("final_late_adjusted_gap", "Final role specialization", "RdBu_r", -1, 1),
               ("episode_return", "Novel-profile episode return", "viridis", 0, 18),
               ("switches_per_round", "Postcommit switches / round", "magma_r", 0, 12))
    for ri, mode in enumerate(("irreversible", "reversible")):
        subset = selected[selected.reversibility == mode]
        for ci, (column, title, cmap, low, high) in enumerate(metrics):
            ax = axes[ri, ci]
            matrix = subset.pivot(index="t", columns="seed", values=column).reindex(index=range(11), columns=range(1, 6)).to_numpy()
            image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=low, vmax=high)
            for y in range(11):
                for x in range(5):
                    value = matrix[y, x]
                    text = f"{value:.2f}" if ci < 2 else f"{value:.1f}"
                    color = "white" if ((ci < 2 and abs(value) > .55) or (ci == 2 and value < 9) or (ci == 3 and value > 4)) else "black"
                    ax.text(x, y, text, ha="center", va="center", fontsize=8, color=color)
            ax.set_title(title, fontsize=10.5)
            ax.set_xticks(range(5), range(1, 6));ax.set_xlabel("Training seed")
            ax.set_yticks(range(11), range(11))
            if ci == 0:
                ax.set_ylabel(mode.capitalize() + "\nObservation time t")
            fig.colorbar(image, ax=ax, shrink=.75, pad=.02)
    fig.suptitle(LABELS[condition] + ": strategy and performance of every training seed", fontsize=14)
    fig.supxlabel("Role specialization = P(partner RED | faster at RED) − P(partner RED | faster at BLUE), adjusted for layout and round parity.\n"
                  "Rounds 11–20; positive values favor the faster specialist goal. Goal speed is a proxy, not an optimal-allocation oracle.", fontsize=9)
    export(fig, out, "seed_overview_" + condition)
    plt.close(fig)


def atlas(profiles, out):
    records = profiles.copy()
    records["pair"] = list(zip(records.delay_red, records.delay_blue))
    pairs = sorted(set(records.pair), key=lambda p: (p[0] > p[1], p[0] if p[0] < p[1] else p[1], p[1] if p[0] < p[1] else p[0]))
    labels = [f"{dr},{db}" for dr, db in pairs]
    population = {row.pair: row.population for _, row in records.iterrows()}
    row_order = [(t, seed) for t in range(11) for seed in range(1, 6)]
    low_return = max(-20, float(np.floor(records.episode_return.min())))
    with PdfPages(out / "all_conditions_profile_strategy_atlas.pdf") as pdf:
        for condition in LABELS:
            fig, axes = plt.subplots(2, 3, figsize=(19, 17), layout="constrained")
            for ri, mode in enumerate(("irreversible", "reversible")):
                subset = records[(records.condition == condition) & (records.reversibility == mode)]
                for ci, (column, title, cmap, low, high) in enumerate((("initial_partner_RED", "Initial partner RED fraction", "RdBu_r", 0, 1),
                                                                     ("final_partner_RED", "Final partner RED fraction", "RdBu_r", 0, 1),
                                                                     ("episode_return", "Episode return by profile", "viridis", low_return, 19))):
                    ax = axes[ri, ci]
                    lookup = {(int(r.t), int(r.seed), r.pair): getattr(r, column) for r in subset.itertuples()}
                    matrix = np.array([[lookup[t, seed, pair] for pair in pairs] for t, seed in row_order])
                    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
                    ax.set_title(mode.capitalize() + ": " + title, fontsize=10)
                    ax.set_xticks(range(len(pairs)), labels, rotation=90, fontsize=5.5)
                    for label, pair in zip(ax.get_xticklabels(), pairs):
                        if population[pair] == "test":label.set_color("#c86b00")
                    ax.axvline(22.5, color="black", lw=.8)
                    for boundary in np.arange(4.5, 54, 5):ax.axhline(boundary, color="black", lw=.5)
                    ax.set_yticks(range(55), [f"t{t}:s{seed}" for t, seed in row_order], fontsize=6)
                    ax.set_xlabel("Partner delays (RED,BLUE); orange labels = novel\nRED specialists on left, BLUE specialists on right", fontsize=8)
                    if ci == 0:ax.set_ylabel("Observation time : learner seed")
                    fig.colorbar(image, ax=ax, shrink=.55, pad=.015)
            note = "Actual partner role; no-influence assignments are environment imposed." if condition.endswith("noinfluence") else "Actual partner role; initial and final choices can differ under reversible allocation."
            fig.suptitle(LABELS[condition] + " — all 110 policies and all 46 partner profiles\n" + note, fontsize=14)
            export(fig, out, "profile_strategy_atlas_" + condition)
            pdf.savefig(fig)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    args = parser.parse_args()
    root = args.audit
    seeds = pd.read_csv(root / "seed_strategy_summary.csv")
    profiles = pd.read_csv(root / "profile_strategy_per_seed.csv")
    require_count = len(seeds) == 880 and len(profiles) == 20240
    if not require_count:raise ValueError("Missing policy/profile data")
    switching_diagnostics(root, seeds)
    out = root / "figures";out.mkdir(exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10})
    for condition in LABELS:seed_overview(seeds, condition, out)
    atlas(profiles, out)
    print("Strategy figures:", out, flush=True)


if __name__ == "__main__":
    main()
