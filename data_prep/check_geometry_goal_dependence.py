"""Audit starting coordinates versus analytical optimal ego goal on the saved corpus.

Enumerate the authoritative 24 training and 22 held-out partner profiles, with
equal weight per layout/profile. Reuse the corpus-selection completion-time
primitive, recomputing shortest paths around walls. This is a static assignment
analysis, not a simulation of learned navigation, collisions, or v2 switching.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import runpy
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_prep.capability_selection import INFEASIBLE, completion_time, layout_bfs_distances
from repo_paths import resolve_path

DEFAULT_CORPUS = ROOT / "data_prep/grids_capability_selected_balanced_1096"
DEFAULT_MANIFEST = ROOT / "train/manifests/sbatch_counterbalanced1096_20261002_235609.json"
COORDINATES = ("ego_row", "ego_column", "partner_row", "partner_column")
POOLS = ("all_46", "training_24", "held_out_22")
POOL_LABELS = {"all_46": "All 46 profiles", "training_24": "24 training profiles",
               "held_out_22": "22 held-out profiles"}


def capability_profiles():
    # Load the authoritative constants without importing JAX or environment code.
    source = ROOT / "jaxmarl/environments/coordination_grid/capability_populations.py"
    population = runpy.run_path(str(source))
    train = list(population["TRAIN_CAPABILITY_PAIRS"])
    test = list(population["TEST_CAPABILITY_PAIRS"])
    if len(train) != 24 or len(test) != 22 or set(train) & set(test):
        raise ValueError("Expected the authoritative disjoint 24/22 capability pools")
    return train + test, source


def load_layouts(corpus):
    paths = sorted((corpus / "layouts/train").glob("*.json"))
    if not paths:
        raise ValueError(f"No layouts under {corpus}")
    records, digest = [], hashlib.sha256()
    for path in paths:
        raw = path.read_bytes()
        digest.update(path.name.encode())
        digest.update(raw)
        layout = json.loads(raw)
        grid = np.asarray(layout["grid"], dtype=int)
        distances = layout_bfs_distances((grid == 1).astype(int),
            tuple(layout["ego_start"]), tuple(layout["partner_start"]),
            tuple(layout["red_goal"]), tuple(layout["blue_goal"]))
        if any(v <= 0 for v in distances.values()):
            raise ValueError(f"Unreachable or coincident start/goal in {path}")
        for key, value in distances.items():
            if value != layout["metadata"][key]:
                raise ValueError(f"Saved BFS metadata mismatch: {path}, {key}")
        er, ec = layout["ego_start"]
        pr, pc = layout["partner_start"]
        records.append(dict(layout_id=layout["layout_id"], geometry_sha256=
            hashlib.sha256(json.dumps({k: layout[k] for k in
                ("grid", "ego_start", "partner_start", "red_goal", "blue_goal")},
                sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            ego_row=er, ego_column=ec, partner_row=pr, partner_column=pc,
            red_goal_row=layout["red_goal"][0], red_goal_column=layout["red_goal"][1],
            blue_goal_row=layout["blue_goal"][0], blue_goal_column=layout["blue_goal"][1],
            height=grid.shape[0], width=grid.shape[1], **distances))
    if len({r["layout_id"] for r in records}) != len(records):
        raise ValueError("Repeated layout IDs")
    return records, digest.hexdigest()


def assignment_times(layouts, profiles):
    """Uncapped times; horizon failures must not create artificial ties."""
    times = np.empty((len(layouts), len(profiles), 2), dtype=np.int64)
    for i, layout in enumerate(layouts):
        for j, (dr, db) in enumerate(profiles):
            times[i, j, 0] = completion_time(layout["ego_to_red"],
                layout["partner_to_blue"], db, INFEASIBLE - 1)
            times[i, j, 1] = completion_time(layout["ego_to_blue"],
                layout["partner_to_red"], dr, INFEASIBLE - 1)
    if np.any(times >= INFEASIBLE):
        raise ValueError("Analytical assignment contains unreachable goals")
    # +1: ego red / partner blue; -1: ego blue / partner red; 0: equal time.
    labels = np.sign(times[:, :, 1] - times[:, :, 0]).astype(np.int8)
    return times, labels


def correlation(x, labels):
    """Point-biserial Pearson r: red=1, blue=0; equal-time cases excluded."""
    x, labels = np.asarray(x), np.asarray(labels)
    valid = labels != 0
    x, y = x[valid], (labels[valid] == 1).astype(float)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def prediction_limit(labels, keys):
    """Exact best classification using only keys, without capability input.

    This is enumeration on the specified finite population, not model fitting
    or an estimate of generalization to unseen grids. Ties are either excluded
    or credited to either goal, explicitly reported as separate accuracies.
    """
    groups = {}
    for row, key in zip(labels, keys):
        groups.setdefault(key, np.zeros(3, dtype=np.int64))
        groups[key] += [(row == 1).sum(), (row == -1).sum(), (row == 0).sum()]
    counts = np.stack(list(groups.values()))
    correct = np.maximum(counts[:, 0], counts[:, 1]).sum()
    strict = counts[:, :2].sum()
    ties = counts[:, 2].sum()
    return dict(n_feature_groups=len(groups), non_tied_accuracy=float(correct / strict)
        if strict else None, tie_credited_accuracy=float((correct + ties) / counts.sum()))


def analyze_pool(layouts, labels, times, max_steps):
    per_layout = []
    for row, label in zip(layouts, labels):
        red, blue, ties = (int((label == value).sum()) for value in (1, -1, 0))
        strict = red + blue
        per_layout.append(dict(**row, red_optimal=red, blue_optimal=blue,
            equal_time=ties, non_tied=strict,
            p_red_non_tied=red / strict if strict else None,
            has_both_strict_optima=bool(red and blue)))
    flat = labels.ravel()
    correlations = [{"feature": key, "r": correlation(
        np.repeat([r[key] for r in layouts], labels.shape[1]), flat),
        "n_non_tied": int((flat != 0).sum())} for key in COORDINATES]
    predictor_keys = {
        "constant_goal": [0] * len(layouts),
        "ego_start_only": [(r["ego_row"], r["ego_column"]) for r in layouts],
        "partner_start_only": [(r["partner_row"], r["partner_column"]) for r in layouts],
        "both_starts_only": [tuple(r[k] for k in COORDINATES) for r in layouts],
        "full_grid_geometry": [r["geometry_sha256"] for r in layouts],
    }
    limits = {key: prediction_limit(labels, values) for key, values in predictor_keys.items()}
    return dict(n_profiles=labels.shape[1], n_layout_profile_pairs=int(labels.size),
        n_red_optimal=int((flat == 1).sum()), n_blue_optimal=int((flat == -1).sum()),
        n_equal_time=int((flat == 0).sum()),
        n_layouts_with_both_strict_optima=sum(r["has_both_strict_optima"] for r in per_layout),
        n_layouts_exactly_50_50_non_tied=sum(r["red_optimal"] == r["blue_optimal"]
            and r["non_tied"] > 0 for r in per_layout),
        oracle_cases_exceeding_horizon=int((times.min(axis=2) > max_steps).sum()),
        worst_oracle_completion_steps=int(times.min(axis=2).max()),
        correlations=correlations, prediction_limits=limits), per_layout


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def start_cells(layouts, labels, pool):
    rows = []
    for agent in ("ego", "partner"):
        h, w = layouts[0]["height"], layouts[0]["width"]
        for r in range(h):
            for c in range(w):
                indices = [i for i, ld in enumerate(layouts)
                    if ld[agent + "_row"] == r and ld[agent + "_column"] == c]
                cells = labels[indices]
                red, blue, ties = (int((cells == value).sum()) for value in (1, -1, 0))
                rows.append(dict(pool=pool, agent=agent, start_row=r, start_column=c,
                    n_layouts=len(indices), red_optimal=red, blue_optimal=blue,
                    equal_time=ties, p_red_non_tied=red / (red + blue) if red + blue else None))
    return rows


def make_figures(layouts, analyses, layout_rows, cell_rows, profiles, labels, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    for i, pool in enumerate(POOLS):
        for j, agent in enumerate(("ego", "partner")):
            h, w = layouts[0]["height"], layouts[0]["width"]
            data = np.full((h, w), np.nan)
            for row in cell_rows:
                if row["pool"] == pool and row["agent"] == agent and row["p_red_non_tied"] is not None:
                    data[row["start_row"], row["start_column"]] = row["p_red_non_tied"]
            im = axes[i, j].imshow(data, cmap=cmap, vmin=0, vmax=1, origin="upper")
            for r, c in zip(*np.where(np.isfinite(data))):
                axes[i, j].text(c, r, f"{100 * data[r,c]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if data[r,c] < .2 or data[r,c] > .8 else "black")
            axes[i, j].set(title=f"{POOL_LABELS[pool]}: {agent} start", xlabel="Start column (x)", ylabel="Start row (y)")
            axes[i, j].set_xticks(range(w)); axes[i, j].set_yticks(range(h))
        corrs = analyses[pool]["correlations"]
        values = [r["r"] if r["r"] is not None else 0 for r in corrs]
        axes[i, 2].barh(range(4), values, color="#667fae")
        for k, row in enumerate(corrs):
            display_r = 0 if row["r"] is not None and abs(row["r"]) < 1e-12 else row["r"]
            axes[i, 2].text(.98, k, "undefined" if display_r is None else f"r = {display_r:+.5f}",
                transform=axes[i, 2].get_yaxis_transform(), ha="right", va="center")
        axes[i, 2].set_yticks(range(4), [r["feature"].replace("_", " ") for r in corrs])
        axes[i, 2].set(xlim=(-1, 1), xlabel="Correlation with ego red-optimal (red=1, blue=0)",
                       title=f"{POOL_LABELS[pool]}: coordinate correlations")
        axes[i, 2].axvline(0, color="black", lw=.7)
    fig.suptitle("Starting positions versus fastest analytical ego goal: 1,096 shared v1/v2 grids", fontsize=15)
    fig.text(.5, .015, "Cells show % red-optimal among unequal-time assignments; grey = no data. Equal weight per layout/profile.\n"
        "Correlation alone does not establish independence from the entire grid. See per-layout prediction limits.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, .96))
    fig.colorbar(im, ax=axes[:, :2], shrink=.7, label="P(ego RED optimal | start cell, unequal times)", pad=.04)
    save_figure(fig, output, "start_position_goal_correlations")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), sharex=True, sharey=True)
    for ax, pool in zip(axes, POOLS):
        values = [r["p_red_non_tied"] for r in layout_rows if r["pool"] == pool and r["p_red_non_tied"] is not None]
        ax.hist(values, bins=np.linspace(-.025, 1.025, 22), color="#667fae", edgecolor="white")
        ax.axvline(.5, color="black", ls="--", lw=1)
        s = analyses[pool]
        ax.set(title=f"{POOL_LABELS[pool]}\n{s['n_layouts_exactly_50_50_non_tied']:,} / {len(layouts):,} grids exactly 50/50",
            xlabel="Fraction of profiles making ego RED optimal", xlim=(0, 1))
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Number of grids")
    fig.suptitle("Does optimal goal change with capability on each identical grid?", fontsize=15)
    fig.text(.5, .015, "Equal-time choices are excluded from red/blue fractions and reported separately in the tables.", ha="center")
    fig.tight_layout(rect=(0, .055, 1, .93))
    save_figure(fig, output, "per_layout_optimal_goal_balance")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = ["Constant goal", "Ego start only", "Partner start only", "Both starts", "Full grid geometry"]
    keys = list(analyses[POOLS[0]]["prediction_limits"])
    for i, (pool, color) in enumerate(zip(POOLS, ("#667fae", "#d88a4d", "#599782"))):
        values = [100 * analyses[pool]["prediction_limits"][key]["non_tied_accuracy"] for key in keys]
        ax.bar(np.arange(5) + (i-1)*.25, values, width=.24, color=color, label=POOL_LABELS[pool])
    ax.axhline(50, color="black", ls="--", lw=1)
    ax.set(xticks=np.arange(5), xticklabels=names, ylim=(0, 105), ylabel="Best achievable goal classification (%)",
        title="Exact prediction limits without partner capability input")
    ax.legend(); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.text(.5, .015, "Finite-population upper bounds, not fitted models. Unequal-time cases only. No confidence intervals: exhaustive enumeration.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    save_figure(fig, output, "geometry_only_prediction_limits")

    fig, ax = plt.subplots(figsize=(8, 7))
    data = np.full((10, 10), np.nan)
    for j, (dr, db) in enumerate(profiles):
        column = labels[:, j]
        if np.any(column != 0):
            data[dr, db] = (column == 1).sum() / (column != 0).sum()
    im = ax.imshow(data, vmin=0, vmax=1, cmap=cmap)
    for dr, db in profiles:
        if np.isfinite(data[dr, db]):
            ax.text(db, dr, f"{100*data[dr,db]:.0f}%", ha="center", va="center",
                color="white" if data[dr,db] < .2 or data[dr,db] > .8 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Partner BLUE delay", ylabel="Partner RED delay",
        title="Ego RED-optimal frequency by partner capability\n24 training + 22 held-out profiles; unequal-time cases")
    fig.colorbar(im, ax=ax, label="Fraction of grids with ego RED optimal")
    fig.tight_layout()
    save_figure(fig, output, "capability_optimal_goal_map")


def main(args):
    corpus = resolve_path(args.corpus_dir)
    output = args.output_dir or corpus / "diagnostics/geometry_goal_dependence"
    layouts, digest = load_layouts(corpus)
    manifest_path = resolve_path(args.training_manifest)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if digest != manifest["layout_files_sha256"] or len(layouts) != manifest["n_layouts"]:
        raise ValueError("Corpus does not match the frozen counterbalanced training manifest")
    frozen, frozen_hash = load_layouts(resolve_path(manifest["frozen_layouts_dir"]).parents[1])
    if frozen_hash != digest or len(frozen) != len(layouts):
        raise ValueError("Frozen training corpus differs from the analyzed layouts")
    profiles, profile_source = capability_profiles()
    times, labels = assignment_times(layouts, profiles)
    masks = {"all_46": np.arange(46), "training_24": np.arange(24), "held_out_22": np.arange(24, 46)}
    analyses, layout_rows, cell_rows, corr_rows = {}, [], [], []
    for pool, indices in masks.items():
        s, rows = analyze_pool(layouts, labels[:, indices], times[:, indices], args.max_steps)
        analyses[pool] = s
        layout_rows.extend(dict(pool=pool, **r) for r in rows)
        cell_rows.extend(start_cells(layouts, labels[:, indices], pool))
        corr_rows.extend(dict(pool=pool, capability_red_delay="", capability_blue_delay="", **r) for r in s["correlations"])
    pair_rows = []
    profile_rows = []
    for j, (dr, db) in enumerate(profiles):
        profile_rows.append(dict(red_delay=dr, blue_delay=db,
            pool="training_24" if j < 24 else "held_out_22",
            n_red_optimal=int((labels[:,j] == 1).sum()),
            n_blue_optimal=int((labels[:,j] == -1).sum()),
            n_equal_time=int((labels[:,j] == 0).sum()),
            optimal_goal_is_layout_invariant=bool(len(np.unique(labels[:,j])) == 1)))
        for key in COORDINATES:
            corr_rows.append(dict(pool="training_24" if j < 24 else "held_out_22",
                capability_red_delay=dr, capability_blue_delay=db, feature=key,
                r=correlation([r[key] for r in layouts], labels[:, j]), n_non_tied=int((labels[:, j] != 0).sum())))
        for i, layout in enumerate(layouts):
            pair_rows.append(dict(layout_id=layout["layout_id"], **{k: layout[k] for k in COORDINATES},
                capability_pool="training_24" if j < 24 else "held_out_22",
                partner_red_delay=dr, partner_blue_delay=db,
                ego_red_completion_steps=int(times[i,j,0]), ego_blue_completion_steps=int(times[i,j,1]),
                optimal_ego_goal={1:"red", -1:"blue", 0:"tie"}[int(labels[i,j])],
                completion_gap_red_minus_blue=int(times[i,j,0]-times[i,j,1]),
                oracle_within_horizon=bool(times[i,j].min() <= args.max_steps)))
    report = dict(n_layouts=len(layouts), n_unique_geometries=len({r["geometry_sha256"] for r in layouts}),
        n_profiles=46, n_layout_profile_pairs=len(pair_rows),
        capability_summary=profile_rows,
        n_profiles_with_layout_invariant_optimum=sum(r["optimal_goal_is_layout_invariant"] for r in profile_rows),
        capability_only_rule=dict(rule="Ego takes the partner's slower goal; partner takes its faster goal",
            matches_all_cases=bool(np.all(labels == np.sign(np.array(profiles)[:,0] - np.array(profiles)[:,1])[None,:]))),
        profiles=[dict(red_delay=dr, blue_delay=db, pool="training" if j < 24 else "held_out") for j,(dr,db) in enumerate(profiles)],
        source=dict(corpus=str(corpus), training_manifest=str(manifest_path),
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), layout_files_sha256=digest,
            frozen_corpus_hash_verified=True, capability_source=str(profile_source),
            capability_source_sha256=hashlib.sha256(profile_source.read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        definitions=dict(start="Ego and partner row/column coordinates; row=y, column=x",
            outcome="Fastest static analytical ego goal assignment; equal times are ties",
            ego_red_time="max(1+ego_to_red, 2+(partner_to_blue-1)*(d_blue+1))",
            ego_blue_time="max(1+ego_to_blue, 2+(partner_to_red-1)*(d_red+1))",
            weighting="Uniform within each explicitly listed capability pool and across grids",
            max_steps=args.max_steps, horizon="Reported for feasibility, not used to turn two timeouts into a tie"),
        existing_checks="Unit tests cover selection formulas, tie handling and capability flips on synthetic examples; no previous exhaustive coordinate-correlation audit of the final 1096 grids.",
        limitations=["Uses the established corpus-selection analytical convention, not an exact environment trajectory oracle.",
            "Shortest paths are independent: agent collisions and v2 switching/destination cooldown changes are not simulated.",
            "This common static assignment target is shared by the v1/v2 corpus; it is not a separate optimal switching policy for v2.",
            "Uniform profile weighting does not reproduce small unfinished-episode exposure differences at the fixed training-step cutoff.",
            "Zero coordinate correlation alone cannot rule out nonlinear geometry information; full-grid prediction limits address the finite enumerated population.",
            "Tied assignments permit either goal; their tie-credited accuracy must not be mistaken for prediction of a uniquely correct goal.",
            "These are exact descriptive statistics for this corpus and these 46 profiles, not a guarantee about other grids or capabilities."],
        pools=analyses)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "layout_capability_assignments.csv", pair_rows)
    write_csv(output / "per_layout_goal_balance.csv", layout_rows)
    write_csv(output / "start_cell_goal_frequencies.csv", cell_rows)
    write_csv(output / "coordinate_correlations.csv", corr_rows)
    write_csv(output / "capability_goal_summary.csv", profile_rows)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    make_figures(layouts, analyses, layout_rows, cell_rows, profiles, labels, output)
    print(json.dumps(dict(output=str(output), pools=analyses), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--training-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    main(args)
