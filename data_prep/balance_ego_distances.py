"""Sample layouts with exact, simultaneous red/blue ego-distance quotas.

Only ego distances enter selection. Within each joint (red, blue) bin,
layouts are sampled uniformly without replacement. Expected joint-bin counts
maximize selection entropy subject to the two marginal quotas; randomized
cycle rounding makes those counts integer while preserving their expectations.
Other properties can change as a consequence of conditioning on ego distance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.special import expit


def expected_counts(capacities: np.ndarray, quota: int) -> np.ndarray:
    """Maximum-entropy fractional selection with each marginal equal to quota."""
    n = capacities.shape[0]
    graph = np.zeros((2 * n + 2, 2 * n + 2), dtype=np.int64)
    graph[0, 1:n + 1] = quota
    graph[1:n + 1, n + 1:2 * n + 1] = capacities
    graph[n + 1:2 * n + 1, -1] = quota
    if maximum_flow(csr_matrix(graph), 0, 2 * n + 1).flow_value != n * quota:
        raise ValueError(f"Cannot select {quota} layouts at every distance for both goals")

    remaining = capacities.copy()
    fixed = np.zeros_like(capacities, dtype=np.float64)
    row_target = np.full(n, quota, dtype=np.int64)
    col_target = np.full(n, quota, dtype=np.int64)
    # Remove forced full/empty rows and columns before solving finite logits.
    while True:
        changed = False
        for axis in (0, 1):
            for i in range(n):
                available = remaining[i, :] if axis == 0 else remaining[:, i]
                target = row_target[i] if axis == 0 else col_target[i]
                total = int(available.sum())
                if total == 0 or target not in (0, total):
                    continue
                chosen = available.copy() if target == total else np.zeros(n, dtype=np.int64)
                if axis == 0:
                    fixed[i, :] += chosen
                    remaining[i, :] = 0
                    row_target[i] -= int(chosen.sum())
                    col_target -= chosen
                else:
                    fixed[:, i] += chosen
                    remaining[:, i] = 0
                    col_target[i] -= int(chosen.sum())
                    row_target -= chosen
                changed = True
        if not changed:
            break
    if not remaining.any():
        return fixed

    def residual(parameters):
        counts = remaining * expit(parameters[:n, None] + parameters[None, n:])
        return np.concatenate((counts.sum(axis=1) - row_target, counts.sum(axis=0) - col_target))

    result = least_squares(residual, np.zeros(2 * n),
                           ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=2000)
    if np.max(np.abs(residual(result.x))) > 1e-7:
        raise RuntimeError("Fractional marginal balancing did not converge")
    counts = fixed + remaining * expit(result.x[:n, None] + result.x[None, n:])
    assert np.allclose(counts.sum(axis=0), quota, atol=1e-7, rtol=0)
    assert np.allclose(counts.sum(axis=1), quota, atol=1e-7, rtol=0)
    return counts


def _fractional_cycle(counts: np.ndarray):
    n = counts.shape[0]
    adjacent = [[] for _ in range(2 * n)]
    for row, col in np.argwhere(np.abs(counts - np.rint(counts)) > 1e-8):
        adjacent[row].append(n + col)
        adjacent[n + col].append(row)
    visited = set()
    path = []

    def visit(node, parent):
        visited.add(node)
        path.append(node)
        for neighbor in adjacent[node]:
            if neighbor == parent:
                continue
            if neighbor in path:
                cycle = path[path.index(neighbor):] + [neighbor]
                return [(a, b - n) if a < n else (b, a - n)
                        for a, b in zip(cycle[:-1], cycle[1:])]
            if neighbor not in visited:
                cycle = visit(neighbor, node)
                if cycle:
                    return cycle
        path.pop()
        return None

    for node in range(2 * n):
        if node not in visited:
            cycle = visit(node, -1)
            if cycle:
                return cycle
    return None


def round_counts(counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Round on alternating cycles, keeping marginal totals and expectations."""
    counts = counts.copy()
    while np.any(np.abs(counts - np.rint(counts)) > 1e-8):
        cycle = _fractional_cycle(counts)
        if cycle is None:
            raise RuntimeError("Fractional counts have no rounding cycle")
        signs = np.array([1 if i % 2 == 0 else -1 for i in range(len(cycle))])
        values = np.array([counts[row, col] for row, col in cycle])
        up, down = np.ceil(values) - values, values - np.floor(values)
        plus = float(np.min(np.where(signs > 0, up, down)))
        minus = float(np.min(np.where(signs > 0, down, up)))
        shift = plus if rng.random() < minus / (plus + minus) else -minus
        for (row, col), sign in zip(cycle, signs):
            counts[row, col] += shift * sign
        near_integer = np.abs(counts - np.rint(counts)) <= 1e-8
        counts[near_integer] = np.rint(counts[near_integer])
    return np.rint(counts).astype(np.int64)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_distance", type=int, default=8)
    parser.add_argument("--per_distance", type=int, default=137)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Output already exists: {args.out_dir}")
    if args.max_distance < 1 or args.per_distance < 1:
        raise ValueError("Distance limit and quota must be positive")
    paths = sorted((args.source_dir / "layouts/train").glob("*.json"))
    layouts = [json.loads(path.read_text()) for path in paths]
    if not layouts:
        raise ValueError("No source layouts found")
    bins = [[[] for _ in range(args.max_distance)] for _ in range(args.max_distance)]
    eligible = []
    for i, layout in enumerate(layouts):
        m = layout["metadata"]
        red, blue = m["ego_to_red"], m["ego_to_blue"]
        if 1 <= red <= args.max_distance and 1 <= blue <= args.max_distance:
            bins[red - 1][blue - 1].append(i)
            eligible.append(i)
    capacity = np.array([[len(cell) for cell in row] for row in bins], dtype=np.int64)
    fractional = expected_counts(capacity, args.per_distance)
    rng = np.random.default_rng(args.seed)
    chosen_counts = round_counts(fractional, rng)
    assert np.all(chosen_counts.sum(axis=0) == args.per_distance)
    assert np.all(chosen_counts.sum(axis=1) == args.per_distance)
    assert np.all((chosen_counts >= 0) & (chosen_counts <= capacity))
    selected = []
    for red in range(args.max_distance):
        for blue in range(args.max_distance):
            selected.extend(rng.choice(bins[red][blue], size=chosen_counts[red, blue], replace=False).tolist())
    rng.shuffle(selected)
    assert len(selected) == len(set(selected)) == args.max_distance * args.per_distance

    layout_dir = args.out_dir / "layouts/train"
    diagnostic_dir = args.out_dir / "diagnostics"
    layout_dir.mkdir(parents=True)
    diagnostic_dir.mkdir()
    # Keep source IDs and exact file bytes; the NPZ uses sorted filename order,
    # matching the environment loader and the plotting utility.
    for i in selected:
        shutil.copyfile(paths[i], layout_dir / paths[i].name)
    saved = [json.loads(path.read_text()) for path in sorted(layout_dir.glob("*.json"))]
    np.savez(args.out_dir / "train_layouts.npz",
             layouts=np.array([layout["grid"] for layout in saved], dtype=np.int8))
    selected_set = set(selected)
    eligible_set = set(eligible)
    audit = []
    for i, layout in enumerate(layouts):
        m = layout["metadata"]
        audit.append({
            "source_layout_id": layout["layout_id"], "seed": layout["seed"],
            "ego_to_red": m["ego_to_red"], "ego_to_blue": m["ego_to_blue"],
            "partner_to_red": m["partner_to_red"], "partner_to_blue": m["partner_to_blue"],
            "realized_wall_density": m["realized_wall_density"],
            "eligible_both_distances_in_range": i in eligible_set, "selected": i in selected_set,
            "exclusion_reason": "" if i in selected_set else
                                ("outside_ego_distance_range" if i not in eligible_set else "random_sampling_for_joint_quotas"),
        })
    _write_csv(diagnostic_dir / "selection_audit.csv", audit)
    _write_csv(diagnostic_dir / "joint_distance_counts.csv", [
        {"ego_to_red": red + 1, "ego_to_blue": blue + 1,
         "available": int(capacity[red, blue]), "expected_selected": float(fractional[red, blue]),
         "selected": int(chosen_counts[red, blue])}
        for red in range(args.max_distance) for blue in range(args.max_distance)
    ])
    (diagnostic_dir / "selected_source_layout_ids.json").write_text(
        json.dumps([layout["layout_id"] for layout in saved], indent=2) + "\n")
    manifest = {
        "source_corpus": str(args.source_dir.resolve()),
        "source_manifest_sha256": hashlib.sha256((args.source_dir / "manifest.json").read_bytes()).hexdigest(),
        "n_source": len(layouts), "n_after_distance_cutoff": len(eligible),
        "n_excluded_by_distance_cutoff": len(layouts) - len(eligible),
        "n_excluded_by_balancing_sample": len(eligible) - len(selected),
        "n_final": len(selected), "n_train": len(selected), "n_val": 0, "n_test": 0,
        "ego_distance_range_inclusive": [1, args.max_distance],
        "per_goal_per_distance_quota": args.per_distance,
        "sampling_seed": args.seed,
        "sampling_method": "Maximum selection entropy for fractional joint-bin counts; expectation-preserving randomized cycle rounding; uniform sampling without replacement within each joint bin",
        "selection_variables": ["ego_to_red", "ego_to_blue"],
        "distribution_note": "Other properties are not selection criteria, but their distributions can change through conditioning on ego distance and finite random sampling",
        "red_distance_counts": chosen_counts.sum(axis=1).tolist(),
        "blue_distance_counts": chosen_counts.sum(axis=0).tolist(),
        "source_layouts_and_metadata_preserved": True,
        "source_generation_filters": json.loads((args.source_dir / "manifest.json").read_text()),
        "sampler_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reproduce_command": f"python data_prep/balance_ego_distances.py --source_dir {args.source_dir.resolve()} --out_dir {args.out_dir.resolve()} --max_distance {args.max_distance} --per_distance {args.per_distance} --seed {args.seed}",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_csv(diagnostic_dir / "filter_stages.csv", [
        {"stage": "source_corpus", "n_input": len(layouts), "n_rejected": 0, "n_surviving": len(layouts)},
        {"stage": "both_ego_distances_in_range", "n_input": len(layouts), "n_rejected": len(layouts) - len(eligible), "n_surviving": len(eligible)},
        {"stage": "random_sample_for_exact_joint_marginals", "n_input": len(eligible), "n_rejected": len(eligible) - len(selected), "n_surviving": len(selected)},
    ])
    print(json.dumps({key: manifest[key] for key in (
        "n_source", "n_after_distance_cutoff", "n_final", "red_distance_counts", "blue_distance_counts")}, indent=2))


if __name__ == "__main__":
    main()
