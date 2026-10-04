"""Text-only summary of a generated grid corpus.

Given a directory of layout JSONs (as produced by env_generator.py — either a
single flat folder of *.json, or a parent folder containing per-split
subdirectories like train/test/val/), prints:

  1. Metadata distributions: median + IQR for every metadata descriptor.
  2. Pairwise descriptor Spearman correlations (numeric, no plots).
  3. Mean success and mean steps-to-success for the three non-communicating
     baselines (nearest, wait_react, random).

Example usage:
    # A corpus with per-split subdirectories, print-only:
    python dev/analyze_grids_lite.py dev/grids/layouts

    # A single flat folder of JSONs (treated as one unnamed set):
    python dev/analyze_grids_lite.py dev/grids/layouts/train --trials 50

    # Also save one <split>_summary.json per split:
    python dev/analyze_grids_lite.py dev/grids/layouts --out_dir dev/grids/summaries
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from coordination_grid_baselines import (
    summarize_baselines,
    NON_COMM_BASELINES,
)

# --- cell codes (mirror env_generator.py) ---
EMPTY, WALL, RED, BLUE, EGO, PARTNER = 0, 1, 2, 3, 4, 5

HIST_FIELDS: List[str] = [
    "realized_wall_density",
    "num_junctions",
    "num_dead_ends",
    "num_shortest_paths_ego_red",
    "num_shortest_paths_ego_blue",
    "num_shortest_paths_partner_red",
    "num_shortest_paths_partner_blue",
    "shortest_path_overlap_assignment_1",
    "shortest_path_overlap_assignment_2",
    "assignment_cost_difference",
    "switching_cost_mean",
    "switching_cost_min",
]

PAIRS: List[Tuple[str, str]] = [
    ("realized_wall_density", "switching_cost_mean"),
    ("assignment_cost_difference", "switching_cost_mean"),
    ("num_junctions", "num_shortest_paths_ego_red"),
    ("shortest_path_overlap_assignment_1", "switching_cost_mean"),
    ("num_junctions", "num_dead_ends"),
    ("realized_wall_density", "assignment_cost_difference"),
    ("realized_wall_density", "num_junctions"),
    ("switching_cost_min", "switching_cost_mean"),
    ("shortest_path_overlap_assignment_1", "assignment_cost_difference"),
]

# --------------------------------------------------------------------------- #
# Corpus loading                                                              #
# --------------------------------------------------------------------------- #

def _load_jsons(dir_: Path) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    rows, envs = [], {}
    for path in sorted(dir_.glob("*.json")):
        d = json.load(open(path))
        # Baselines need this so they can build a CoordinationGrid from the
        # same layout file the metadata was scraped from.
        d["_path"] = str(path.resolve())
        rows.append({"layout_id": d["layout_id"], "seed": d["seed"], **d["metadata"]})
        envs[d["layout_id"]] = d
    return pd.DataFrame(rows), envs


def load_corpus(root: Path) -> Dict[str, Tuple[pd.DataFrame, Dict[str, dict]]]:
    """Return `{split_name: (df, envs)}`.

    If `root` contains subdirectories with JSONs, each subdirectory is one
    split. Otherwise `root` is treated as a single unnamed set (key: `root.name`).
    """
    subdirs = [p for p in sorted(root.iterdir()) if p.is_dir() and any(p.glob("*.json"))]
    if subdirs:
        return {p.name: _load_jsons(p) for p in subdirs}
    if any(root.glob("*.json")):
        return {root.name: _load_jsons(root)}
    raise FileNotFoundError(f"No layout JSONs found under {root}")


# --------------------------------------------------------------------------- #
# 1. Metadata distributions                                                   #
# --------------------------------------------------------------------------- #

def summarize_metadata(df: pd.DataFrame) -> Dict[str, dict]:
    print("[1] Metadata distributions: median [Q1, Q3]")
    width = max(len(f) for f in HIST_FIELDS)
    out: Dict[str, dict] = {}
    for f in HIST_FIELDS:
        if f not in df.columns:
            print(f"    {f:<{width}}  (missing)")
            out[f] = None
            continue
        vals = df[f].to_numpy(dtype=float)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        print(f"    {f:<{width}}  {med:8.3f}  [{q1:7.3f}, {q3:7.3f}]")
        out[f] = {"median": float(med), "q1": float(q1), "q3": float(q3)}
    return out


# --------------------------------------------------------------------------- #
# 2. Pairwise Spearman                                                        #
# --------------------------------------------------------------------------- #

def summarize_correlations(df: pd.DataFrame) -> List[dict]:
    print("[2] Pairwise Spearman correlations")
    left_w  = max(len(x) for x, _ in PAIRS)
    right_w = max(len(y) for _, y in PAIRS)
    out: List[dict] = []
    for x, y in PAIRS:
        if x not in df.columns or y not in df.columns:
            print(f"    {x:<{left_w}} x {y:<{right_w}}  (missing column)")
            out.append({"x": x, "y": y, "rho": None, "p": None})
            continue
        rho, p = spearmanr(df[x], df[y])
        print(f"    {x:<{left_w}} x {y:<{right_w}}  rho={rho:+.3f}  p={p:.2e}")
        out.append({"x": x, "y": y, "rho": float(rho), "p": float(p)})
    return out


# --------------------------------------------------------------------------- #
# 3. Non-communicating baselines                                              #
#    Delegated to coordination_grid_baselines so semantics match the env      #
#    exactly (collision rules, walls, t=0 free-comm step, partner sampling,   #
#    horizon, deterministic tie-break). See dev/coordination_grid_baselines.  #
# --------------------------------------------------------------------------- #

# Re-export so callers importing summarize_baselines from lite still work.
__all__ = ["summarize_metadata", "summarize_correlations", "summarize_baselines",
           "load_corpus", "NON_COMM_BASELINES"]


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", type=Path,
                   help="Directory of layout JSONs, or a parent folder of "
                        "split subdirectories (e.g. dev/grids/layouts).")
    p.add_argument("--trials", type=int, default=100,
                   help="Baseline trials per layout.")
    p.add_argument("--max_steps", type=int, default=15,
                   help="Per-episode step budget for baseline simulation "
                        "(should match CoordinationGrid.max_steps).")
    p.add_argument("--wait_k", type=int, default=2,
                   help="Wait-and-react observation window (env steps).")
    p.add_argument("--partner_z", type=float, default=0.5,
                   help="Partner type z. NONE-only baselines are z-invariant.")
    p.add_argument("--seed", type=int, default=0,
                   help="Master seed for baseline trials.")
    p.add_argument("--skip_baselines", action="store_true",
                   help="Only report [1] and [2].")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="If set, also write one <split>_summary.json per split "
                        "containing all three reported sections.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    corpus = load_corpus(args.dir)

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for split, (df, envs) in corpus.items():
        header = f"=== Split: {split}  (N={len(df)}) ==="
        print("\n" + header)
        print("=" * len(header))
        meta = summarize_metadata(df)
        print()
        corrs = summarize_correlations(df)
        print()
        baselines = None
        if not args.skip_baselines:
            baselines = summarize_baselines(envs,
                                            trials=args.trials,
                                            max_steps=args.max_steps,
                                            wait_k=args.wait_k,
                                            seed=args.seed,
                                            partner_z=args.partner_z)

        if args.out_dir is not None:
            payload = {
                "split": split,
                "n": int(len(df)),
                "metadata_distributions": meta,
                "pairwise_spearman": corrs,
                "baselines": baselines,   # None if --skip_baselines
            }
            out_path = args.out_dir / f"{split}_summary.json"
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
