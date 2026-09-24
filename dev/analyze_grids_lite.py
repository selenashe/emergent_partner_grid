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
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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

NON_COMM_BASELINES = ["nearest", "wait_react", "random"]

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


# --------------------------------------------------------------------------- #
# Corpus loading                                                              #
# --------------------------------------------------------------------------- #

def _load_jsons(dir_: Path) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    rows, envs = [], {}
    for path in sorted(dir_.glob("*.json")):
        d = json.load(open(path))
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
# --------------------------------------------------------------------------- #

def _bool_grid(env: dict) -> np.ndarray:
    return np.array(env["grid"], dtype=np.int8) != WALL


def _bfs(grid_b: np.ndarray, src: Tuple[int, int]) -> np.ndarray:
    H, W = grid_b.shape
    dist = -np.ones((H, W), dtype=np.int32)
    dist[src] = 0
    q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid_b[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _greedy_step(grid_b, dist_to_goal, pos, rng):
    r, c = pos
    cur = dist_to_goal[r, c]
    if cur <= 0:
        return pos
    cands = []
    H, W = grid_b.shape
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and grid_b[nr, nc] and dist_to_goal[nr, nc] == cur - 1:
            cands.append((nr, nc))
    if not cands:
        return pos
    return cands[rng.randrange(len(cands))]


def _resolve(pos_e, pos_p, next_e, next_p):
    if next_e == next_p and (next_e != pos_e or next_p != pos_p):
        return pos_e, pos_p
    if next_e == pos_p and next_p == pos_e:
        return pos_e, pos_p
    return next_e, next_p


def _success(pos_e, pos_p, red, blue) -> bool:
    return {tuple(pos_e), tuple(pos_p)} == {tuple(red), tuple(blue)}


def _simulate(env, ego_goal, partner_goal, rng, max_steps: int):
    gb = _bool_grid(env)
    d_e = _bfs(gb, tuple(ego_goal))
    d_p = _bfs(gb, tuple(partner_goal))
    pos_e = tuple(env["ego_start"]); pos_p = tuple(env["partner_start"])
    if _success(pos_e, pos_p, env["red_goal"], env["blue_goal"]):
        return True, 0
    for t in range(1, max_steps + 1):
        ne = _greedy_step(gb, d_e, pos_e, rng) if pos_e != tuple(ego_goal) else pos_e
        np_ = _greedy_step(gb, d_p, pos_p, rng) if pos_p != tuple(partner_goal) else pos_p
        pos_e, pos_p = _resolve(pos_e, pos_p, ne, np_)
        if _success(pos_e, pos_p, env["red_goal"], env["blue_goal"]):
            return True, t
    return False, max_steps


def _nearest_goal(gb, src, red, blue, rng):
    d = _bfs(gb, tuple(src))
    dr, db = int(d[tuple(red)]), int(d[tuple(blue)])
    if dr < db: return red
    if db < dr: return blue
    return red if rng.random() < 0.5 else blue


def _baseline_nearest(env, rng, max_steps):
    gb = _bool_grid(env)
    ego_g = _nearest_goal(gb, env["ego_start"], env["red_goal"], env["blue_goal"], rng)
    par_g = _nearest_goal(gb, env["partner_start"], env["red_goal"], env["blue_goal"], rng)
    return _simulate(env, ego_g, par_g, rng, max_steps)


def _baseline_random(env, rng, max_steps):
    goals = [env["red_goal"], env["blue_goal"]]
    return _simulate(env, goals[rng.randrange(2)], goals[rng.randrange(2)], rng, max_steps)


def _baseline_wait_react(env, rng, max_steps, k: int):
    """Matches analyze_grids.ipynb: ego waits up to k steps, then commits to
    the goal the partner is not heading toward. Wait steps are charged to
    the total step budget.
    """
    gb = _bool_grid(env)
    partner_goal = _nearest_goal(gb, env["partner_start"], env["red_goal"], env["blue_goal"], rng)
    d_par = _bfs(gb, tuple(partner_goal))
    pos_e = tuple(env["ego_start"]); pos_p = tuple(env["partner_start"])

    actual_wait_steps = 0
    for _ in range(k):
        if pos_p == tuple(partner_goal):
            break
        np_ = _greedy_step(gb, d_par, pos_p, rng)
        _, pos_p = _resolve(pos_e, pos_p, pos_e, np_)
        actual_wait_steps += 1

    d_pr = int(_bfs(gb, tuple(env["red_goal"]))[pos_p])
    d_pb = int(_bfs(gb, tuple(env["blue_goal"]))[pos_p])
    inferred = env["red_goal"] if d_pr < d_pb else env["blue_goal"] if d_pb < d_pr else (env["red_goal"] if rng.random() < 0.5 else env["blue_goal"])
    ego_goal = env["blue_goal"] if tuple(inferred) == tuple(env["red_goal"]) else env["red_goal"]

    d_ego = _bfs(gb, tuple(ego_goal))
    if _success(pos_e, pos_p, env["red_goal"], env["blue_goal"]):
        return True, actual_wait_steps
    for t in range(1, max_steps - actual_wait_steps + 1):
        ne = _greedy_step(gb, d_ego, pos_e, rng) if pos_e != tuple(ego_goal) else pos_e
        np_ = _greedy_step(gb, d_par, pos_p, rng) if pos_p != tuple(partner_goal) else pos_p
        pos_e, pos_p = _resolve(pos_e, pos_p, ne, np_)
        if _success(pos_e, pos_p, env["red_goal"], env["blue_goal"]):
            return True, actual_wait_steps + t
    return False, max_steps


def summarize_baselines(envs: Dict[str, dict],
                        trials: int,
                        max_steps: int,
                        wait_k: int,
                        seed: int) -> Dict[str, dict]:
    """Aggregate over layouts:
      - mean success = mean over layouts of per-layout success rate.
      - mean cost    = mean over layouts of per-layout mean steps-to-success
                       (skipping layouts where the baseline never succeeded).
    """
    baselines = {
        "nearest":    lambda e, r: _baseline_nearest(e, r, max_steps),
        "wait_react": lambda e, r: _baseline_wait_react(e, r, max_steps, wait_k),
        "random":     lambda e, r: _baseline_random(e, r, max_steps),
    }

    per_layout: Dict[str, List[Tuple[float, float]]] = {n: [] for n in baselines}
    master = random.Random(seed)
    for env in envs.values():
        for name, fn in baselines.items():
            rng = random.Random(master.randrange(2**31))
            n_succ = 0
            step_sum = 0
            for _ in range(trials):
                s, t = fn(env, rng)
                if s:
                    n_succ += 1
                    step_sum += t
            per_layout[name].append((n_succ / trials,
                                     (step_sum / n_succ) if n_succ else float("nan")))

    print(f"[3] Non-communicating baselines (mean over layouts; {trials} trials/layout, "
          f"max_steps={max_steps}, wait_k={wait_k})")
    print(f"    {'baseline':<12}  {'mean_success':>13}  {'mean_cost':>10}  {'n_cost':>6}")

    out: Dict[str, dict] = {"config": {"trials": trials, "max_steps": max_steps,
                                        "wait_k": wait_k, "seed": seed},
                             "baselines": {}}
    for name in NON_COMM_BASELINES:
        arr = np.array(per_layout[name], dtype=float)
        mean_succ = float(arr[:, 0].mean())
        cost_col  = arr[:, 1]
        ok = ~np.isnan(cost_col)
        mean_cost = float(cost_col[ok].mean()) if ok.any() else float("nan")
        n_cost = int(ok.sum())
        print(f"    {name:<12}  {mean_succ:13.3f}  {mean_cost:10.3f}  {n_cost:6d}")
        out["baselines"][name] = {
            "mean_success": mean_succ,
            "mean_cost": mean_cost if not np.isnan(mean_cost) else None,
            "n_cost": n_cost,
        }
    return out


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
    p.add_argument("--max_steps", type=int, default=40,
                   help="Per-episode step budget for baseline simulation.")
    p.add_argument("--wait_k", type=int, default=2,
                   help="Wait-and-react observation window.")
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
                                            seed=args.seed)

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
