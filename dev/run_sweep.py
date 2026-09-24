"""Run a full-factorial sensitivity sweep over layout-generator parameters.

Fixed: `--grid_size 7`, `--n 100`, single train-only split
       (`--train_test_val_ratio 1.0 0.0 0.0`), `--max_attempts 5000`.

Swept:
    wall_density              [.15,.35], [.25,.45], [.35,.55]
    max_assignment_gap        3, 5, 7
    min_switching_cost        1, 2, 3
    geometric_pref_threshold  1, 2, 3

That's 3**4 = 81 cells. For each cell we invoke env_generator.py then
analyze_grids_lite.py; both write into dev/sweep/cells/<cell_id>/. The final
dev/sweep/index.json records per-cell params, status (ok / failed / skipped),
wall time, and output paths.

Idempotent: a cell whose `summ/*_summary.json` already exists is skipped, so
the sweep can be safely re-run after interruption.

Example:
    python dev/run_sweep.py --workers 6
    python dev/run_sweep.py --workers 6 --dry_run   # just prints the plan
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ENV_GENERATOR = HERE / "env_generator.py"
ANALYZE_LITE  = HERE / "analyze_grids_lite.py"
DEFAULT_ROOT  = HERE / "sweep"

PYTHON = sys.executable   # inherit the same interpreter this script is run with

# ---- sweep grid ----
WALL_DENSITY_VALUES: List[Tuple[float, float]] = [
    (0.15, 0.35),
    (0.25, 0.45),
    (0.35, 0.55),
]
MAX_ASSIGNMENT_GAP_VALUES: List[int]        = [3, 5, 7]
MIN_SWITCHING_COST_VALUES: List[float]      = [1, 2, 3]
GEOMETRIC_PREF_THRESHOLD_VALUES: List[int]  = [1, 2, 3]

# ---- fixed generation config ----
N              = 100
GRID_SIZE      = 7
MAX_ATTEMPTS   = 5000
SPLIT_RATIO    = (1.0, 0.0, 0.0)   # train-only

# ---- fixed analysis config ----
TRIALS         = 100
WAIT_K         = 2
ANALYSIS_SEED  = 0


def _cell_id(wd: Tuple[float, float], gap: int, sw: float, geo: int) -> str:
    """Stable filesystem-safe id for a sweep cell."""
    return f"wd{wd[0]:.2f}-{wd[1]:.2f}_gap{gap}_sw{sw:g}_geo{geo}"


def _cell_params(wd, gap, sw, geo) -> dict:
    return {
        "wall_density":             list(wd),
        "max_assignment_gap":       int(gap),
        "min_switching_cost":       float(sw),
        "geometric_pref_threshold": int(geo),
    }


def _summary_files(summ_dir: Path) -> List[Path]:
    return list(summ_dir.glob("*_summary.json"))


def _run_cell(cell_id: str, params: dict, root: Path,
              seed: int) -> dict:
    """Run generator + lite-analysis for a single cell. Returns a status dict."""
    cell_dir = root / "cells" / cell_id
    gen_dir  = cell_dir / "gen"
    summ_dir = cell_dir / "summ"
    log_path = cell_dir / "log.txt"
    cell_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    status: dict = {
        "cell_id": cell_id,
        "params": params,
        "seed": seed,
        "gen_dir": str(gen_dir.relative_to(root)),
        "summ_dir": str(summ_dir.relative_to(root)),
        "status": None,
        "elapsed_seconds": None,
        "error": None,
    }

    # Idempotence: if a summary already exists, skip this cell.
    if summ_dir.is_dir() and _summary_files(summ_dir):
        status["status"] = "skipped_cached"
        status["elapsed_seconds"] = 0.0
        return status

    # Clean gen/summ so a partial run doesn't confuse env_generator (which
    # appends indexed layouts inside layouts/<split>/).
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    if summ_dir.exists():
        shutil.rmtree(summ_dir)

    gen_cmd = [
        PYTHON, str(ENV_GENERATOR),
        "--n", str(N),
        "--master_seed", str(seed),
        "--grid_size", str(GRID_SIZE),
        "--wall_density", f"{params['wall_density'][0]}", f"{params['wall_density'][1]}",
        "--train_test_val_ratio", str(SPLIT_RATIO[0]), str(SPLIT_RATIO[1]), str(SPLIT_RATIO[2]),
        "--max_assignment_gap", str(params["max_assignment_gap"]),
        "--min_switching_cost", str(params["min_switching_cost"]),
        "--geometric_pref_threshold", str(params["geometric_pref_threshold"]),
        "--max_attempts", str(MAX_ATTEMPTS),
        "--out_dir", str(gen_dir),
    ]
    ana_cmd = [
        PYTHON, str(ANALYZE_LITE),
        str(gen_dir / "layouts"),
        "--trials", str(TRIALS),
        "--wait_k", str(WAIT_K),
        "--seed", str(ANALYSIS_SEED),
        "--out_dir", str(summ_dir),
    ]

    with open(log_path, "w") as log:
        log.write("=== gen cmd ===\n")
        log.write(" ".join(gen_cmd) + "\n\n")
        try:
            subprocess.run(gen_cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            status["status"] = "failed_generation"
            status["error"]  = f"env_generator exit {e.returncode}"
            status["elapsed_seconds"] = round(time.time() - started, 2)
            return status

        log.write("\n=== analysis cmd ===\n")
        log.write(" ".join(ana_cmd) + "\n\n")
        try:
            subprocess.run(ana_cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            status["status"] = "failed_analysis"
            status["error"]  = f"analyze_grids_lite exit {e.returncode}"
            status["elapsed_seconds"] = round(time.time() - started, 2)
            return status

    status["status"] = "ok"
    status["elapsed_seconds"] = round(time.time() - started, 2)
    return status


def _build_grid() -> List[Tuple[str, dict]]:
    combos = list(itertools.product(WALL_DENSITY_VALUES,
                                    MAX_ASSIGNMENT_GAP_VALUES,
                                    MIN_SWITCHING_COST_VALUES,
                                    GEOMETRIC_PREF_THRESHOLD_VALUES))
    return [(_cell_id(wd, gap, sw, geo), _cell_params(wd, gap, sw, geo))
            for wd, gap, sw, geo in combos]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help="Sweep root directory.")
    p.add_argument("--workers", type=int, default=6,
                   help="Parallel workers.")
    p.add_argument("--seed", type=int, default=0,
                   help="Master seed for env generation, offset per cell by index.")
    p.add_argument("--dry_run", action="store_true",
                   help="Just print the plan and exit.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    grid = _build_grid()
    total = len(grid)
    print(f"planned {total} cells; workers={args.workers}; root={args.root}")
    if args.dry_run:
        for cid, params in grid[:5]:
            print(f"  {cid}  {params}")
        print(f"  ... ({total - 5} more)")
        return

    started = time.time()
    results: Dict[str, dict] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_run_cell, cid, params, args.root, args.seed + i): cid
            for i, (cid, params) in enumerate(grid)
        }
        done = 0
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"cell_id": cid, "status": "crashed", "error": repr(e)}
            results[cid] = res
            done += 1
            elapsed = res.get("elapsed_seconds", "?")
            print(f"[{done:>3d}/{total}] {res['status']:<20s} {cid}  ({elapsed}s)")

    # Preserve grid order in the index.
    ordered = {cid: results[cid] for cid, _ in grid}

    n_ok       = sum(1 for r in ordered.values() if r.get("status") == "ok")
    n_cached   = sum(1 for r in ordered.values() if r.get("status") == "skipped_cached")
    n_failed   = total - n_ok - n_cached

    index = {
        "config": {
            "n_per_cell": N, "grid_size": GRID_SIZE, "max_attempts": MAX_ATTEMPTS,
            "split_ratio": list(SPLIT_RATIO),
            "trials": TRIALS, "wait_k": WAIT_K, "analysis_seed": ANALYSIS_SEED,
            "master_seed": args.seed,
        },
        "grid": {
            "wall_density":             [list(v) for v in WALL_DENSITY_VALUES],
            "max_assignment_gap":       MAX_ASSIGNMENT_GAP_VALUES,
            "min_switching_cost":       MIN_SWITCHING_COST_VALUES,
            "geometric_pref_threshold": GEOMETRIC_PREF_THRESHOLD_VALUES,
        },
        "counts": {"total": total, "ok": n_ok, "cached": n_cached, "failed": n_failed},
        "wall_seconds": round(time.time() - started, 2),
        "cells": ordered,
    }
    idx_path = args.root / "index.json"
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2)

    print()
    print(f"summary:  ok={n_ok}  cached={n_cached}  failed={n_failed}  "
          f"wall={index['wall_seconds']}s")
    if n_failed:
        print("failed cells:")
        for cid, r in ordered.items():
            if r.get("status") not in ("ok", "skipped_cached"):
                print(f"  {cid}  status={r.get('status')}  err={r.get('error')}")
    print(f"index written to {idx_path}")


if __name__ == "__main__":
    main()
