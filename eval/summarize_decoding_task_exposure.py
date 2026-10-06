"""Describe partner goal exposure alongside round-20 decoding, for all seeds.

Goal occupancy is an exposure proxy, not a count of partner movements or a
causal test. Uses only valid evaluation steps and pre-reset goal information.
"""

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.representation_analysis import CONDITIONS, REPO_ROOT, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="eval/representation_results/counterbalanced1096_20261002_235609")
    root = Path(parser.parse_args().results_root)
    rows = []
    for version in ("v1", "v2"):
        result_dir = root / f"{version}_balanced_training"
        metadata = json.loads((result_dir / "analysis_metadata.json").read_text())
        eval_dir = Path(metadata["arguments"]["eval_dir"])
        if not eval_dir.is_absolute():
            eval_dir = REPO_ROOT / eval_dir
        scores = pd.read_csv(result_dir / "probe_by_round_per_seed.csv")
        scores = scores[(scores.round_idx == 19) & (scores.capability_subset == "all")]
        for condition in CONDITIONS:
            for seed in range(1, 6):
                step_red = step_total = round_red = round_total = 0
                for population in ("train", "test"):
                    with h5py.File(eval_dir / f"{condition}_seed{seed}_{population}.h5", "r") as handle:
                        done = handle["dones"][:].astype(bool)
                        require(done.any(axis=1).all(), "Every episode needs a terminal step")
                        lengths = done.argmax(axis=1) + 1
                        valid = np.arange(done.shape[1])[None, :] < lengths[:, None]
                        goal = handle["partner_goal"][:]
                        step_mask = valid & ~handle["is_t0"][:].astype(bool)
                        round_mask = valid & handle["round_done"][:].astype(bool)
                        require(np.isin(goal[step_mask], [1, 2]).all(), "Unknown partner goal")
                        require(np.isin(goal[round_mask], [1, 2]).all(), "Unknown terminal-round goal")
                        require(round_mask.sum() == len(done) * 20, "Expected 20 completed rounds per episode")
                        step_red += ((goal == 1) & step_mask).sum()
                        step_total += step_mask.sum()
                        round_red += ((goal == 1) & round_mask).sum()
                        round_total += round_mask.sum()
                row = {"version": version, "condition": condition, "training_seed": seed,
                       "partner_red_step_pct": 100 * step_red / step_total,
                       "partner_red_round_end_pct": 100 * round_red / round_total,
                       "valid_non_t0_steps": step_total, "rounds": round_total}
                for target in ("d_R", "d_B"):
                    group = scores[(scores.condition == condition) & (scores.training_seed == seed)
                                   & (scores.target == target)]
                    require(len(group) == 1, "Missing/ambiguous decoding endpoint")
                    row[f"{target}_score"] = group.iloc[0].distance_accuracy
                    row[f"{target}_exact_accuracy"] = group.iloc[0].exact_accuracy
                rows.append(row)
    table = pd.DataFrame(rows)
    require(len(table) == 30, "Expected all 30 recurrent checkpoints")
    out_dir = root / "comparison"
    table.to_csv(out_dir / "task_exposure_per_seed.csv", index=False)
    columns = [c for c in table if c.endswith(("_pct", "_score", "_accuracy"))]
    summary = table.groupby(["version", "condition"])[columns].mean().reset_index()
    summary["n_training_seeds"] = 5
    summary.to_csv(out_dir / "task_exposure_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
