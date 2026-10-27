"""Plot unsmoothed completed-episode returns from the final training update."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.plot_episode_return_dots import CONDITION_STYLES, plot_episode_return_dots

VERSIONS = ("v1_matched", "v2", "v3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=Path, default=ROOT / "train/training_curves/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/updates.csv")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    source = args.updates.resolve()
    out_dir = (args.out_dir or source.parent / "final_update_dots").resolve()
    expected = {(v, c, s) for v in VERSIONS for c, _, _ in CONDITION_STYLES for s in range(1, 6)}
    final = {}
    with source.open(newline="") as file:
        for row in csv.DictReader(file):
            key = (row["version"], row["condition"], int(row["seed"]))
            if key not in expected:
                continue
            update = int(row["update"])
            if key not in final or update > final[key]["update"]:
                final[key] = {
                    "version": key[0], "condition": key[1], "seed": key[2],
                    "update": update, "env_steps": int(row["env_steps"]),
                    "episode_return": float(row["episode_return"]),
                    "episode_steps": float(row["episode_steps"]),
                }
    if set(final) != expected:
        raise ValueError(f"Missing training policies: {expected-set(final)}")
    if {(r["update"], r["env_steps"]) for r in final.values()} != {(915, 59965440)}:
        raise ValueError("All policies must reach the original final training budget")
    if any(not math.isfinite(r["episode_return"]) or r["episode_steps"] <= 0 for r in final.values()):
        raise ValueError("Every final update must have a valid completed-episode mean")
    aggregates = []
    for version in VERSIONS:
        for condition, _, _ in CONDITION_STYLES:
            values = [final[(version, condition, s)]["episode_return"] for s in range(1, 6)]
            aggregates.append({
                "version": version, "condition": condition, "n_seeds": len(values),
                "episode_return_mean": statistics.mean(values),
                "episode_return_sd": statistics.stdev(values),
            })
    report = {
        "population": "train", "seeds": list(range(1, 6)), "aggregates": aggregates,
        "plot_title": "Episode return at final training update",
        "plot_subtitle": "Update 915 · 59,965,440 environment steps · unsmoothed training returns",
        "source_updates": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "metric": "Per-policy mean return of full 20-round episodes completed in the final rollout collection update; dots weight all five policies equally.",
        "training_layouts": "Same frozen 1,096 grid layouts used during training.",
        "training_partners": {"rnn_single_influence": [[1, 4]], "other_conditions": "24-profile training population"},
        "limitations": [
            "This is the final logged collection update, not a new evaluation of the post-update checkpoint.",
            "This uses raw final-update returns, not the trailing-window endpoint from the smoothed curves.",
            "Training returns were printed to three decimals; exact episode counts per update were not persisted.",
            "Single-partner and diverse-partner conditions have different training partner populations.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, rows in [("per_seed.csv", [final[key] for key in sorted(final)]), ("aggregates.csv", aggregates)]:
        with (out_dir / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for path in plot_episode_return_dots(report, out_dir, version_order=VERSIONS):
        print(path)
    for entry in aggregates:
        print(f"{entry['version']} {entry['condition']}: {entry['episode_return_mean']:.3f} ± {entry['episode_return_sd']:.3f}")


if __name__ == "__main__":
    main()
