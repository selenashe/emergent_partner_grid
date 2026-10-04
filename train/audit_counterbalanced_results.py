"""Verify empirical exposure against the allocated queue and unfinished tail."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_path
CONDITIONS = ("rnn_diverse_influence", "mlp_diverse_influence",
              "rnn_single_influence", "rnn_diverse_noinfluence")


def main(batch):
    manifest = json.loads((ROOT / "train/manifests" / f"sbatch_{batch}.json").read_text())
    plans = {}
    rows = []
    profile_rows = []
    for version in ("v1", "v2"):
        directory = ROOT / "train/train_logs" / f"{version}_balanced_training" / batch
        for condition in CONDITIONS:
            for seed in range(1, 6):
                path = directory / f"{condition}_seed{seed}_sampling_audit.json"
                audit = json.loads(path.read_text())
                schedule_path = str(resolve_path(audit["schedule_path"]))
                if schedule_path not in plans:
                    with np.load(schedule_path) as data:
                        plans[schedule_path] = {k: data[k] for k in data.files}
                plan = plans[schedule_path]
                with np.load(path.with_suffix(".npz")) as data:
                    actual = {k: data[k] for k in data.files}
                pairs = plan["capability_pairs"]
                c, n = len(pairs), int(plan["n_layouts"])
                r = plan["layouts"].shape[1]
                prefix = audit["allocated_episode_prefix"]
                full, tail = divmod(prefix, c)
                assert r == 20 and n == 1096
                assert c == (1 if condition == "rnn_single_influence" else 24)
                assert np.array_equal(actual["capability_pairs"], pairs)
                # Every full packet is allocated once to every profile.
                common = np.bincount(plan["layouts"][:full].ravel(), minlength=n)
                planned = np.tile(common, (c, 1))
                allocated = np.full(c, full, dtype=np.int64)
                for profile in plan["profile_order"][full, :tail]:
                    np.add.at(planned[profile], plan["layouts"][full], 1)
                    allocated[profile] += 1
                assert np.array_equal(allocated, actual["episodes_allocated"])
                assert int(np.ptp(allocated)) <= 1
                expected_started, expected_completed = planned.copy(), planned.copy()
                active_profiles = []
                not_started_profiles = []
                ids = actual["active_episode_ids"]
                assert len(ids) == len(np.unique(ids)) == 256
                assert ids.min() >= 0 and ids.max() < prefix
                for slot, episode in enumerate(ids):
                    packet, offset = divmod(int(episode), c)
                    profile = int(plan["profile_order"][packet, offset])
                    assert np.array_equal(actual["active_capability"][slot], pairs[profile])
                    active_profiles.append(profile)
                    idx, time = int(actual["active_round_idx"][slot]), int(actual["active_round_time"][slot])
                    assert 0 <= idx < r and 0 <= time < 100
                    if idx == 0 and time == 0:
                        not_started_profiles.append(profile)
                    # The current round has started iff its time is positive.
                    start = idx + int(time > 0)
                    np.add.at(expected_started[profile], plan["layouts"][packet, start:], -1)
                    np.add.at(expected_completed[profile], plan["layouts"][packet, idx:], -1)
                assert np.array_equal(expected_started, actual["rounds_started"])
                assert np.array_equal(expected_completed, actual["rounds_completed"])
                assert np.array_equal(allocated - np.bincount(active_profiles, minlength=c), actual["episodes_completed"])
                assert np.array_equal(allocated - np.bincount(not_started_profiles, minlength=c), actual["episodes_started"])
                steps = int(actual["environment_steps"].sum())
                assert steps == audit["collected_environment_steps"] == manifest["effective_timesteps_per_policy"]
                assert int(expected_started.sum()) == audit["rounds_started"]
                assert int(expected_completed.sum()) == audit["rounds_completed"]
                coverage = np.count_nonzero(expected_started, axis=1)
                assert np.all(coverage == n)
                round_totals = expected_started.sum(axis=1)
                completed_totals = expected_completed.sum(axis=1)
                step_totals = actual["environment_steps"].sum(axis=1)
                for profile, pair in enumerate(pairs):
                    profile_rows.append(dict(version=version, condition=condition, seed=seed,
                        partner_red_delay=int(pair[0]), partner_blue_delay=int(pair[1]),
                        episodes_allocated=int(allocated[profile]),
                        episodes_started=int(actual["episodes_started"][profile]),
                        episodes_completed=int(actual["episodes_completed"][profile]),
                        rounds_started=int(round_totals[profile]),
                        rounds_completed=int(completed_totals[profile]),
                        environment_steps=int(step_totals[profile]),
                        unique_layouts_started=int(coverage[profile]),
                        visits_per_layout_min=int(expected_started[profile].min()),
                        visits_per_layout_max=int(expected_started[profile].max())))
                rows.append(dict(version=version, condition=condition, seed=seed,
                    allocated_episodes=prefix, allocated_episodes_per_profile_min=int(allocated.min()),
                    allocated_episodes_per_profile_max=int(allocated.max()),
                    started_episodes_per_profile_min=int(actual["episodes_started"].min()),
                    started_episodes_per_profile_max=int(actual["episodes_started"].max()),
                    completed_episodes=int(actual["episodes_completed"].sum()),
                    rounds_started=int(expected_started.sum()), rounds_completed=int(expected_completed.sum()),
                    started_rounds_per_profile_min=int(round_totals.min()),
                    started_rounds_per_profile_max=int(round_totals.max()),
                    completed_rounds_per_profile_min=int(completed_totals.min()),
                    completed_rounds_per_profile_max=int(completed_totals.max()),
                    environment_steps_per_profile_min=int(step_totals.min()),
                    environment_steps_per_profile_max=int(step_totals.max()),
                    environment_steps=steps, unique_layouts_per_profile_min=int(coverage.min()),
                    profile_layout_started_count_min=int(expected_started.min()),
                    profile_layout_started_count_max=int(expected_started.max()),
                    completed_round_count_difference_between_profiles=int(np.ptp(expected_completed.sum(axis=1))),
                    largest_difference_between_profiles_on_any_layout=int(np.ptp(expected_started, axis=0).max()),
                    unfinished_rounds=int(planned.sum() - expected_completed.sum()),
                    verification="exact match to schedule prefix after subtracting active unfinished episodes",
                    source=str(path)))
    out = ROOT / "train/sampling_audits" / batch
    out.mkdir(parents=True, exist_ok=True)
    report = dict(batch=batch, verified_runs=len(rows),
                  note="Episode allocation is balanced within one episode. Actual exposure reflects the 256 unfinished episodes at the fixed-step cutoff; environment steps are not balanced across profiles.",
                  runs=rows)
    (out / "sampling_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    for filename, data in (("actual_exposure_by_run.csv", rows),
                           ("actual_exposure_by_profile.csv", profile_rows)):
        with (out / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    for version in ("v1", "v2"):
        for condition in CONDITIONS:
            rs = [row for row in rows if row["version"] == version and row["condition"] == condition]
            print(version, condition, "allocated episodes/profile range",
                  min(row["allocated_episodes_per_profile_min"] for row in rs),
                  max(row["allocated_episodes_per_profile_max"] for row in rs),
                  "max per-layout between-profile count difference",
                  max(row["largest_difference_between_profiles_on_any_layout"] for row in rs))
    print(f"Verified {len(rows)} runs; saved {out / 'sampling_verification.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch")
    main(parser.parse_args().batch)
