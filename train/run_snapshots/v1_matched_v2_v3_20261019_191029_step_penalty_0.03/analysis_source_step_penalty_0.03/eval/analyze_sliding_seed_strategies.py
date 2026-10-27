"""Audit partner-conditioned allocation strategies in all sliding policies.

Read scalar rollout fields only. Cache terminal-inclusive round summaries on
scratch, then compare each of five learner seeds within every t/condition cell.
Goal-specialization metrics are behavioral proxies, not geometry-aware oracles.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import sys
import uuid

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_record_paths

CONDITIONS = ("rnn_diverse_influence", "mlp_diverse_influence", "rnn_single_influence", "rnn_diverse_noinfluence")
CACHE_VERSION = 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def round_arrays(handle, t, influence, reversible, rounds=20):
    """Keep the terminal transition and drop fixed-scan padding."""
    names = ("dones", "round_done", "success", "rewards", "round_idx", "round_time",
             "is_t0", "partner_assignment", "ego_alloc_action", "layout_idx")
    a = {name: np.asarray(handle[name]) for name in names}
    pool, profile = np.asarray(handle["capability_pool"]), np.asarray(handle["capability_index_per_ep"])
    capability = np.asarray(handle["capability"])
    require(a["dones"].ndim == 2 and a["dones"].any(axis=1).all(), "Missing terminal episode transition")
    shape = a["dones"].shape
    require(all(value.shape == shape for value in a.values()), "Scalar rollout shapes disagree")
    require(pool.ndim == 2 and pool.shape[1] == 2 and profile.shape == (shape[0],), "Invalid profile metadata")
    require(len(set(map(tuple, pool.tolist()))) == len(pool), "Duplicate capability profiles")
    require(np.isin(profile, np.arange(len(pool))).all(), "Invalid profile index")
    lengths = a["dones"].argmax(axis=1) + 1
    alive = np.arange(shape[1])[None, :] < lengths[:, None]
    require(capability.shape == shape + (2,), "Invalid capability shape")
    require(not ((capability != pool[profile, None, :]) & alive[:, :, None]).any(), "Episode capability changed")
    require(np.isfinite(a["rewards"][alive]).all(), "Nonfinite valid rewards")
    endings = a["round_done"].astype(bool) & alive
    commitments = a["is_t0"].astype(bool) & alive
    require((endings.sum(axis=1) == rounds).all() and (commitments.sum(axis=1) == rounds).all(), "Missing/extra round endings or commitments")
    ep = np.broadcast_to(np.arange(shape[0])[:, None], shape)
    rid = a["round_idx"].astype(int)
    ordered = np.broadcast_to(np.arange(rounds), (shape[0], rounds))
    require(np.array_equal(rid[endings].reshape(shape[0], rounds), ordered), "Endings not in round order")
    require(np.array_equal(rid[commitments].reshape(shape[0], rounds), ordered), "Commitments not in round order")
    require((a["round_time"][commitments] == t + 1).all(), "Incorrect commitment timing")
    require(np.isin(a["partner_assignment"][alive], (0, 1, 2)).all(), "Unknown assignment code")
    require(np.isin(a["partner_assignment"][commitments | endings], (1, 2)).all(), "Unset committed assignment")
    if influence:
        require(np.array_equal(a["ego_alloc_action"][commitments], a["partner_assignment"][commitments]), "Influence action and assignment disagree")
    post = alive & (a["round_time"] >= t + 1)
    require(np.isin(a["partner_assignment"][post], (1, 2)).all(), "Unset postcommit assignment")
    # Assignment denotes ego's goal; partner receives its complementary goal.
    actual_red = a["partner_assignment"] == 2
    labels = ep * rounds + rid
    def sum_round(value, mask=alive):
        return np.bincount(labels[mask], weights=np.asarray(value)[mask], minlength=shape[0] * rounds).reshape(shape[0], rounds)
    steps = a["round_time"][endings].reshape(shape[0], rounds)
    require(np.array_equal(steps.sum(axis=1), lengths), "Round lengths do not recover episode lengths")
    switches = np.zeros(shape, bool)
    switches[:, 1:] = ((a["partner_assignment"][:, 1:] != a["partner_assignment"][:, :-1]) &
                       (rid[:, 1:] == rid[:, :-1]) & alive[:, 1:] &
                       (a["round_time"][:, 1:] > t + 1))
    if "assignment_changed" in handle:
        saved = np.asarray(handle["assignment_changed"], dtype=bool)
        valid = alive & (a["round_time"] > t + 1)
        require(np.array_equal(saved[valid], switches[valid]), "Saved switch flags disagree with actual assignment changes")
    if not reversible or not influence:
        require(not switches.any(), "Uncontrollable/irreversible assignment switched")
    result = dict(pool=pool, profile=profile, episode_steps=lengths,
                  episode_return=np.where(alive, a["rewards"], 0).sum(axis=1, dtype=np.float64),
                  layout=a["layout_idx"][commitments].reshape(shape[0], rounds),
                  initial_red=actual_red[commitments].reshape(shape[0], rounds),
                  final_red=actual_red[endings].reshape(shape[0], rounds),
                  success=a["success"][endings].reshape(shape[0], rounds).astype(bool),
                  steps=steps, returns=sum_round(a["rewards"]), switches=sum_round(switches))
    counts = sum_round(np.ones(shape), post)
    require((counts > 0).all(), "No postcommit observations")
    result["red_occupancy"] = sum_round(actual_red, post) / counts
    require(np.array_equal(a["layout_idx"][endings].reshape(shape[0], rounds), result["layout"]), "Layout changed inside round")
    # The observed reward identity uses the successful terminal transition.
    require(np.allclose(result["returns"], 1.01 * result["success"] - .01 * steps, atol=2e-6), "Round reward formula mismatch")
    return result


def adjusted_gap(red, layout, orientation, round_ids, episode_indices=None):
    """Within-layout/parity RED-choice contrast, weighted by overlap counts."""
    if episode_indices is not None:
        red, layout, orientation = red[episode_indices], layout[episode_indices], orientation[episode_indices]
    strata = (layout * 2 + np.asarray(round_ids)[None, :] % 2).astype(int)
    size = int(strata.max()) + 1
    counts, positives = [], []
    for direction in (True, False):
        mask = orientation == direction
        counts.append(np.bincount(strata[mask].ravel(), minlength=size))
        positives.append(np.bincount(strata[mask].ravel(), weights=red[mask].ravel(), minlength=size))
    common = (counts[0] > 0) & (counts[1] > 0)
    if not common.any():
        return float("nan"), 0.
    left, right = counts[0][common], counts[1][common]
    weights = left * right / (left + right)
    difference = positives[0][common] / left - positives[1][common] / right
    return float(np.average(difference, weights=weights)), float((left.sum() + right.sum()) / red.size)


def bootstrap_gaps(data, samples, seed):
    """Resample whole 20-round episodes within each fixed capability profile."""
    rng = np.random.default_rng(seed)
    profile, pool = data["profile"], data["pool"]
    grouped = np.array([np.flatnonzero(profile == i) for i in range(len(pool))])
    require(grouped.shape == (len(pool), 20), "Require twenty episodes per profile")
    draws = rng.integers(0, 20, size=(samples, len(pool), 20))
    selected = np.take_along_axis(np.broadcast_to(grouped, draws.shape), draws, axis=2).reshape(samples, -1)
    orientation = pool[profile, 0] < pool[profile, 1]
    late = np.arange(10, 20)
    stats = {}
    for stage in ("initial", "final"):
        red = data[stage + "_red"][:, late]
        estimates = [adjusted_gap(red, data["layout"][:, late], orientation, late, indices)[0] for indices in selected]
        stats[stage + "_late_gap_ci_low"], stats[stage + "_late_gap_ci_high"] = map(float, np.nanquantile(estimates, (.025, .975)))
    return stats


def profile_and_seed_rows(record, population, data, samples):
    pool, profile = data["pool"], data["profile"]
    orientation = pool[profile, 0] < pool[profile, 1]
    meta = {key: record[key] for key in ("variant", "t", "reversibility", "condition", "seed", "influence")}
    meta["population"] = population
    rows = []
    for index, (dr, db) in enumerate(pool):
        select = profile == index
        require(select.sum() == 20 and dr != db, "Unequal/tied profile episodes")
        row = dict(meta, delay_red=int(dr), delay_blue=int(db), orientation="faster_RED" if dr < db else "faster_BLUE", n_episodes=20,
                   episode_return=float(data["episode_return"][select].mean()), success=float(data["success"][select].mean()),
                   episode_steps=float(data["episode_steps"][select].mean()), switches_per_round=float(data["switches"][select].mean()))
        for stage in ("initial", "final"):
            red = data[stage + "_red"][select]
            row[stage + "_partner_RED"] = float(red.mean())
            row[stage + "_partner_faster"] = float((red == (dr < db)).mean())
            row[stage + "_late_partner_RED"] = float(red[:, 10:].mean())
        rows.append(row)
    summary = dict(meta, episode_return=float(data["episode_return"].mean()), success=float(data["success"].mean()),
                   episode_steps=float(data["episode_steps"].mean()), switches_per_round=float(data["switches"].mean()),
                   return_faster_RED=float(data["episode_return"][orientation].mean()),
                   return_faster_BLUE=float(data["episode_return"][~orientation].mean()))
    for stage in ("initial", "final"):
        red = data[stage + "_red"]
        for suffix, rounds in (("", np.arange(20)), ("_first", np.array([0])), ("_late", np.arange(10, 20))):
            selected = red[:, rounds]
            left, right = selected[orientation].mean(), selected[~orientation].mean()
            summary[stage + suffix + "_red_given_faster_RED"] = float(left)
            summary[stage + suffix + "_red_given_faster_BLUE"] = float(right)
            summary[stage + suffix + "_raw_gap"] = float(left - right)
            gap, coverage = adjusted_gap(selected, data["layout"][:, rounds], orientation, rounds)
            summary[stage + suffix + "_adjusted_gap"] = gap
            summary[stage + suffix + "_overlap_fraction"] = coverage
        summary[stage + "_partner_RED"] = float(red.mean())
        summary[stage + "_partner_faster"] = float((red == orientation[:, None]).mean())
    rng_seed = int(hashlib.sha256((record["variant"] + record["condition"] + str(record["seed"]) + population).encode()).hexdigest()[:8], 16)
    summary.update(bootstrap_gaps(data, samples, rng_seed))
    for stage in ("initial", "final"):
        gap = summary[stage + "_late_adjusted_gap"]
        summary[stage + "_positive_specialization"] = bool(record["influence"] and gap >= .20 and summary[stage + "_late_gap_ci_low"] > 0)
        summary[stage + "_negative_specialization"] = bool(record["influence"] and gap <= -.20 and summary[stage + "_late_gap_ci_high"] < 0)
        summary[stage + "_strong_color_bias"] = bool(summary[stage + "_partner_RED"] >= .90 or summary[stage + "_partner_RED"] <= .10)
    return rows, summary


def worker(task):
    record, population, out, samples = task
    path = Path(record["prefix"] + "_" + population + ".h5").resolve(strict=True)
    summary_path = Path(record["prefix"] + "_summary.json")
    source = dict(path=str(path), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                  summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(), cache_version=CACHE_VERSION,
                  bootstrap_samples=samples)
    stem = out / "cache" / record["variant"] / record["condition"] / ("seed" + str(record["seed"]) + "_" + population)
    meta_path, cache_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
    if meta_path.exists() and cache_path.exists():
        cached = json.loads(meta_path.read_text())
        if cached["source"] == source:
            return cached["profiles"], cached["seed"]
    with h5py.File(path) as handle:
        if not record["baseline"]:
            for key in ("t", "reversible", "condition", "seed"):
                require(handle.attrs[key] == record[key], "Rollout metadata disagrees with manifest: " + str(path))
        data = round_arrays(handle, record["t"], record["influence"], record["reversible"])
    populations = runpy.run_path(str(ROOT / "jaxmarl/environments/coordination_grid/capability_populations.py"))
    expected = populations["TRAIN_CAPABILITY_PAIRS" if population == "train" else "TEST_CAPABILITY_PAIRS"]
    require(set(map(tuple, data["pool"].tolist())) == set(expected), "Unexpected capability population")
    original = json.loads(summary_path.read_text())[population]
    require(np.isclose(data["episode_return"].mean(), original["mean_ep_return"], rtol=1e-6, atol=2e-5), "Return does not reproduce saved summary")
    require(np.isclose(data["success"].mean(), original["round_success_rate"], atol=2e-7), "Success does not reproduce saved summary")
    require(data["success"].size == original["n_rounds"], "Round count mismatch")
    require(np.allclose(data["success"].mean(axis=0), original["per_round_success_rate"], atol=2e-7), "Round success curve mismatch")
    profiles, seed = profile_and_seed_rows(record, population, data, samples)
    stem.parent.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_name(cache_path.name + "." + uuid.uuid4().hex)
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **data)
    temp.replace(cache_path)
    meta_path.write_text(json.dumps(dict(source=source, profiles=profiles, seed=seed), indent=2) + "\n")
    return profiles, seed


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def group_analysis(profile_rows, seed_rows):
    lookup = {(r["variant"], r["condition"], r["population"], r["seed"]): r for r in seed_rows}
    groups, pairs = [], []
    for population in ("test", "train"):
        for mode in ("irreversible", "reversible"):
            for t in range(11):
                variant = f"t{t}_{mode}"
                for condition in CONDITIONS:
                    seeds = [lookup[variant, condition, population, seed] for seed in range(1, 6)]
                    returns = np.array([s["episode_return"] for s in seeds])
                    success = np.array([s["success"] for s in seeds])
                    steps = np.array([s["episode_steps"] for s in seeds])
                    profile_keys = sorted({(r["delay_red"], r["delay_blue"]) for r in profile_rows if r["variant"] == variant and r["condition"] == condition and r["population"] == population})
                    pro = {(r["seed"], r["delay_red"], r["delay_blue"]): r for r in profile_rows if r["variant"] == variant and r["condition"] == condition and r["population"] == population}
                    matrix = np.array([[pro[(s,) + key]["episode_return"] for key in profile_keys] for s in range(1, 6)])
                    residual = matrix - matrix.mean(axis=0, keepdims=True)
                    seed_ss = len(profile_keys) * np.sum((returns - returns.mean()) ** 2)
                    interaction_ss = np.sum((residual - residual.mean(axis=1, keepdims=True)) ** 2)
                    base = dict(variant=variant, t=t, reversibility=mode, condition=condition, population=population,
                                episode_return_mean=float(returns.mean()), episode_return_sd=float(returns.std(ddof=1)),
                                minimum_return_seed=int(np.argmin(returns) + 1), minimum_return=float(returns.min()), maximum_return=float(returns.max()),
                                return_sd_without_lowest_seed=float(np.delete(returns, np.argmin(returns)).std(ddof=1)),
                                weakest_seed_fraction_of_return_ss=float((returns.min() - returns.mean()) ** 2 / np.sum((returns - returns.mean()) ** 2)),
                                uniform_seed_shift_ss=float(seed_ss), seed_by_profile_interaction_ss=float(interaction_ss),
                                profile_specific_fraction_of_between_seed_profile_ss=float(interaction_ss / (seed_ss + interaction_ss)),
                                return_variance_from_success=float(20.2 ** 2 * success.var(ddof=1)),
                                return_variance_from_steps=float(.01 ** 2 * steps.var(ddof=1)),
                                return_variance_covariance_term=float(-2 * 20.2 * .01 * np.cov(success, steps, ddof=1)[0, 1]),
                                influence=seeds[0]["influence"])
                    for stage in ("initial", "final"):
                        base[stage + "_positive_specialization_seeds"] = sum(s[stage + "_positive_specialization"] for s in seeds)
                        base[stage + "_strong_color_bias_seeds"] = sum(s[stage + "_strong_color_bias"] for s in seeds)
                        base[stage + "_mean_late_adjusted_gap"] = float(np.mean([s[stage + "_late_adjusted_gap"] for s in seeds]))
                    distances = {"initial": [], "final": []}
                    for s in range(1, 6):
                        for u in range(s + 1, 6):
                            pair = dict(variant=variant, t=t, reversibility=mode, condition=condition, population=population, seed_a=s, seed_b=u,
                                        return_difference=float(returns[s - 1] - returns[u - 1]),
                                        centered_profile_return_rmse=float(np.sqrt(np.mean(((matrix[s - 1] - returns[s - 1]) - (matrix[u - 1] - returns[u - 1])) ** 2))))
                            for stage in ("initial", "final"):
                                difference = np.mean([abs(pro[(s,) + key][stage + "_partner_RED"] - pro[(u,) + key][stage + "_partner_RED"]) for key in profile_keys])
                                pair[stage + "_mean_profile_assignment_difference"] = float(difference)
                                distances[stage].append(difference)
                            pairs.append(pair)
                    for stage in distances:
                        base[stage + "_mean_between_seed_assignment_difference"] = float(np.mean(distances[stage]))
                    groups.append(base)
    return groups, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--limit", type=int, help="Extraction-only preflight; no all-experiment reports")
    args = parser.parse_args()
    records, manifest_hashes = {}, {}
    for path in args.manifests:
        manifest_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = resolve_record_paths(json.loads(path.read_text()))
        for record in manifest["evaluation_records"]:
            key = (record["variant"], record["condition"], record["seed"])
            if key in records:
                require(records[key] == record and record["baseline"], "Inconsistent duplicated policy")
            records[key] = record
    ordered = sorted(records.values(), key=lambda r: (CONDITIONS.index(r["condition"]), r["reversibility"], r["t"], r["seed"]))
    require(len(ordered) == 440, "Require all 440 unique sliding policies")
    if args.limit:
        ordered = ordered[:args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(record, population, args.out_dir, args.bootstrap_samples) for record in ordered for population in ("test", "train")]
    profiles, seeds = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            pro, seed = future.result()
            profiles.extend(pro); seeds.append(seed)
            if index % 20 == 0 or index == len(tasks):
                print(f"Validated {index}/{len(tasks)} scalar rollouts; last {seed['variant']} {seed['condition']} seed{seed['seed']} {seed['population']}", flush=True)
    if args.limit:
        print("Extraction preflight passed", flush=True)
        return
    groups, pairs = group_analysis(profiles, seeds)
    for name, rows in (("profile_strategy_per_seed.csv", profiles), ("seed_strategy_summary.csv", seeds),
                       ("condition_strategy_summary.csv", groups), ("seed_pair_strategy_distances.csv", pairs)):
        save_csv(args.out_dir / name, sorted(rows, key=lambda r: (r["condition"], r["reversibility"], r["t"], r["population"], r.get("seed", r.get("seed_a", 0)))))
    metadata = dict(completed_utc=datetime.now(timezone.utc).isoformat(), n_policies=440, n_rollouts=len(seeds), n_profile_seed_rows=len(profiles),
                    bootstrap_samples=args.bootstrap_samples, bootstrap_unit="whole episode resampled within each of the fixed capability profiles; pointwise intervals for each fixed trained policy",
                    role_gap="P(partner assigned RED | d_R<d_B) - P(partner assigned RED | d_R>d_B)",
                    adjustment="Within initial-layout x round-parity strata with both orientations; overlap weights n_RED*n_BLUE/(n_RED+n_BLUE)",
                    late_rounds="zero-based rounds 10..19", positive_specialization_rule="influence enabled, late adjusted orientation gap >=0.20 and episode-bootstrap lower 95% bound >0; descriptive screening rule, not familywise inference",
                    color_bias_rule="actual partner RED fraction <=0.10 or >=0.90", profile_variance_decomposition="Seed shifts vs seed-by-profile interaction after removing shared profile difficulty; this is not a causal explained fraction of pooled-return error bars",
                    weakest_seed_diagnostic="Remove the lowest-return seed after observing results; a descriptive sensitivity diagnostic, not an unbiased SD estimate",
                    limitations=["Goal-speed specialization is not a geometry/collision/switch-cost-aware optimal allocation oracle.", "Allocation data cannot fully identify ego navigation strategies: no positions or movement actions are stored.", "No-influence actual role assignment is environment imposed, not a learned allocation strategy.", "Final roles and switching can differ from initial commitments in reversible policies.", "Rollout-bootstrap confidence intervals are conditional on fixed trained policies and evaluated profiles, not training-seed population intervals.", "These are exploratory associations and do not establish that allocation choices cause the return differences."],
                    manifest_sha256=manifest_hashes)
    (args.out_dir / "audit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Strategy tables complete:", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
