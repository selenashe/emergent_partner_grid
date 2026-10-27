"""Behavior plots and preserved affine decoding for the sliding observation experiment.

The manifest identifies the requested observation times and two read-only t=0
sliding irreversible/v3 baselines, each with four model conditions and learner seeds 1..5. Probe
fits reuse representation_analysis functions and never load the RL environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_record_paths, save_numpy_archive
from eval import representation_analysis as rep

SEEDS = tuple(range(1, 6))
CONDITIONS = (
    "rnn_diverse_influence", "rnn_single_influence", "mlp_diverse_influence",
    "rnn_diverse_noinfluence",
)
CONDITION_LABELS = {
    "rnn_diverse_influence": "Diverse RNN + influence",
    "rnn_single_influence": "Single RNN + influence",
    "mlp_diverse_influence": "Diverse MLP + influence",
    "rnn_diverse_noinfluence": "Diverse RNN without influence",
}
BEHAVIOR_METRICS = (
    "success", "episode_return", "episode_steps", "successful_completion_steps",
    "switches_per_round",
)
CAVEATS = [
    "The t=0 baselines reuse existing counterbalanced sliding irreversible/v3 policies and rollouts; no baseline retraining.",
    "Exactly five learner seeds (1..5) are used in every condition, including both existing baselines.",
    "Layouts remain familiar: this is a partner capability evaluation, not a layout holdout.",
    "Each policy is evaluated categorically on 24 familiar and 22 novel profiles, with 20 episodes per profile and 20 rounds per episode.",
    "Observation transitions consume the existing round budget and reward schedule; episode returns include their costs and all failures.",
    "The irreversible branch retains sliding_irreversible_baseline PPO recurrent replay/reset alignment; the reversible branch retains v3 alignment.",
    "No-influence initialization retains the inherited sliding_irreversible_baseline round-parity assignment and v3 random assignment.",
    "Legacy t=0 geometry-oracle regret and online speed-adherence diagnostics are excluded for delayed commitment; switch rates are recomputed from valid postcommit transitions only.",
    "Seed standard deviations and paired nominal-seed contrasts are descriptive; nominal seed matching does not establish identical experiences.",
]


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def manifest_variants(manifest: dict) -> tuple[str, ...]:
    times = manifest.get("new_times", [1, 2, 3, 4])
    rep.require(bool(times) and all(type(t) is int and t > 0 for t in times)
                and times == sorted(set(times)), "Require distinct positive observation times in sorted order")
    return tuple(f"t{t}_{mode}" for t in (0, *times) for mode in ("irreversible", "reversible"))


def load_manifest(path: Path) -> tuple[dict, list[dict]]:
    """Reject omitted cells, duplicate policies, changed seed counts and path ambiguity."""
    manifest = resolve_record_paths(json.loads(path.read_text()))
    rep.require(manifest.get("action_selection", "categorical") == "categorical",
                "Sliding evaluation must retain categorical action sampling")
    rep.require(tuple(manifest.get("evaluation_seeds", ())) == SEEDS,
                "Sliding experiment requires exactly learner seeds 1..5")
    records = manifest.get("evaluation_records", [])
    variants = manifest_variants(manifest)
    expected = {(v, c, s) for v in variants for c in CONDITIONS for s in SEEDS}
    actual = [(r["variant"], r["condition"], r["seed"]) for r in records]
    rep.require(len(actual) == len(set(actual)), "Duplicate evaluation policy records")
    rep.require(set(actual) == expected,
                f"Expected all {len(expected)} policies ({len(variants) - 2} new variants + 2 t=0 baselines); "
                f"missing={sorted(expected-set(actual))}, extra={sorted(set(actual)-expected)}")
    for record in records:
        variant = record["variant"]
        t, mode = variant.split("_", 1)
        reversible = mode == "reversible"
        rep.require(record["t"] == int(t[1:]) and bool(record["reversible"]) == reversible,
                    f"{variant}: inconsistent observation-time/reversibility metadata")
        rep.require(bool(record.get("baseline", False)) == (record["t"] == 0),
                    f"{variant}: t=0 must reuse a baseline; t>0 must identify a new policy")
        rep.require(bool(record.get("reversibility", mode) == mode),
                    f"{variant}: inconsistent reversibility label")
        rep.require("prefix" in record and Path(record["prefix"]).is_absolute(),
                    f"{variant}: require an explicit absolute rollout prefix")
    order = {v: i for i, v in enumerate(variants)}
    records = sorted(records, key=lambda r: (order[r["variant"]], CONDITIONS.index(r["condition"]), r["seed"]))
    return manifest, records


def policy_metadata(record: dict) -> dict:
    return {"variant": record["variant"], "t": record["t"],
            "reversible": bool(record["reversible"]),
            "reversibility": "reversible" if record["reversible"] else "irreversible",
            "condition": record["condition"], "seed": record["seed"],
            "baseline": bool(record["baseline"])}


def rollout_paths(record: dict) -> dict[str, Path]:
    return {population: Path(f"{record['prefix']}_{population}.h5")
            for population in ("train", "test")}


def _attr(value):
    return value.decode() if isinstance(value, bytes) else value


def validate_rollout_metadata(handle, record: dict, path: Path) -> None:
    expected_protocol = "online_v3" if record["reversible"] else "fixed_allocation"
    rep.require(_attr(handle.attrs.get("allocation_protocol", "fixed_allocation")) == expected_protocol,
                f"{path}: allocation protocol disagrees with manifest")
    if not record["baseline"]:
        for key in ("t", "reversible", "condition", "seed"):
            rep.require(key in handle.attrs and _attr(handle.attrs[key]) == record[key],
                        f"{path}: missing/mismatched {key} metadata")
    timing = _attr(handle.attrs.get("hidden_state_timing", "post_observation_before_action"))
    rep.require(timing == "post_observation_before_action", f"{path}: unsupported hidden-state timing")


def read_behavior_policy(record: dict, population: str) -> dict:
    summary_path = Path(f"{record['prefix']}_summary.json")
    summary = json.loads(summary_path.read_text())[population]
    path = rollout_paths(record)[population]
    expected_episodes = (24 if population == "train" else 22) * 20
    rep.require(summary["n_episodes"] == expected_episodes and
                summary["n_rounds"] == expected_episodes * 20,
                f"{summary_path}: invalid evaluation episode/round counts")
    with h5py.File(path, "r") as handle:
        validate_rollout_metadata(handle, record, path)
        dones = np.asarray(handle["dones"], dtype=bool)
        endings = np.asarray(handle["round_done"], dtype=bool)
        changes = np.asarray(handle["assignment_changed"], dtype=bool) if "assignment_changed" in handle else None
        round_time = np.asarray(handle["round_time"]) if "round_time" in handle else None
        assignment = np.asarray(handle["partner_assignment"]) if "partner_assignment" in handle else None
        round_idx = np.asarray(handle["round_idx"]) if "round_idx" in handle else None
    rep.require(dones.ndim == 2 and dones.shape[0] == expected_episodes and
                endings.shape == dones.shape and dones.any(axis=1).all(),
                f"{path}: incomplete or malformed episode scan")
    lengths = dones.argmax(axis=1) + 1
    alive = np.arange(dones.shape[1])[None, :] < lengths[:, None]
    rep.require(np.all((endings & alive).sum(axis=1) == 20), f"{path}: missing valid round endings")
    episode_steps = float(lengths.mean())
    inferred_steps = (1.01 * 20 * summary["round_success_rate"] - summary["mean_ep_return"]) / .01
    rep.require(abs(episode_steps - inferred_steps) < .02,
                f"{path}: rollout steps disagree with the unchanged reward/return formula")
    per_round = np.asarray(summary["per_round_success_rate"], dtype=float)
    rep.require(per_round.shape == (20,) and np.isfinite(per_round).all(),
                f"{summary_path}: malformed per-round success")
    if not record["reversible"]:
        switches = 0.0
        if not record["baseline"]:
            rep.require(assignment is not None and round_idx is not None and round_time is not None,
                        f"{path}: delayed irreversible commitment requires saved assignment and round timing")
            rep.require(assignment.shape == round_idx.shape == round_time.shape == dones.shape,
                        f"{path}: invalid irreversible assignment timing shape")
            for round_id in range(20):
                committed = alive & (round_idx == round_id) & ((round_time - 1) >= record["t"])
                rep.require(committed.any(axis=1).all(), f"{path}: missing commitment in round{round_id}")
                high = np.where(committed, assignment, -1).max(axis=1)
                low = np.where(committed, assignment, np.iinfo(np.int32).max).min(axis=1)
                rep.require(np.array_equal(high, low), f"{path}: irreversible assignment changed after commitment")
    elif changes is not None and round_time is not None:
        rep.require(changes.shape == round_time.shape == dones.shape, f"{path}: invalid switch timing shape")
        postcommit = alive & ((round_time - 1) > record["t"])
        switches = float(changes[postcommit].sum() / (expected_episodes * 20))
    else:
        rep.require(record["baseline"], f"{path}: delayed reversible commitment requires saved assignment_changed and round_time")
        switches = summary.get("assignment_switches_per_round", 0.0)
    return {**policy_metadata(record), "population": population,
            "success": summary["round_success_rate"], "episode_return": summary["mean_ep_return"],
            "episode_steps": episode_steps,
            "successful_completion_steps": summary["mean_successful_completion_time"],
            "switches_per_round": switches,
            "per_round_success_rate": per_round.tolist(),
            "summary_path": str(summary_path), "rollout_path": str(path)}


def summarize_behavior(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, condition, population), group in table.groupby(["variant", "condition", "population"], sort=False):
        rep.require(sorted(group.seed.tolist()) == list(SEEDS), "Behavior summary must contain exactly five learner seeds")
        first = group.iloc[0]
        row = dict(variant=variant, t=int(first.t), reversible=bool(first.reversible),
                   reversibility=first.reversibility, condition=condition, population=population, n_seeds=5)
        for metric in BEHAVIOR_METRICS:
            values = group[metric].to_numpy(dtype=float)
            # Successful-only completion time can be undefined if a seed has no successes.
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_sd"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_behavior(table: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(CONDITIONS, ("#386cb0", "#f28e2b", "#7a5195", "#2b9b7c")))
    metrics = (("success", 100, "Round success (%)"),
               ("episode_return", 1, "Episode return"),
               ("episode_steps", 1, "Steps per 20-round episode"))
    for population in ("train", "test"):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
        for ri, mode in enumerate(("irreversible", "reversible")):
            for ax, (metric, scale, label) in zip(axes[ri], metrics):
                for condition in CONDITIONS:
                    group = summary[(summary.population == population) &
                                    (summary.reversibility == mode) & (summary.condition == condition)].sort_values("t")
                    ax.errorbar(group.t, group[metric + "_mean"] * scale,
                                yerr=group[metric + "_sd"] * scale, color=colors[condition],
                                marker="o", capsize=3, label=CONDITION_LABELS[condition])
                    points = table[(table.population == population) &
                                   (table.reversibility == mode) & (table.condition == condition)]
                    ax.scatter(points.t + (points.seed - 3) * .025, points[metric] * scale,
                               color=colors[condition], alpha=.35, s=10)
                ax.set_title(mode.capitalize()); ax.set_ylabel(label)
                ax.set_xticks(sorted(summary.t.unique())); ax.grid(alpha=.2)
                if ri == 1:
                    ax.set_xlabel("Pre-commit observed transitions t")
        axes[0, 0].legend(fontsize=8)
        partner = "Familiar" if population == "train" else "Novel"
        fig.suptitle(f"{partner} partner performance across pre-commit observation time")
        fig.text(.5, .015, "Five learner seeds per point; bars: sample SD. t=0 reuses existing counterbalanced sliding irreversible/v3.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .96))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"performance_{population}.{ext}", dpi=180)
        plt.close(fig)


def aggregate(manifest_path: Path) -> None:
    manifest, records = load_manifest(manifest_path)
    out = Path(manifest["aggregate_root"])
    out.mkdir(parents=True, exist_ok=True)
    rows = [read_behavior_policy(record, population) for record in records for population in ("train", "test")]
    table = pd.DataFrame(rows)
    summary = summarize_behavior(table)
    table.drop(columns="per_round_success_rate").to_csv(out / "behavior_per_seed.csv", index=False)
    summary.to_csv(out / "behavior_summary.csv", index=False)
    round_rows = [{**{k: v for k, v in row.items() if k != "per_round_success_rate"},
                   "round_idx": i, "round_success": success}
                  for row in rows for i, success in enumerate(row["per_round_success_rate"])]
    pd.DataFrame(round_rows).to_csv(out / "behavior_by_round_per_seed.csv", index=False)
    plot_behavior(table, summary, out)
    write_json(out / "behavior_completion.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "manifest": str(manifest_path.resolve()),
        "n_policies": len(records), "new_policies": sum(not r["baseline"] for r in records),
        "reused_baseline_policies": sum(r["baseline"] for r in records),
        "learner_seeds": list(SEEDS), "variants": list(manifest_variants(manifest)), "conditions": list(CONDITIONS),
        "n_population_rollouts": len(rows), "caveats": CAVEATS,
    })
    (out / "README.md").write_text(
        "# Sliding pre-commit observation experiment\n\n"
        f"Observation times {manifest.get('new_times', [1, 2, 3, 4])} × irreversible/reversible, four model "
        f"conditions and five learner seeds give {sum(not r['baseline'] for r in records)} new policies. "
        "Existing t=0 sliding irreversible/v3 contributes 40 policies from seeds 1..5.\n\n"
        "`behavior_per_seed.csv` and `behavior_summary.csv` retain t, reversibility, model condition and seed. "
        "The familiar/novel figures plot all included times separately for each reversibility branch; error bars show "
        "sample SD across five trained seeds.\n\n" + "\n".join(f"- {c}" for c in CAVEATS) + "\n")
    print(f"Behavior aggregation and plots complete for {len(records)} policies: {out}", flush=True)


def _decorate(table: pd.DataFrame, record: dict, include_policy=True) -> pd.DataFrame:
    table = table.copy()
    for key, value in policy_metadata(record).items():
        if include_policy or key not in ("condition", "seed"):
            table[key] = value
    return table


def commitment_hidden(paths: dict[str, Path], data: rep.RolloutDataset, record: dict) -> np.ndarray:
    """Exact h_t after observing o_t, before this round's first allocation action.

    is_t0 is recorded from the observation passed to the actor. round_time is
    post-transition, so its commitment tick must equal t+1. No hidden state
    from after this round's allocation enters this supplementary decoder.
    """
    features = np.empty((20, len(data.episodes), rep.HIDDEN_DIM), dtype=np.float32)
    for population, path in paths.items():
        with h5py.File(path, "r") as handle:
            required = {"is_t0", "round_time", "round_idx", "hidden_state", "dones"}
            rep.require(required <= set(handle), f"{path}: missing commitment-timing arrays")
            for canonical_idx, row in data.episodes[data.episodes.capability_slice == population].iterrows():
                episode = int(row.source_episode)
                length = int(row.valid_length)
                flags = np.asarray(handle["is_t0"][episode, :length], dtype=bool)
                indices = np.flatnonzero(flags)
                rep.require(len(indices) == 20, f"{path}, episode{episode}: expected one commitment per round")
                rounds = np.asarray(handle["round_idx"][episode, indices])
                times = np.asarray(handle["round_time"][episode, indices])
                rep.require(np.array_equal(rounds, np.arange(20)), f"{path}: commitment round ordering is invalid")
                rep.require(np.all(times == record["t"] + 1), f"{path}: commitment occurred at the wrong observation time")
                features[:, canonical_idx] = handle["hidden_state"][episode, indices, :]
    rep.require(np.isfinite(features).all(), "Non-finite exact commitment hidden states")
    return features


def run_commitment_probes(features: np.ndarray, data: rep.RolloutDataset, split: dict,
                          record: dict, analysis_seed: int, device: str, out: Path) -> list[dict]:
    """Supplementary exact-state probes share the original split/optimizer settings."""
    train, test = rep.split_masks(data, split)
    specs = [(round_idx, target_idx, target) for round_idx in range(20)
             for target_idx, target in enumerate(rep.TARGETS)]
    x = np.asarray([features[round_idx] for round_idx, _, _ in specs])
    y = np.asarray([data.labels[:, target_idx] for _, target_idx, _ in specs])
    initializations = [analysis_seed + 10_000 + target_idx for _, target_idx, _ in specs]
    predictions, parameters = rep.train_linear_probe(x, y, train, initializations, device)
    save_numpy_archive(out / "probe_models" / f"commitment_{record['condition']}_seed{record['seed']}.npz", compressed=True,
                        **parameters, round_idx=np.array([r for r, _, _ in specs]),
                        target=np.array([target for _, _, target in specs]), initialization_seed=initializations,
                        t=record["t"], reversible=record["reversible"], condition=record["condition"], seed=record["seed"])
    rows = []
    for i, (round_idx, _, target) in enumerate(specs):
        for subset in ("all", "familiar", "novel"):
            mask = test.copy()
            if subset != "all":
                mask &= data.capability_slice == ("train" if subset == "familiar" else "test")
            rows.append({**policy_metadata(record), "training_seed": record["seed"],
                         "round_idx": round_idx, "target": target, "capability_subset": subset,
                         "feature": "exact_post_observation_hidden_before_initial_allocation",
                         "n_train": int(train.sum()), "probe_initialization_seed": initializations[i],
                         "final_train_loss": float(parameters["final_train_loss"][i]),
                         **rep.evaluate_probe(y[i], predictions[i], mask)})
    return rows


def fit_variant(records: list[dict], out: Path, analysis_seed: int, device: str) -> None:
    """Fit the original decoders independently for each policy, with one shared split."""
    metadata_path = out / "analysis_metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        rep.require(previous["analysis_seed"] == analysis_seed and previous["learner_seeds"] == list(SEEDS)
                    and previous["variant"] == records[0]["variant"], "Completed probe output has incompatible metadata")
        print(f"[reuse] completed decoding: {records[0]['variant']}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe_models").mkdir(exist_ok=True)
    (out / "features").mkdir(exist_ok=True)
    split = rep.make_probe_split(analysis_seed)
    write_json(out / "probe_split.json", split)
    times, rounds, commitments, shuffles, episodes, validations = [], [], [], [], [], []
    canonical = baseline = None
    for record in records:
        paths = rollout_paths(record)
        for path in paths.values():
            with h5py.File(path, "r") as handle:
                validate_rollout_metadata(handle, record, path)
        data = rep.load_checkpoint_rollouts(paths)
        train, test = rep.split_masks(data, split)
        fingerprint = np.column_stack((data.labels, data.rollout_rep, train.astype(int)))
        if canonical is None:
            canonical = fingerprint
            baseline = _decorate(rep.run_random_baseline(data, split, device, analysis_seed, out), record, False)
            baseline.to_csv(out / "random_baseline.csv", index=False)
        else:
            rep.require(np.array_equal(canonical, fingerprint), "Profiles/repetition/split differ between policies")
        exact_hidden = commitment_hidden(paths, data, record)
        metadata = policy_metadata(record)
        validations.append({**metadata, **data.validation,
                            "probe_train_episodes": int(train.sum()), "probe_test_episodes": int(test.sum())})
        episode_table = _decorate(data.episodes, record)
        episode_table["training_seed"] = record["seed"]
        episode_table["probe_partition"] = np.where(train, "train", "test")
        episodes.append(episode_table)
        save_numpy_archive(out / "features" / f"{record['condition']}_seed{record['seed']}.npz", compressed=True,
                            timestep_features=data.timestep_features, round_features=data.round_features,
                            final50_features=data.final50_features, commitment_hidden=exact_hidden, labels=data.labels,
                            rollout_rep=data.rollout_rep, valid_lengths=data.lengths, probe_train=train,
                            t=record["t"], reversible=record["reversible"], condition=record["condition"], seed=record["seed"])
        print(f"[probe] {record['variant']} {record['condition']} seed{record['seed']}: 58 primary affine decoders +2 shuffle controls", flush=True)
        tr, rr, sr = rep.run_probes(data, split, record["condition"], record["seed"], analysis_seed, device, out)
        commitments.extend(run_commitment_probes(exact_hidden, data, split, record, analysis_seed, device, out))
        times.extend(_decorate(pd.DataFrame(tr), record).to_dict("records"))
        rounds.extend(_decorate(pd.DataFrame(rr), record).to_dict("records"))
        shuffles.extend(_decorate(pd.DataFrame(sr), record).to_dict("records"))
        # Preserve completed expensive work if a later input fails.
        pd.DataFrame(times).to_csv(out / "probe_timestep_per_seed.csv", index=False)
        pd.DataFrame(rounds).to_csv(out / "probe_by_round_per_seed.csv", index=False)
        pd.DataFrame(commitments).to_csv(out / "probe_commitment_per_seed.csv", index=False)
        pd.DataFrame(shuffles).to_csv(out / "shuffled_label_baseline.csv", index=False)
        write_json(out / "validation.json", {"checkpoints": validations})
    timestep, by_round, exact_commitment, shuffle = pd.DataFrame(times), pd.DataFrame(rounds), pd.DataFrame(commitments), pd.DataFrame(shuffles)
    expected_policies = len(rep.CONDITIONS) * len(SEEDS)
    expected_timestep_rows = expected_policies * len(rep.REFERENCE_TIMES) * len(rep.TARGETS) * 3
    expected_round_rows = expected_policies * rep.N_ROUNDS * len(rep.TARGETS) * 3
    rep.require(len(records) == expected_policies and len(timestep) == expected_timestep_rows
                and len(by_round) == len(exact_commitment) == expected_round_rows,
                "Variant probe outputs must include all three RNN conditions × five seeds")
    pd.concat(episodes, ignore_index=True).to_csv(out / "episode_manifest.csv", index=False)
    for table, axis, filename in ((timestep, "reference_t", "probe_timestep_summary.csv"),
                                  (by_round, "round_idx", "probe_by_round_summary.csv"),
                                  (exact_commitment, "round_idx", "probe_commitment_summary.csv")):
        summary = rep.summarize_probes(table, axis, analysis_seed + 30_000, SEEDS)
        _decorate(summary, records[0], False).to_csv(out / filename, index=False)
    contrasts = pd.concat([rep.summarize_condition_differences(timestep, "reference_t", analysis_seed + 30_000, SEEDS),
                           rep.summarize_condition_differences(by_round, "round_idx", analysis_seed + 30_000, SEEDS)], ignore_index=True)
    _decorate(contrasts, records[0], False).to_csv(out / "probe_condition_differences.csv", index=False)
    random_means = baseline.groupby(["target", "capability_subset"]).distance_accuracy.mean()
    shuffle["random_baseline_mean"] = [random_means.loc[(r.target, r.capability_subset)] for r in shuffle.itertuples()]
    shuffle["difference_from_random"] = shuffle.distance_accuracy - shuffle.random_baseline_mean
    shuffle["leakage_screen_pass"] = shuffle.difference_from_random.abs() <= .10
    shuffle.to_csv(out / "shuffled_label_baseline.csv", index=False)
    rep.require(bool(shuffle[shuffle.capability_subset == "all"].leakage_screen_pass.all()),
                "Primary shuffled-label leakage screen failed; inspect saved controls before interpreting decoding")
    write_json(metadata_path, {
        **{k: v for k, v in policy_metadata(records[0]).items() if k not in ("condition", "seed")},
        "completed_utc": datetime.now(timezone.utc).isoformat(), "analysis_seed": analysis_seed,
        "learner_seeds": list(SEEDS), "conditions": list(rep.CONDITIONS), "n_policies": len(records),
        "n_episodes_per_policy": 920, "hidden_state_timing": "post_observation_before_action",
        "features": "primary: valid terminal-inclusive episode-prefix means; supplementary: exact hidden before each initial allocation",
        "probe_split": split, "probe": {"type": "independent affine 128->10 red/blue softmax", "steps": rep.PROBE_STEPS,
        "optimizer": "full-batch Adam", "learning_rate": rep.PROBE_LR, "normal_controls": 5,
        "preprocessing": "none", "model_selection": "none", "primary_shuffle_screen_pass": True},
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "h5py", "torch")},
        "caveats": CAVEATS,
    })


def plot_decoding(summary: pd.DataFrame, out: Path, figure_tag="decoding_across_t",
                  title="Partner capability decoding after round20") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(rep.CONDITIONS, ("#386cb0", "#f28e2b", "#2b9b7c")))
    for subset in ("all", "familiar", "novel"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
        for ri, mode in enumerate(("irreversible", "reversible")):
            for ci, target in enumerate(rep.TARGETS):
                ax = axes[ri, ci]
                for condition in rep.CONDITIONS:
                    group = summary[(summary.reversibility == mode) & (summary.target == target) &
                                    (summary.condition == condition) & (summary.capability_subset == subset)].sort_values("t")
                    ax.errorbar(group.t, group.distance_accuracy_mean,
                                yerr=np.vstack((group.distance_accuracy_mean - group.distance_accuracy_ci_low,
                                                group.distance_accuracy_ci_high - group.distance_accuracy_mean)),
                                marker="o", capsize=3, color=colors[condition], label=CONDITION_LABELS[condition])
                ax.set_title(f"{mode.capitalize()}: {target}"); ax.set_xticks(sorted(summary.t.unique())); ax.grid(alpha=.2)
                if ci == 0:
                    ax.set_ylabel("Distance-aware decoding accuracy")
                if ri == 1:
                    ax.set_xlabel("Pre-commit observed transitions t")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{title} ({subset} profiles)")
        fig.text(.5, .015, "Five learner seeds; bars: 95% bootstrap intervals over policies. MLP has no GRU state.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .96))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{figure_tag}_{subset}.{ext}", dpi=180)
        plt.close(fig)


def probes(manifest_path: Path, analysis_seed: int, device: str, threads: int, variant=None) -> None:
    manifest, records = load_manifest(manifest_path)
    out = Path(manifest.get("probe_root", str(Path(manifest["aggregate_root"]) / "probes")))
    rep.require((Path(manifest["aggregate_root"]) / "behavior_completion.json").is_file(),
                "Representation stage must follow successful behavior evaluation, aggregation and plotting")
    import torch
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        rep.require(torch.cuda.is_available(), "CUDA requested but unavailable")
    out.mkdir(parents=True, exist_ok=True)
    selected = [r for r in records if r["condition"] in rep.CONDITIONS]
    variants = manifest_variants(manifest)
    rep.require(variant is None or variant in variants, "Probe variant must be present in the manifest")
    for target_variant in ((variant,) if variant else variants):
        fit_variant([r for r in selected if r["variant"] == target_variant], out / target_variant, analysis_seed, device)
    if variant is None:
        probe_aggregate(manifest_path, analysis_seed)


def probe_aggregate(manifest_path: Path, analysis_seed: int = 0) -> None:
    manifest, records = load_manifest(manifest_path)
    out = Path(manifest.get("probe_root", str(Path(manifest["aggregate_root"]) / "probes")))
    rep.require((Path(manifest["aggregate_root"]) / "behavior_completion.json").is_file(),
                "Probe aggregation requires completed behavior plots")
    split = rep.make_probe_split(analysis_seed)
    variants = manifest_variants(manifest)
    selected = [r for r in records if r['condition'] in rep.CONDITIONS]
    n_policies = len(selected)
    for variant in variants:
        metadata = json.loads((out / variant / "analysis_metadata.json").read_text())
        rep.require(metadata["variant"] == variant and metadata["learner_seeds"] == list(SEEDS)
                    and metadata["n_policies"] == 15, f"{variant}: incomplete policy coverage")
        rep.require(json.loads((out / variant / "probe_split.json").read_text()) == split,
                    "All variants must use the same fixed episode/profile split")
    write_json(out / "probe_split.json", split)
    for filename in ("probe_timestep_per_seed.csv", "probe_by_round_per_seed.csv", "probe_timestep_summary.csv",
                     "probe_by_round_summary.csv", "probe_commitment_per_seed.csv", "probe_commitment_summary.csv", "random_baseline.csv", "shuffled_label_baseline.csv", "probe_condition_differences.csv"):
        pd.concat([pd.read_csv(out / v / filename) for v in variants], ignore_index=True).to_csv(out / filename, index=False)
    round_summary = pd.read_csv(out / "probe_by_round_summary.csv")
    plot_decoding(round_summary[round_summary.round_idx == 19], out)
    commitment_summary = pd.read_csv(out / "probe_commitment_summary.csv")
    for round_idx in (0, 19):
        plot_decoding(commitment_summary[commitment_summary.round_idx == round_idx], out,
                      f"commitment_decoding_round{round_idx + 1}_across_t",
                      f"Exact hidden decoding before initial allocation in round{round_idx + 1}")
    write_json(out / "completion_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "learner_seeds": list(SEEDS), "variants": list(variants), "n_policies": n_policies,
        "n_new_policies": sum(not r['baseline'] for r in selected),
        "n_baseline_policies": sum(r['baseline'] for r in selected), "n_input_rollouts": 2 * n_policies,
        "n_primary_affine_decoders": n_policies * (len(rep.REFERENCE_TIMES) + rep.N_ROUNDS) * len(rep.TARGETS),
        "n_exact_commitment_affine_decoders": n_policies * rep.N_ROUNDS * len(rep.TARGETS),
        "n_shuffled_label_decoders": n_policies * len(rep.TARGETS),
        "n_normal_feature_decoders": len(variants) * 5 * len(rep.TARGETS),
        "same_probe_split_all_variants": True, "mlp_excluded": "No recurrent hidden state; included in behavior only",
        "method_source_sha256": hashlib.sha256(Path(rep.__file__).read_bytes()).hexdigest(),
        "caveats": CAVEATS,
    })
    (out / "README.md").write_text(
        "# Sliding observation representation decoding\n\n"
        f"{n_policies} frozen recurrent policies ({len(variants)} variants × three RNN conditions × five seeds), with 920 episodes "
        "each. MLP policies appear in behavior comparisons and have no GRU state for this analysis.\n\n"
        "The original method is reused: post-observation hidden states before actions; terminal-inclusive "
        "valid-prefix means at episode steps 1,50,..,400 and completed rounds 1..20; one seeded 16/4 episode "
        "split within each of 46 capability profiles; independent red/blue affine 128→10 softmax decoders "
        "trained for 1000 full-batch Adam updates at learning rate 0.01. Five independent Normal-feature "
        "controls and paired shuffled-label screens use the same split and optimizer. No probe selection, "
        "standardization, or early stopping. Bootstrap 95% intervals resample the five policy seeds.\n\n"
        "Supplementary exact-state decoders use the hidden state after the commitment observation and "
        "before that round’s first allocation action, for each round independently. Saved is_t0 and "
        "post-transition round_time=t+1 verify alignment. Primary prefix-mean probes remain unchanged.\n\n"
        "`features/` within each variant saves averaged and exact commitment hidden arrays and split metadata; the source HDF5 "
        "files retain complete hidden-state scans. Saved probe weights, per-seed scores, summaries, controls "
        "and episode manifests identify observation time, reversibility, model condition and seed.\n\n"
        "Probe tests hold out episodes within profiles; familiar/novel labels describe policy training exposure, "
        "not probe profile generalization. Decoding demonstrates recoverability and does not establish causal use.\n\n"
        + "\n".join(f"- {c}" for c in CAVEATS) + "\n")
    print(f"Representation decoding completed for {n_policies} policies: {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("aggregate", "probes", "probe-aggregate"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--variant", help="Run one manifest variant for a Slurm probe stage")
    parser.add_argument("--analysis-seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    rep.require(args.analysis_seed >= 0 and args.threads > 0, "Analysis seed must be nonnegative and threads positive")
    if args.mode == "aggregate":
        aggregate(args.manifest)
    elif args.mode == "probes":
        probes(args.manifest, args.analysis_seed, args.device, args.threads, args.variant)
    else:
        rep.require(args.variant is None, "probe-aggregate merges all variants")
        probe_aggregate(args.manifest, args.analysis_seed)


if __name__ == "__main__":
    main()
