"""Fit independent faster-goal probes to h_t at every saved observation, t=0..400.

Uses saved rollouts only. There is no averaging of hidden states across time.
The initial observation is t=0; after termination the final valid state is held.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import h5py
import numpy as np

import full_capability_evaluation as full

STEPS = np.arange(401)
CHUNK = 64


def instant_features(hidden, lengths, steps=STEPS):
    """Return [time, episode, hidden] using one state, never a time average."""
    full.require(hidden.ndim == 3 and len(hidden) == len(lengths), "Wrong state dimensions")
    full.require(np.all(lengths > 0), "Empty episode")
    indices = np.minimum(np.asarray(steps)[None, :], np.asarray(lengths)[:, None] - 1)
    full.require(indices.max() < hidden.shape[1], "Insufficient saved states")
    return hidden[np.arange(len(hidden))[:, None], indices].transpose(1, 0, 2).copy()


def load_states(record):
    batches = []
    paths = list(record["original_rollouts"].values()) + [str(Path(record["directory"]) / "additional_profiles.h5")]
    for path in paths:
        with h5py.File(path, "r") as handle:
            done = handle["dones"][:]
            full.require(done.any(axis=1).all(), "Missing terminal")
            pairs = handle["capability"][:, 0, :]
            np.testing.assert_array_equal(handle["capability_pool"][:][handle["capability_index_per_ep"][:]], pairs)
            hidden = handle["hidden_state"][:, :len(STEPS)]
            full.require(hidden.shape[2] == 128 and np.isfinite(hidden).all(), "Invalid saved states")
            counters = {}; repetitions = []
            for pair in map(tuple, pairs):
                repetitions.append(counters.get(pair, 0)); counters[pair] = repetitions[-1] + 1
            full.require(set(counters.values()) == {20}, "Wrong episode repetitions")
            batches.append(dict(hidden=hidden, lengths=done.argmax(axis=1) + 1, pairs=pairs,
                                repetitions=np.array(repetitions)))
    data = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
    del batches
    order = np.lexsort((data["repetitions"], data["pairs"][:, 1], data["pairs"][:, 0]))
    data = {key: value[order] for key, value in data.items()}
    full.require(len(data["pairs"]) == 1800 and len(np.unique(data["pairs"], axis=0)) == 90, "Wrong profiles")
    with np.load(Path(record["directory"]) / "probe_models.npz") as prior:
        for key in ("pairs", "repetitions"):
            np.testing.assert_array_equal(data[key], prior[key])
        data["train"] = prior["train"].copy(); data["test"] = prior["test"].copy()
    full.require((data["train"].sum(), data["test"].sum()) == (1440, 360), "Wrong probe split")
    return data


def profile_metrics(predictions, probabilities, pairs, truth, test, lengths):
    profiles = np.unique(pairs, axis=0)
    accuracy = []; probability = []; active = []
    for pair in profiles:
        chosen = np.flatnonzero(test & (pairs == pair).all(axis=1))
        full.require(len(chosen) == 4, "Expected four test episodes per profile")
        accuracy.append((predictions[:, chosen] == truth[chosen]).mean(axis=1))
        probability.append(probabilities[:, chosen, truth[chosen]].mean(axis=1))
        active.append((lengths[chosen][None, :] > STEPS[:, None]).sum(axis=1))
    return dict(profiles=profiles, accuracy=np.array(accuracy).T,
                correct_probability=np.array(probability).T, active_test_episodes=np.array(active).T)


def worker(path, index):
    import torch
    torch.set_num_threads(1)
    run = full.read(path); manifest = full.read(run["input_manifest"]); record = run["policies"][index]
    out = Path(record["instant_directory"]); out.mkdir(parents=True, exist_ok=True)
    if (out / "completion.json").exists():
        return
    full.verify(manifest, record)
    extra = full.read(Path(record["directory"]) / "evaluation_completion.json")
    full.require(full.stats(Path(record["directory"]) / "additional_profiles.h5") == extra["rollout_stats"], "Rollouts changed")
    print("[read saved states]", full.identity(record), flush=True)
    data = load_states(record)
    binary, _ = full.references(manifest)
    truth = binary.relative_classes(data["pairs"])
    chunks = []; model_keys = ("weight", "bias", "class_weights", "final_train_loss")
    for start in range(0, len(STEPS), CHUNK):
        steps = STEPS[start:start + CHUNK]
        target = out / f"chunk_{start:03d}.npz"
        if not target.exists():
            features = instant_features(data["hidden"], data["lengths"], steps)
            print("[fit independent h_t probes]", int(steps[0]), int(steps[-1]), flush=True)
            prediction, probability, models = binary.train_probes(features,
                np.broadcast_to(truth, (len(steps), len(truth))).copy(), data["train"], [10000] * len(steps))
            np.savez_compressed(target, **models, predictions=prediction, probabilities=probability, env_steps=steps)
            del features
        with np.load(target) as saved:
            np.testing.assert_array_equal(saved["env_steps"], steps)
            chunks.append({key: saved[key].copy() for key in model_keys + ("predictions", "probabilities")})
    output = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    full.require(output["probabilities"].shape == (401, 1800, 2), "Incomplete timestep probes")
    full.require(np.isfinite(output["probabilities"]).all(), "Nonfinite probe result")
    np.testing.assert_allclose(output["probabilities"].sum(axis=-1), 1, atol=1e-6)
    # Independently fit controls from the earlier analysis provide regression anchors.
    early = Path(run["early_root"]) / "policies" / record["version"] / f"{record['condition']}_seed{record['seed']}" / "early_probes.npz"
    with np.load(early) as prior:
        for key in ("pairs", "repetitions", "train", "test", "truth"):
            np.testing.assert_array_equal(data[key] if key != "truth" else truth, prior[key])
        anchors = prior["instant_steps"]
        previous = prior["probabilities"][len(prior["env_steps"]):]
        # Different CPU instruction sets can slightly perturb Adam's final
        # probabilities. Check both fitted weights and exact classifications;
        # the plotted accuracy must reproduce the prior instantaneous controls.
        np.testing.assert_allclose(output["weight"][anchors], prior["weight"][len(prior["env_steps"]):], atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(output["probabilities"][anchors], previous, atol=1e-3, rtol=1e-4)
        disagreement = output["predictions"][anchors] != previous.argmax(axis=-1)
        full.require(not disagreement.any(), "Anchor classification discrepancy")
        anchor_max_error = float(np.abs(output["probabilities"][anchors] - previous).max())
    metrics = profile_metrics(output["predictions"], output["probabilities"], data["pairs"], truth,
                              data["test"], data["lengths"])
    target = out / "instantaneous_probes.npz"
    np.savez_compressed(target, **output, **metrics, truth=truth, pairs=data["pairs"],
        repetitions=data["repetitions"], train=data["train"], test=data["test"], lengths=data["lengths"], env_steps=STEPS)
    full.write(out / "completion.json", dict(status="completed", completed_utc=full.now(), **full.identity(record),
        archive_sha256=full.sha(target), episodes=1800, profiles=90, instantaneous_timepoints=401,
        anchor_timepoints=anchors.tolist(), anchor_max_probability_error=anchor_max_error,
        anchor_prediction_disagreements=int(disagreement.sum()), policy_retraining=False, new_rollouts=False))
    print("[completed]", full.identity(record), flush=True)


def bootstrap_seeds(values):
    """Exact RNG-30000 percentile bootstrap, efficiently grouping repeated counts.

    The first axis is the five policy seeds. All remaining axes are independent
    statistics. Grouping identical bootstrap count vectors preserves all 10,000
    sampled means and NumPy's linear quantile interpolation.
    """
    values = np.asarray(values, dtype=np.float64)
    full.require(values.shape[0] == 5 and np.isfinite(values).all(), "Require five finite seeds")
    draws = np.random.default_rng(30000).integers(0, 5, size=(10000, 5))
    counts = np.stack([(draws == seed).sum(axis=1) for seed in range(5)], axis=1)
    unique, frequencies = np.unique(counts, axis=0, return_counts=True)
    flattened = values.reshape(5, -1)
    bounds = np.empty((2, flattened.shape[1]))
    for start in range(0, flattened.shape[1], 4096):
        means = (unique @ flattened[:, start:start + 4096]) / 5
        order = np.argsort(means, axis=0)
        sorted_means = np.take_along_axis(means, order, axis=0)
        cumulative = np.cumsum(frequencies[order], axis=0)
        columns = np.arange(means.shape[1])
        for row, q in enumerate((.025, .975)):
            h = (len(draws) - 1) * q; lower = int(np.floor(h)); upper = int(np.ceil(h))
            low = (cumulative <= lower).sum(axis=0); high = (cumulative <= upper).sum(axis=0)
            bounds[row, start:start + means.shape[1]] = (1 - (h - lower)) * sorted_means[low, columns] + (h - lower) * sorted_means[high, columns]
    return values.mean(axis=0), bounds[0].reshape(values.shape[1:]), bounds[1].reshape(values.shape[1:])


def prepare(root, out):
    root = Path(root).resolve(); out = Path(out).resolve()
    full.require(not out.exists(), "Use a fresh output directory")
    manifest = full.read(root / "analysis_run.json")
    full.require(manifest["status"] == "completed" and full.read(root / "completion_audit.json")["status"] == "passed", "Incomplete source analysis")
    full.require(full.read(root / "early_dynamics/completion_audit.json")["status"] == "passed", "Missing instantaneous controls")
    policies = [dict(record, instant_directory=str(out / "policies" / record["version"] / f"{record['condition']}_seed{record['seed']}"))
                for record in manifest["policies"] if record["condition"] in full.CONDITIONS]
    full.require(len(policies) == 45, "Expected 45 RNN policies")
    out.mkdir(parents=True)
    for name in ("analyze_instantaneous_decoding.py", "plot_instantaneous_decoding.py", "full_capability_evaluation.py"):
        target = out / "source" / name; target.parent.mkdir(exist_ok=True)
        shutil.copyfile(Path(__file__).with_name(name), target)
    (out / "logs").mkdir(); (out / "jobs").mkdir()
    run = dict(status="prepared", root=str(out), input_manifest=str(root / "analysis_run.json"),
        early_root=str(root / "early_dynamics"), policies=policies, env_steps=STEPS.tolist(), seeds=[1,2,3,4,5],
        analysis_python=full.ANALYSIS_PYTHON, prepared_utc=full.now(), jobs={},
        feature="Single post-observation, pre-action recurrent state h_t; no explicit temporal averaging. t=0 is the initial observation, before any transition. 401 steps t=0..400 inclusive.",
        probe="Independent affine 128->2 at every step; reference full-batch class-balanced Adam, 1000 updates, lr=.01, initialization 10000. All 90 profiles; unchanged repetition split, 16 fit / 4 test per profile.",
        aggregation="Mean five policy seeds; 95% percentile bootstrap, 10000 resamples, RNG30000. Category means first average profiles within each seed.",
        terminal="At steps beyond termination carry the final valid hidden state forward, retaining identical episodes and probe partitions; never read padding as a feature. Active test episode counts are saved.",
        policy_retraining=False, new_rollouts=False)
    run["source_sha256"] = {p.name: full.sha(p) for p in (out / "source").glob("*.py")}
    path = out / "analysis_run.json"; full.write(path, run)
    return path


def submit(path):
    path = Path(path).resolve(); run = full.read(path); out = Path(run["root"])
    for name, digest in run["source_sha256"].items():
        full.require(full.sha(out / "source" / name) == digest, "Frozen analysis changed")
    clean = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
    for mode in ("worker", "aggregate"):
        if mode in run["jobs"]: continue
        command = [run["analysis_python"], "-u", str(out / "source/analyze_instantaneous_decoding.py"), mode, str(path)]
        if mode == "worker": command += ["--index"]
        lines = ["#!/bin/bash", "set -euo pipefail", "export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1",
            "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1",
            "export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=", 'export XLA_FLAGS=""',
            "export MPLCONFIGDIR=" + shlex.quote(str(out / "mpl_cache")),
            "exec " + shlex.join(command) + (' "${SLURM_ARRAY_TASK_ID}"' if mode == "worker" else "")]
        script = out / "jobs" / (mode + ".sh"); script.write_text("\n".join(lines) + "\n")
        subprocess.run(["bash", "-n", str(script)], check=True)
        cmd = ["sbatch", "--parsable", "--account=nlp", "--partition=sphinx", "--exclude=sphinx9",
            "--cpus-per-task=1", "--mem=8G", "--time=01:00:00", "--job-name=instant_decode_" + mode,
            "--chdir=" + str(Path(run["input_manifest"]).parents[3]), "--kill-on-invalid-dep=yes",
            f"--output={out}/logs/{mode}_%A_%a.out", f"--error={out}/logs/{mode}_%A_%a.err"]
        if mode == "worker": cmd += ["--array=0-44%30"]
        else: cmd += ["--dependency=afterok:" + run["jobs"]["worker"]["job_id"]]
        job_id = subprocess.check_output(cmd + [str(script)], text=True, env=clean).strip().split(";")[0]
        full.require(job_id.isdigit(), "Invalid job ID")
        run["jobs"][mode] = dict(job_id=job_id, command=cmd + [str(script)])
        run["status"] = "submitted"; full.write(path, run); print(mode, job_id, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "submit", "worker", "aggregate"))
    parser.add_argument("path", type=Path); parser.add_argument("--out", type=Path); parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "prepare": print(prepare(args.path, args.out), flush=True)
    elif args.mode == "submit": submit(args.path)
    elif args.mode == "worker": worker(args.path, args.index)
    else:
        from plot_instantaneous_decoding import aggregate
        aggregate(args.path)


if __name__ == "__main__": main()
