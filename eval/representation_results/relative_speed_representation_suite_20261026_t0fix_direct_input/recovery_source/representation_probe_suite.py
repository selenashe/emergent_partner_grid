"""Matched faster-goal probes for eight frozen-policy representations.

Recover missing inputs with the original evaluation seeds and batch sizes,
validate them against archived readouts/rollouts, and retain separate result
directories for every representation. No policy parameters are optimized.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd

import full_capability_evaluation as full
from representation_probe_features import checkpoint_heads, frozen_heads, masked_probabilities, representation_features

REPO = Path(__file__).resolve().parents[1]
PYTHON = "/nlp/scr/jshe/miniconda3/envs/emergent_partner_model/bin/python"
DEFAULT_BASE = REPO / "eval/representation_results/relative_speed_train_profiles_to_all90_v1_matched_v2_v3_20261022_t0fix_direct_input"
DEFAULT_OUT = REPO / "eval/representation_results/relative_speed_representation_suite_20261026_t0fix_direct_input"
REPRESENTATIONS = {
    "current_observation": (249, "Current observation x_t"),
    "previous_memory": (128, "Previous memory h_(t-1)"),
    "updated_memory": (128, "Updated memory h_t"),
    "actor_features": (128, "Frozen actor features"),
    "critic_features": (128, "Frozen critic features"),
    "action_logits": (15, "Unmasked action logits"),
    "action_probabilities": (15, "Masked action probabilities"),
    "value_estimate": (1, "Frozen value estimate"),
}
STEPS = np.arange(401)
FIELDS = ("rewards", "dones", "success", "round_done", "round_idx", "layout_idx",
          "partner_assignment", "ego_alloc_action", "partner_goal", "round_time", "is_t0")


def policy_name(record):
    return f"{record['condition']}_seed{record['seed']}"


def cache_directory(suite, record):
    return Path(suite["root"]) / "inputs" / record["version"] / policy_name(record)


def baseline_archive(suite, record):
    return Path(suite["baseline_root"]) / "policies" / record["version"] / policy_name(record) / "training_capability_probes.npz"


def prepare(base, out):
    base, out = Path(base).resolve(), Path(out).resolve()
    full.require(not out.exists(), "Use a fresh suite directory")
    original = full.read(base / "analysis_run.json")
    inventory = pd.read_csv(base / "capability_inventory.csv")
    review = full.read(REPO / "eval/evaluation_plans/t0fix_direct_input_capability_review_20261022/capability_review.json")
    out.mkdir(parents=True)
    for name in ("inputs", "jobs", "logs", "source", "mpl_cache"):
        (out / name).mkdir()
    suite = dict(status="prepared", root=str(out), baseline_root=str(base),
                 baseline_manifest_sha256=full.sha(base / "analysis_run.json"),
                 prepared_utc=full.now(), policies=original["policies"],
                 profiles=inventory.to_dict("records"), fit_profiles=original["probe_fit_profiles"],
                 original_test_profiles=review["existing_held_out_profiles"],
                 conditions=original["conditions"], versions=list(full.VERSIONS), seeds=[1,2,3,4,5],
                 representations=list(REPRESENTATIONS), evaluation_seed=12345, jobs={},
                 policy_retraining=False, recovery="Same original batch sizes, profiles, reset seeds, action keys and frozen checkpoints; strict archived-output checks.")
    for record in suite["policies"]:
        for key in ("config", "checkpoint"):
            full.require(full.sha(record[key]) == record[key + "_sha256"], "Changed policy " + key)
        full.require(Path(record["layouts_dir"]).exists(), "Missing frozen layout corpus")
        cache_directory(suite, record).mkdir(parents=True)
    inventory.to_csv(out / "capability_inventory.csv", index=False)
    full.write(out / "input_manifest.json", dict(profiles=suite["profiles"], policies=suite["policies"],
                                              original_manifest_unavailable=True, baseline_manifest=str(base / "analysis_run.json")))
    full.write(out / "suite_run.json", suite)
    print(out / "suite_run.json", flush=True)
    return out / "suite_run.json"


def recover_rollout(evaluator, record, pool, steps=401):
    """Replay a batch without changing its RNG shape; capture pre-action inputs."""
    jax, jnp = evaluator.jax, evaluator.jnp
    config = full.read(record["config"])
    kwargs = dict(config["ENV_KWARGS"], layouts_dir=record["layouts_dir"],
                  augment_symmetries=False, partner_capability_pairs=np.asarray(pool).tolist(),
                  influence=bool(config["INFLUENCE"]))
    kwargs.pop("layout_path", None); kwargs.pop("layout_paths", None)
    env = evaluator.CoordinationGrid(**kwargs)
    network, initialize, _ = evaluator.trainer_mod.build_network(config, env.n_ego_actions)
    params = evaluator.load_params(record["checkpoint"])
    batch = len(pool) * 20
    indices = jnp.repeat(jnp.arange(len(pool), dtype=jnp.int32), 20)
    reset_key, action_key = jax.random.split(jax.random.PRNGKey(12345))
    obs, state = jax.vmap(env.reset)(jax.random.split(reset_key, batch))
    state = state.replace(capability=env.partner_capability_pairs[indices])
    obs = jax.vmap(env.get_obs)(state)

    def body(carry, unused):
        obs, state, hidden, previous_done, key = carry
        current = obs["agent_0"]
        network_input = jax.tree_util.tree_map(lambda v: v[jnp.newaxis, :], current)
        new_hidden, pi, value = network.apply(params, hidden, (network_input, previous_done[jnp.newaxis, :]))
        key, ka, ks = jax.random.split(key, 3)
        action = pi.sample(seed=ka).squeeze(0)
        next_obs, next_state, reward, done, info = jax.vmap(env.step_env, in_axes=(0,0,{"agent_0":0}))(
            jax.random.split(ks, batch), state, {"agent_0": action})
        vector = jnp.concatenate((current["grid"].reshape(batch, -1), current["last_allocation"],
                                  current["is_t0"][:, None]), axis=-1)
        saved = dict(hidden_state=new_hidden, observation=vector, rewards=reward["agent_0"],
                     dones=done["__all__"], success=info["success"], round_done=info["round_done"],
                     round_idx=info["round_idx"], layout_idx=info["layout_idx"],
                     partner_assignment=info["partner_assignment"], ego_alloc_action=info["ego_alloc_action"],
                     partner_goal=info["partner_goal"], round_time=info["round_time"],
                     is_t0=current["is_t0"] > .5, policy_probabilities=pi.probs.squeeze(0),
                     policy_value=value.squeeze(0), legal_actions=jnp.isfinite(pi.logits.squeeze(0)))
        return (next_obs, next_state, new_hidden, done["__all__"], key), saved

    (_, _, _, _, _), saved = jax.lax.scan(body, (obs, state, initialize(batch), jnp.zeros(batch, bool), action_key), None, length=steps)
    result = {key: np.asarray(value).swapaxes(0,1) for key, value in saved.items()}
    result.update(pairs=np.repeat(np.asarray(pool), 20, axis=0), repetitions=np.tile(np.arange(20), len(pool)))
    return result


def validate_recovery(suite, record, label, result):
    with np.load(baseline_archive(suite, record)) as old:
        indices = np.array([np.flatnonzero((old["pairs"] == pair).all(axis=1) & (old["repetitions"] == rep))[0]
                            for pair, rep in zip(result["pairs"], result["repetitions"])])
        lengths = old["lengths"][indices]
        time = np.minimum(STEPS[None, :], lengths[:, None] - 1)
        hidden = result["hidden_state"][np.arange(len(indices))[:, None], time]
        probabilities = old["probabilities"][:, indices]
        weights, bias = old["weight"], old["bias"]
        logits = np.einsum("eth,thc->tec", hidden, weights, optimize=True) + bias[:, None, :]
        logits -= logits.max(axis=-1, keepdims=True)
        replay_probability = np.exp(logits); replay_probability /= replay_probability.sum(axis=-1, keepdims=True)
        errors = np.abs(replay_probability - probabilities)
        disagreements = replay_probability.argmax(axis=-1) != old["predictions"][:, indices]
        mask = old["train"][indices] | old["test"][indices]
        audit = dict(slice=label, episodes=len(indices), probability_max_error=float(errors.max()),
                     probability_q999_error=float(np.quantile(errors,.999)),
                     prediction_disagreements=int(disagreements.sum()),
                     used_prediction_disagreements=int(disagreements[:, mask].sum()),
                     test_prediction_disagreements=int(disagreements[:, old["test"][indices]].sum()))
        expected_terminal = STEPS[None, :] == lengths[:, None] - 1
        active = STEPS[None, :] < lengths[:, None]
        audit["terminal_mismatches"] = int(((result["dones"] != expected_terminal) & active).sum())
        if label in record["original_rollouts"]:
            with h5py.File(record["original_rollouts"][label], "r") as source:
                np.testing.assert_array_equal(source["capability_pool"][:], result["pairs"][::20])
                mismatches = {}
                for field in FIELDS:
                    reference = source[field][:, :401]
                    equal = np.isclose(result[field], reference, atol=1e-6) if reference.dtype.kind == "f" else result[field] == reference
                    mismatches[field] = int((~equal & active).sum())
                saved_hidden = source["hidden_state"][:, :401]
                audit["original_field_mismatches"] = mismatches
                audit["original_hidden_max_error"] = float(np.max(np.abs(saved_hidden-result["hidden_state"])[active]))
                # Preserve the exact recorded representation wherever it survives.
                result["hidden_state"] = saved_hidden
        result["lengths"] = lengths
    audit["status"] = "diagnostic"
    return audit


def recover(path, index, only=None, diagnostic=False):
    suite = full.read(path); record = suite["policies"][index]
    evaluator = full.load_evaluator(record)
    full.require(evaluator.jax.default_backend() == "gpu", "Recovery must use the original GPU backend")
    pools = dict(train=suite["fit_profiles"], test=suite["original_test_profiles"],
                 additional=[(p["red_delay"],p["blue_delay"]) for p in suite["profiles"] if p["requires_new_profile_rollout"]])
    directory = cache_directory(suite, record)
    for label, pool in pools.items():
        if only is not None and label != only: continue
        target = directory / (label + ".h5")
        if target.exists() and target.with_suffix(".json").exists() and not diagnostic:
            full.require(full.read(target.with_suffix(".json"))["status"] == "passed", "Unvalidated cached input")
            continue
        print("[recover]", full.identity(record), label, len(pool)*20, flush=True)
        result = recover_rollout(evaluator, record, pool)
        audit = validate_recovery(suite, record, label, result)
        print("[recovery audit]", json.dumps(audit), flush=True)
        full.write(directory / (label + "_diagnostic.json"), audit)
        if diagnostic: continue
        full.require(audit["terminal_mismatches"] == 0, "Recovered episode lengths differ")
        full.require(audit["used_prediction_disagreements"] == 0, "Recovered trajectory changes an archived decoding decision")
        full.require(audit["probability_max_error"] <= .002, "Recovered states differ from archived decoder outputs")
        full.require(not any(audit.get("original_field_mismatches", {}).values()), "Replay differs from surviving behavior")
        temporary = target.with_suffix(".partial.h5")
        with h5py.File(temporary, "w") as handle:
            for key, values in result.items():
                values = np.asarray(values)
                if key == "observation":
                    np.testing.assert_array_equal(values, values.astype(np.uint8))
                    values = values.astype(np.uint8)
                handle.create_dataset(key, data=values, compression="gzip", compression_opts=1, shuffle=True)
            handle.attrs.update(evaluation_seed=12345, checkpoint_sha256=record["checkpoint_sha256"],
                                original_batch_size=len(pool)*20, reconstructed=True)
        temporary.replace(target)
        audit.update(status="passed", archive_sha256=full.sha(target), archive_stats=full.stats(target),
                     producer_sha256=full.sha(__file__), completed_utc=full.now())
        full.write(target.with_suffix(".json"), audit)
    print("[recovery finished]", full.identity(record), flush=True)


SOURCE_FILES = ("representation_probe_suite.py", "representation_probe_features.py", "relative_speed_representation.py",
                "representation_analysis.py", "full_capability_evaluation.py", "analyze_instantaneous_decoding.py",
                "plot_instantaneous_decoding.py", "analyze_decoding_behavior.py", "analyze_early_decoding.py",
                "plot_early_decoding.py", "analyze_replayed_motion.py", "analyze_decoding_behavior_correlations.py")


def initialize_runs(path):
    suite = full.read(path); root = Path(suite["root"]); base = Path(suite["baseline_root"])
    repo = Path(suite.get("repo_root", REPO))
    suite["repo_root"] = str(repo)
    original = full.read(base / "analysis_run.json")
    for destination in [root / "source"] + [root / name / "source" for name in REPRESENTATIONS]:
        destination.mkdir(parents=True, exist_ok=True)
        for name in SOURCE_FILES:
            shutil.copyfile(repo / "eval" / name, destination / name)
        shutil.copyfile(repo / "repo_paths.py", destination / "repo_paths.py")
    for name, (dimension, label) in REPRESENTATIONS.items():
        directory = root / name
        for sub in ("policies", "jobs", "logs", "mpl_cache"):
            (directory / sub).mkdir(exist_ok=True)
        # Titles identify the feature actually used, including combined figures.
        plotting = directory / "source/plot_instantaneous_decoding.py"
        code = plotting.read_text().replace("Single-state h_t decoding at every step", label + " decoding at every step")
        code = code.replace("Single-state faster-goal decoding", label + " faster-goal decoding")
        code = code.replace("h_t at every environment step", label + " at every environment step")
        code = code.replace("single-state decoding; every step", label + " decoding; every step")
        plotting.write_text(code)
        behavioral = directory / "source/analyze_decoding_behavior.py"
        code = behavioral.read_text().replace('"Single-state decoding"', repr(label + " decoding"))
        code = code.replace("Training-capability probes · early decoding and evaluation behavior", label + " · early decoding and evaluation behavior")
        behavioral.write_text(code)
        records = []
        for record in suite["policies"]:
            output = directory / "policies" / record["version"] / policy_name(record)
            output.mkdir(parents=True, exist_ok=True)
            records.append(dict(record, probe_directory=str(output), baseline_probe_directory=record["probe_directory"],
                                feature_directory=str(cache_directory(suite, record))))
        run = dict(original, status="prepared", root=str(directory), input_manifest=str(root / "input_manifest.json"),
                   input_manifest_sha256=full.sha(root / "input_manifest.json"), suite_manifest=str(path.resolve()),
                   representation=name, representation_label=label, feature_dimension=dimension, policies=records,
                   baseline_root=str(base), prepared_utc=full.now(), jobs={},
                   method=f"Separate affine {dimension}->2 probes for {label}; 384 familiar-profile fit episodes; unchanged 360 held-out all90 episodes; Adam1000 lr=.01 init10000; raw float32 features.",
                   terminal="Hold each representation's final valid vector forward. Previous memory at terminal observation L-1 is h_(L-2).",
                   source_sha256={p.name: full.sha(p) for p in (directory / "source").glob("*.py")},
                   trajectory_reconstruction=True, policy_retraining=False, new_rollouts=False)
        full.write(directory / "analysis_run.json", run)
        shutil.copyfile(root / "capability_inventory.csv", directory / "capability_inventory.csv")
        (directory / "PLAN.md").write_text(description(run))
    suite["source_sha256"] = {p.name: full.sha(p) for p in (root / "source").glob("*.py")}
    suite["status"] = "initialized"
    full.write(path, suite)
    print("[initialized eight representation directories]", root, flush=True)


def description(run):
    return (f"# {run['representation_label']} faster-goal decoding\n\n"
            f"Feature dimension: {run['feature_dimension']}; one vector per episode at each observation index t=0..400. "
            "The frozen policy and all actor/critic transformations remain unchanged. Only the affine two-class readout is fitted.\n\n"
            "Fit 24 actual policy-training profiles x16 held-in evaluation episodes =384 examples, 192 per class. "
            "Test all 90 unequal profiles x4 held-out episodes =360 examples; repetition IDs 1,9,14,15. "
            "The other 1056 examples never enter fitting or reported testing. No temporal averaging or feature normalization. "
            "Independent full-batch class-balanced Adam1000, lr .01, no weight decay, initialization 10000 with scale 1/sqrt(D).\n\n"
            "Five policy seeds for each of v1_matched, v2 and v3; influence and no-influence conditions reported separately. "
            "Single-partner control remains omitted because its policy-training support has one faster-goal class. "
            "Profile accuracy uses four episodes; category curves give profiles equal weight within seed. "
            "Intervals bootstrap five policy seeds, 10000 resamples, RNG30000.\n\n"
            "Trajectories reconstructed from original batch sizes, profile order, reset/action RNG seeds and frozen checkpoints. "
            "Reconstruction audits compare surviving original behavior, archived decoding outputs, and episode terminals. "
            "Inputs are shared with the other representations and archived separately. No replacement episode IDs.\n\n"
            f"Temporal boundary rule: {run['terminal']}\n\n"
            "Outputs mirror the baseline per-profile/category results and include representation-specific early-behavior diagnostics.\n")


def verify_sources(run):
    root = Path(run["root"])
    for name, digest in run["source_sha256"].items():
        full.require(full.sha(root / "source" / name) == digest, "Changed frozen source " + name)


def load_cache(record):
    parts = []
    for name in ("train", "test", "additional"):
        path = Path(record["feature_directory"]) / (name + ".h5")
        audit = full.read(path.with_suffix(".json"))
        full.require(audit["status"] == "passed", "Unvalidated feature input")
        if "archive_stats" in audit:
            full.require(full.stats(path) == audit["archive_stats"], "Changed feature cache")
        with h5py.File(path) as handle:
            parts.append({key: handle[key][:] for key in handle})
    result = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    order = np.lexsort((result["repetitions"], result["pairs"][:,1], result["pairs"][:,0]))
    return {key: values[order] for key, values in result.items()}


def feature_chunk(run, record, data, steps, weights):
    features = representation_features(run["representation"], data["hidden_state"], data["observation"],
                                       data["legal_actions"], data["lengths"], steps, weights)
    full.require(features.shape == (len(steps),1800,run["feature_dimension"]), "Wrong feature dimensions")
    full.require(np.isfinite(features).all(), "Nonfinite probe input")
    return features


def worker(path, index):
    import torch
    from relative_speed_representation import train_probes
    from analyze_instantaneous_decoding import profile_metrics
    torch.set_num_threads(1)
    run = full.read(path); verify_sources(run); record = run["policies"][index]
    output = Path(record["probe_directory"])
    if (output / "completion.json").exists():
        full.require(full.read(output / "completion.json")["status"] == "completed", "Incomplete prior worker")
        return
    for key in ("config", "checkpoint"):
        full.require(full.sha(record[key]) == record[key + "_sha256"], "Changed policy " + key)
    data = load_cache(record)
    baseline = Path(record["baseline_probe_directory"]) / "training_capability_probes.npz"
    with np.load(baseline) as old:
        for key in ("pairs", "repetitions", "lengths"):
            np.testing.assert_array_equal(data[key], old[key])
        metadata = {key: old[key].copy() for key in ("pairs","repetitions","lengths","truth","train","test","unused",
                    "probe_fit_profile","probe_fit_profiles","policy_training_profiles")}
    full.require(tuple(int(metadata[k].sum()) for k in ("train","test","unused")) == (384,360,1056), "Changed fit/test masks")
    np.testing.assert_array_equal(np.unique(metadata["repetitions"][metadata["test"]]), [1,9,14,15])
    heads = checkpoint_heads(record["checkpoint"])
    # Validate deterministic head extraction against the actual frozen network.
    chosen = np.arange(0,1800,20)
    actor, critic, logits, value = frozen_heads(data["hidden_state"][chosen,:31], heads)
    probs = masked_probabilities(logits, data["legal_actions"][chosen,:31])
    np.testing.assert_allclose(value[...,0], data["policy_value"][chosen,:31], atol=2e-3, rtol=1e-4)
    np.testing.assert_allclose(probs, data["policy_probabilities"][chosen,:31], atol=2e-3, rtol=1e-4)
    for filename in ("episode_manifest.csv", "probe_split.json"):
        shutil.copyfile(Path(record["baseline_probe_directory"]) / filename, output / filename)
    chunks = []; model_keys = ("weight","bias","class_weights","final_train_loss","predictions","probabilities")
    for start in range(0,401,64):
        steps = STEPS[start:start+64]; target = output / f"chunk_{start:03d}.npz"
        if not target.exists():
            print("[fit]", run["representation"], full.identity(record), int(steps[0]), int(steps[-1]), flush=True)
            features = feature_chunk(run, record, data, steps, heads)
            prediction, probability, models = train_probes(features, np.broadcast_to(metadata["truth"],(len(steps),1800)).copy(),
                                                         metadata["train"], [10000]*len(steps))
            np.savez_compressed(target, **models, predictions=prediction, probabilities=probability,
                                env_steps=steps, train=metadata["train"], test=metadata["test"])
            del features
        with np.load(target) as saved:
            np.testing.assert_array_equal(saved["env_steps"], steps)
            for key in ("train","test"): np.testing.assert_array_equal(saved[key],metadata[key])
            chunks.append({key:saved[key].copy() for key in model_keys})
    result = {key:np.concatenate([part[key] for part in chunks]) for key in model_keys}
    full.require(result["probabilities"].shape == (401,1800,2), "Missing fitted readouts")
    np.testing.assert_allclose(result["probabilities"].sum(axis=-1),1,atol=1e-6)
    metrics = profile_metrics(result["predictions"], result["probabilities"], data["pairs"],metadata["truth"],metadata["test"],data["lengths"])
    # Fixed step-20 decoder is evaluated on the same representation at every t.
    fixed_probability = []
    for start in range(0,401,64):
        features = feature_chunk(run, record, data, STEPS[start:start+64], heads)
        z = features @ result["weight"][20] + result["bias"][20]
        z -= z.max(axis=-1,keepdims=True); p = np.exp(z); p /= p.sum(axis=-1,keepdims=True)
        fixed_probability.append(p)
    fixed_probability = np.concatenate(fixed_probability)
    archive = output / "training_capability_probes.npz"
    np.savez_compressed(archive, **result, **metrics, **metadata, env_steps=STEPS,
                        fixed20_probabilities=fixed_probability, representation=run["representation"], feature_dimension=run["feature_dimension"])
    done = dict(status="completed", completed_utc=full.now(), **full.identity(record), representation=run["representation"],
                feature_dimension=run["feature_dimension"], archive_sha256=full.sha(archive), timepoints=401, profiles=90,
                fit_profiles=24, fit_episodes=384, test_episodes=360, unused_episodes=1056, novel_profiles_in_probe_fit=0,
                calibration_scope=record["calibration_scope"], policy_retraining=False, new_rollouts=False,
                trajectory_reconstruction=True)
    if run["representation"] == "updated_memory":
        with np.load(baseline) as old:
            error = np.abs(result["probabilities"]-old["probabilities"])
            done["baseline_probability_max_error"] = float(error.max())
            done["baseline_test_decision_disagreements"] = int((result["predictions"][:,metadata["test"]] != old["predictions"][:,metadata["test"]]).sum())
    full.write(output / "completion.json", done)
    print("[worker completed]", run["representation"], full.identity(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "recover"))
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_OUT / "suite_run.json")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--only", choices=("train", "test", "additional"))
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare": prepare(args.base, args.out)
    else: recover(args.path, args.index, args.only, args.diagnostic)


if __name__ == "__main__":
    main()
