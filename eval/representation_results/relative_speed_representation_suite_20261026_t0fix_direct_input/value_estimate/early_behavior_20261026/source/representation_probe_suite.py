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


def recovery_pools(suite, record):
    # RNG slots depend on the source's stored pool order, not sorted membership.
    # In particular the original held-out pool interleaves two specialist blocks.
    pools = {}
    for label in ("train","test"):
        with h5py.File(record["original_rollouts"][label], "r") as handle:
            pools[label] = handle["capability_pool"][:].tolist()
    pools["additional"] = [(p["red_delay"],p["blue_delay"]) for p in suite["profiles"] if p["requires_new_profile_rollout"]]
    return pools


def recover(path, index, only=None, diagnostic=False):
    suite = full.read(path); record = suite["policies"][index]
    evaluator = full.load_evaluator(record)
    full.require(evaluator.jax.default_backend() == "gpu", "Recovery must use the original GPU backend")
    pools = recovery_pools(suite,record)
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
        selector = 'paired[(paired.version==version)&(paired.condition==full.CONDITIONS[0])&(paired.category==cat)&(paired.cohort=="one_goal_no_round")]'
        code = code.replace("g="+selector+".iloc[0]", "selected="+selector+"\n            if selected.empty:\n                ax.text(j,.02,'Insufficient cohort',rotation=90,ha='center',va='bottom',fontsize=8)\n                continue\n            g=selected.iloc[0]")
        behavioral.write_text(code)
        shared_actions = base / "action_probabilities_rewards_first50_20261023"
        if shared_actions.exists() and not (directory / shared_actions.name).exists():
            (directory / shared_actions.name).symlink_to(shared_actions,target_is_directory=True)
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
            "The action_probabilities_rewards_first50_20261023 directory links to the unchanged frozen-policy action/reward analysis; "
            "those quantities depend on the policy and episode cohort and are identical for all probe representations.\n\n"
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


def load_behavior_cache(record):
    """Read only the first101 behavioral rows needed by the early diagnostics."""
    fields = ("partner_assignment","ego_alloc_action","partner_goal","round_idx","round_time",
              "round_done","success","is_t0")
    parts = []
    for name in ("train","test","additional"):
        path = Path(record["feature_directory"]) / (name+".h5")
        audit = full.read(path.with_suffix(".json"))
        full.require(audit["status"]=="passed", "Unvalidated behavioral input")
        full.require(full.stats(path)==audit["archive_stats"], "Changed behavioral input")
        with h5py.File(path) as h:
            item = {key:h[key][:,:101] for key in fields}
            item.update({key:h[key][:] for key in ("pairs","repetitions","lengths")})
            parts.append(item)
    result = {key:np.concatenate([part[key] for part in parts]) for key in parts[0]}
    order = np.lexsort((result["repetitions"],result["pairs"][:,1],result["pairs"][:,0]))
    return {key:values[order] for key,values in result.items()}


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


def aggregate(path, early=True):
    from analyze_instantaneous_decoding import bootstrap_seeds, profile_metrics
    run = full.read(path); verify_sources(run); root = Path(run["root"])
    sys.path.insert(0, str(root / "source"))
    from plot_instantaneous_decoding import plot_profiles, plot_categories
    profiles = full.read(run["input_manifest"])["profiles"]
    pairs = np.array([(p["red_delay"],p["blue_delay"]) for p in profiles])
    conditions = tuple(run["conditions"])
    accuracy = np.full((3,len(conditions),5,401,90),np.nan)
    active = np.zeros_like(accuracy,dtype=np.uint8)
    pooled = []; audits = []
    for record in run["policies"]:
        directory = Path(record["probe_directory"]); archive = directory / "training_capability_probes.npz"
        done = full.read(directory / "completion.json")
        full.require(done["status"] == "completed" and full.sha(archive) == done["archive_sha256"], "Changed or incomplete readout")
        with np.load(archive) as z:
            np.testing.assert_array_equal(z["profiles"],pairs)
            np.testing.assert_array_equal(z["truth"],(z["pairs"][:,1]<z["pairs"][:,0]).astype(int))
            np.testing.assert_array_equal(z["predictions"],z["probabilities"].argmax(axis=-1))
            with np.load(Path(record["baseline_probe_directory"]) / "training_capability_probes.npz") as old:
                for key in ("pairs","repetitions","train","test","unused","truth","lengths"):
                    np.testing.assert_array_equal(z[key],old[key])
            recomputed = profile_metrics(z["predictions"],z["probabilities"],z["pairs"],z["truth"],z["test"],z["lengths"])
            for key,value in recomputed.items(): np.testing.assert_array_equal(z[key],value)
            v = full.VERSIONS.index(record["version"]); c = conditions.index(record["condition"]); s = record["seed"]-1
            accuracy[v,c,s] = z["accuracy"]; active[v,c,s] = z["active_test_episodes"]
            correct = z["predictions"] == z["truth"][None,:]
            test = z["test"]; truth = z["truth"]
            for t in STEPS:
                pooled.append(dict(**full.identity(record), env_step=int(t), accuracy=float(correct[t,test].mean()),
                                   red_faster_recall=float(correct[t,test & (truth==0)].mean()),
                                   blue_faster_recall=float(correct[t,test & (truth==1)].mean()), n_test_episodes=360))
        audits.append(done)
    full.require(np.isfinite(accuracy).all(), "Missing policy/profile observations")
    means,lows,highs = bootstrap_seeds(accuracy.transpose(2,0,1,3,4))
    np.savez_compressed(root / "profile_probe_summary.npz",accuracy_mean=means,accuracy_ci_low=lows,accuracy_ci_high=highs,
                        accuracy_per_seed=accuracy,active_test_episodes=active,env_steps=STEPS,profiles=pairs,
                        versions=full.VERSIONS,conditions=conditions,seeds=np.arange(1,6),probe_fit_profiles=run["probe_fit_profiles"],
                        representation=run["representation"])
    rows = []
    for v,version in enumerate(full.VERSIONS):
        for c,condition in enumerate(conditions):
            for i,p in enumerate(profiles):
                for t in STEPS:
                    rows.append(dict(version=version,condition=condition,d_R=p["red_delay"],d_B=p["blue_delay"],category=p["category"],
                                     probe_fit_profile=bool(p["seen_by_diverse_training"]),env_step=int(t),accuracy_mean=means[v,c,t,i],
                                     accuracy_ci_low=lows[v,c,t,i],accuracy_ci_high=highs[v,c,t,i],seed_count=5,
                                     active_test_episodes=int(active[v,c,:,t,i].sum())))
    pd.DataFrame(rows).to_csv(root / "profile_probe_summary.csv",index=False)
    pd.DataFrame(pooled).to_csv(root / "pooled_probe_per_seed.csv",index=False)
    labels = [full.LABELS[full.CONDITIONS.index(c)] for c in conditions]
    note = run["representation_label"] + f" ({run['feature_dimension']} inputs) · Fit: 24 profiles ×16 episodes; test: all90 ×4; five policy seeds."
    print("[plot all90 profiles]", run["representation"],flush=True)
    plot_profiles(means,lows,highs,profiles,root,conditions=conditions,labels=labels,method_note=note)
    plot_categories(accuracy,profiles,root,conditions=conditions,labels=labels,method_note=note)
    full.require(len(list((root / "per_profile").glob("*.png")))==90,"Missing individual profile figures")
    if early:
        early_behavior(run)
    run.update(status="completed",completed_utc=full.now())
    full.write(path,run)
    full.write(root / "completion_audit.json",dict(status="passed",completed_utc=full.now(),representation=run["representation"],
               feature_dimension=run["feature_dimension"],policies=30,policy_seed_count=5,versions=3,profiles=90,
               fit_profiles=24,novel_fit_profiles=0,fit_episodes_per_policy=384,test_episodes_per_policy=360,
               unused_episodes_per_policy=1056,train_test_overlap=0,test_episode_split_matches_previous=True,
               timepoints=401,total_independent_probes=30*401,per_profile_figures=90,metrics_recomputed=True,
               trajectory_reconstruction=True,policy_retraining=False,early_behavior=early,
               baseline_test_decision_disagreements=sum(a.get("baseline_test_decision_disagreements",0) for a in audits)))
    (root / "README.md").write_text(description(run) + "\nAll per-profile metrics and episode partitions were independently verified.\n")
    print("[representation completed]",run["representation"],root,flush=True)


def early_behavior(run):
    import analyze_decoding_behavior as behavior
    from plot_early_decoding import summarize
    root = Path(run["root"]); out = root / "early_behavior_20261026"
    out.mkdir(exist_ok=True); (out / "source").mkdir(exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copyfile(root / "source" / name,out / "source" / name)
    original = full.read(run["input_manifest"])
    onsets = behavior.onset_statistics(root)
    onsets.to_csv(out / "onset_summary.csv",index=False)
    collections = [[] for _ in range(5)]
    for record in run["policies"]:
        print("[early behavior]",run["representation"],full.identity(record),flush=True)
        data = load_behavior_cache(record)
        with np.load(Path(record["probe_directory"]) / "training_capability_probes.npz") as z:
            for key in ("test","truth","pairs","repetitions","lengths"): data[key] = z[key].copy()
            data["predictions"] = z["predictions"][:101]
            data["probabilities"] = z["probabilities"][:101]
            data["fixed20_probabilities"] = z["fixed20_probabilities"][:101].transpose(1,0,2)
        for key in ("partner_assignment","ego_alloc_action","partner_goal","round_idx","round_time","round_done","success","is_t0"):
            data[key] = data[key][:,:101]
        for bucket,table in zip(collections,behavior.policy_tables(record,data,original,onsets)):
            bucket.append(table)
    names = ("curves","paired_cohorts","event_aligned","episode_events","pink_strata")
    tables = {name:pd.concat(items,ignore_index=True) for name,items in zip(names,collections)}
    for name,table in tables.items(): table.to_csv(out / (name+"_per_seed.csv"),index=False)
    coverage = tables["paired_cohorts"].groupby(["version","condition","category","cohort"]).agg(
        n_seeds=("training_seed","nunique"),episodes_min=("n_episodes","min"),episodes_max=("n_episodes","max"))
    coverage.to_csv(out / "paired_cohort_coverage.csv")
    summary = behavior.summarize_curves(tables["curves"])
    summary.to_csv(out / "behavior_summary.csv",index=False)
    paired = summarize(tables["paired_cohorts"],["version","condition","category","cohort"],
                       ["accuracy_before","accuracy_after","paired_gain","fixed20_gain","n_episodes","n_profiles"])
    paired.to_csv(out / "paired_cohort_summary.csv",index=False)
    strata = summarize(tables["pink_strata"],["version","condition","stratum","env_step"],["accuracy"])
    strata.to_csv(out / "pink_strata_summary.csv",index=False)
    summarize(tables["event_aligned"],["version","condition","category","event","lag"],
              ["accuracy","correct_probability","fixed20_probability"],min_seeds=3).to_csv(out / "event_aligned_summary.csv",index=False)
    snapshots = []
    for row in onsets.itertuples():
        for t in (int(row.peak5_start),int(row.peak5_end)):
            g = summary[(summary.version==row.version)&(summary.condition==row.condition)&(summary.category==row.category)&(summary.env_step==t)].iloc[0]
            snapshots.append(dict(version=row.version,condition=row.condition,category=row.category,env_step=t,
                                  **{metric:g[metric+"_mean"] for metric in behavior.METRICS}))
    pd.DataFrame(snapshots).to_csv(out / "behavior_at_rise_windows.csv",index=False)
    behavior.figures(summary,onsets,paired,strata,out)
    full.write(out / "saved_behavior_audit.json",dict(status="passed",representation=run["representation"],
               policies=30,primary_multi_partner_policies=15,primary_test_episodes=4320,timepoints=101,
               identical_episode_split=True,equal_profile_weights=True,fixed_decoder_step=20,trajectory_reconstruction=True,
               onset="Largest five-step gain within 0..50; category/version-specific windows recomputed for this representation.",
               temporal_rule="Prior-event histories exclude transition t; action/assignment at t follows the observation.",policy_retraining=False))
    primary = onsets[onsets.condition=="rnn_diverse_influence"]
    lines = ["# Early behavior and decoding\n", "Representation: "+run["representation_label"]+". "
             "Windows are recomputed from each version/category mean curve as the maximum five-step accuracy increase within0..50. "
             "The same held-out episode IDs and five-seed aggregation are used as in the baseline.\n",
             "| Version | Category | Window | Accuracy before | Accuracy after |",
             "|---|---|---|---:|---:|"]
    for row in primary.itertuples():
        lines.append(f"| {row.version} | {row.category} | {int(row.peak5_start)}–{int(row.peak5_end)} | {row.accuracy_start:.4f} | {row.accuracy_end:.4f} |")
    lines += ["\nPaired gains use identical selected episodes at the window endpoints, equal profile weights within seed, "
              "and all five policy seeds. A category/cohort lacking any seed is omitted from its paired dot and identified in paired_cohort_coverage.csv. "
              "Motion diagnostics retain the original fixed cohort validated through step30; window endpoints outside this motion range are not summarized.\n"]
    (out / "FINDINGS.md").write_text("\n".join(lines))
    full.write(out / "analysis_findings.json",dict(representation=run["representation"],
               onsets=json.loads(onsets.to_json(orient="records")),
               paired_cohorts=json.loads(paired.to_json(orient="records"))))
    early_motion(run,out,onsets)


def early_motion(run,out,onsets):
    import analyze_decoding_behavior as behavior
    import analyze_replayed_motion as motion
    from plot_early_decoding import summarize
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    original = Path(run["baseline_root"]) / "early_behavior_20261023"
    link = out / "anchored_replay"
    if not link.exists(): link.symlink_to(original / "anchored_replay",target_is_directory=True)
    prior = pd.read_csv(original / "motion_per_seed.csv")
    lookup = {(p["red_delay"],p["blue_delay"]):p["category"] for p in full.read(run["input_manifest"])["profiles"]}
    frames = []; total = 0; valid_total = 0
    primary = [record for record in run["policies"] if record["condition"]=="rnn_diverse_influence"]
    for record in primary:
        data,audit = motion.load_motion(record,original)
        valid = data["verified_prefix"][:,30]; total += len(valid); valid_total += int(valid.sum())
        with np.load(Path(record["probe_directory"]) / "training_capability_probes.npz") as z:
            test = z["test"]; np.testing.assert_array_equal(data["pairs"],z["pairs"][test])
            correct = z["predictions"][:31,test] == z["truth"][test][None,:]
        labels = np.array([lookup[tuple(p)] for p in data["pairs"]])
        frame = prior[(prior.version==record["version"])&(prior.training_seed==record["seed"])].copy()
        for cat in behavior.CATEGORIES:
            chosen = valid & (labels==cat)
            indices = frame[frame.category==cat].sort_values("env_step").index
            np.testing.assert_array_equal(frame.loc[indices,"n_episodes"],np.repeat(chosen.sum(),31))
            frame.loc[indices,"accuracy"] = [behavior.equal_profile_mean(correct[t,chosen],data["pairs"][chosen]) for t in range(31)]
        frames.append(frame)
    table = pd.concat(frames,ignore_index=True)
    table.to_csv(out / "motion_per_seed.csv",index=False)
    metrics = [key for key in table if key not in ("version","condition","training_seed","category","env_step")]
    summary = summarize(table,["version","condition","category","env_step"],metrics)
    summary.to_csv(out / "motion_summary.csv",index=False)
    snapshots = []
    for row in onsets[onsets.condition=="rnn_diverse_influence"].itertuples():
        for t in (int(row.peak5_start),int(row.peak5_end)):
            g = summary[(summary.version==row.version)&(summary.category==row.category)&(summary.env_step==t)]
            if len(g): snapshots.append(g.iloc[0].to_dict())
    pd.DataFrame(snapshots).to_csv(out / "motion_at_rise_windows.csv",index=False)
    fig,axes = plt.subplots(3,3,figsize=(16,10.5),sharex=True,sharey=True)
    specs = (("accuracy",run["representation_label"]+" decoding","#9380ff"),
             ("first_partner_move_seen","First partner move observed","#555555"),
             ("second_partner_move_seen","At least two partner moves observed","#b36b00"),
             ("four_waits_after_move_seen","Four consecutive off-goal stays after a move","#168e88"),
             ("round_feedback_seen","First round feedback already observed","#111111"))
    for r,cat in enumerate(behavior.CATEGORIES):
        for c,version in enumerate(full.VERSIONS):
            ax = axes[r,c]; g = summary[(summary.version==version)&(summary.category==cat)].sort_values("env_step")
            for key,label,color in specs: ax.plot(g.env_step,g[key+"_mean"],label=label,color=color,lw=1.6)
            onset = onsets[(onsets.version==version)&(onsets.condition=="rnn_diverse_influence")&(onsets.category==cat)].iloc[0]
            ax.axvspan(onset.peak5_start,onset.peak5_end,color="gray",alpha=.15)
            ax.set_facecolor(full.CATEGORY_COLORS[cat]);ax.spines[["top","right"]].set_visible(False)
            ax.set(title=f"{version} · {full.CATEGORY_LABELS[cat]}",xlim=(0,30),ylim=(0,1.02),xlabel="Environment step",ylabel="Accuracy / episode fraction")
            ax.title.set_fontsize(10)
    fig.suptitle(run["representation_label"]+" · decoding and reconstructed partner movement\nSame baseline cohort validated through step30; five seeds; gray windows recomputed per representation",fontsize=12)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.94))
    for ext in ("png","pdf"): fig.savefig(out / f"decoding_and_partner_motion.{ext}",dpi=180)
    plt.close(fig)
    full.write(out / "motion_audit.json",dict(status="passed",representation=run["representation"],policies=15,
               test_episodes=total,verified_episode_prefixes_through_step30=valid_total,
               same_baseline_validated_cohort=True,behavior_metrics_reused_from=str(original / "motion_per_seed.csv"),
               window_snapshots="Only selected window endpoints within the verified 0..30 motion range are reported.",policy_retraining=False))
    example_figure(run,out,primary[0],onsets)
    import analyze_decoding_behavior_correlations as correlation
    directory = out / "correlation_analysis_20261026"; directory.mkdir(exist_ok=True)
    values,matched,audits = correlation.load_matched(run,out)
    matched.to_csv(directory / "matched_curves_per_seed.csv",index=False)
    np.savez_compressed(directory / "matched_curves.npz",values=values,versions=full.VERSIONS,categories=behavior.CATEGORIES,
                        metrics=("accuracy",)+correlation.METRICS,env_steps=np.arange(31),seeds=np.arange(1,6))
    summary,panel,differences = correlation.summarize(values,directory)
    correlation.plot(summary,panel,directory)
    full.write(directory / "analysis_audit.json",dict(status="passed",representation=run["representation"],policy_audits=audits,
               same_validated_cohort=True,panels=9,steps=[0,30],policy_retraining=False,probe_refitting=False))
    (directory / "FINDINGS.md").write_text("# Behavior/decoding curve associations\n\nRepresentation: "+run["representation_label"]+
        ". Same baseline replay-validated episode cohort; equal profile weights, then five-seed means. "
        "Pooled correlations center each panel separately. Levels use31 timepoints, one-step changes30, five-step changes26 per panel. "
        "Displayed intervals use the stratified policy-seed Fisher-z jackknife with t12.\n\n"+summary.to_csv(index=False))


def example_figure(run,out,record,onsets):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    replay = out / "anchored_replay/v1_matched/rnn_diverse_influence_seed1/train_replay.npz"
    with np.load(replay) as z:
        e = np.flatnonzero((z["pairs"]==[1,7]).all(axis=1)&(z["repetitions"]==1))[0]
        full.require(z["verified_prefix"][e,:31].all(),"Unvalidated illustrative episode")
        positions = z["pre_positions"][e,:31]
        goals = z["red_goal"][e,0],z["blue_goal"][e,0]
        walls = z["wall_maps"][int(z["pre_layout"][e,0])]
        moved = (z["pre_positions"][e,:,1]!=z["post_positions"][e,:,1]).any(axis=-1)&~z["round_done"][e]
        first_moves = np.flatnonzero(moved[:30])+1
        clock = z["pre_clock"][e,:31]
    with np.load(Path(record["probe_directory"]) / "training_capability_probes.npz") as z:
        chosen = (z["pairs"]==[1,7]).all(axis=1)&(z["repetitions"]==1)
        p = z["probabilities"][:31,chosen,0].ravel(); fixed = z["fixed20_probabilities"][:31,chosen,0].ravel()
    pd.DataFrame(dict(env_step=np.arange(31),red_faster_probability=p,fixed20_red_probability=fixed,
                      partner_x=positions[:,1,0],partner_y=positions[:,1,1],partner_cooldown_clock=clock,
                      partner_move_just_observed=np.isin(np.arange(31),first_moves))).to_csv(out / "example_1_7_seed1_rep1.csv",index=False)
    fig = plt.figure(figsize=(15,8));grid=fig.add_gridspec(3,6,height_ratios=[1.2,1,.35],hspace=.55)
    for c,t in enumerate((0,2,4,6,8,10)):
        ax=fig.add_subplot(grid[0,c]);ax.imshow(walls,cmap="Greys",vmin=0,vmax=1,origin="upper")
        for goal,color in zip(goals,("#d94b41","#3779bf")): ax.add_patch(Rectangle((goal[0]-.4,goal[1]-.4),.8,.8,color=color,alpha=.5))
        ax.plot(positions[t,0,0],positions[t,0,1],"o",color="#e67e00",ms=10)
        ax.plot(positions[t,1,0],positions[t,1,1],"s",color="#222222",ms=9)
        ax.set(title=f"Observation t={t}",xticks=[],yticks=[])
    ax=fig.add_subplot(grid[1,:]);ax.plot(np.arange(31),p,color="#9380ff",lw=2,label="Time-specific decoder")
    ax.plot(np.arange(31),fixed,color="#5d3b9c",lw=1.5,ls="--",label="Fixed step-20 decoder")
    for t in first_moves: ax.axvline(t,color="#333333",alpha=.3,lw=1)
    onset=onsets[(onsets.version=="v1_matched")&(onsets.condition=="rnn_diverse_influence")&(onsets.category=="training_profile")].iloc[0]
    ax.axvspan(onset.peak5_start,onset.peak5_end,color="gray",alpha=.15,label="Category steepest window")
    ax.axhline(.5,color="gray",ls=":");ax.set(xlim=(0,25),ylim=(0,1.03),ylabel="Probability: red faster",xlabel="Environment step")
    ax.spines[["top","right"]].set_visible(False);ax.legend(loc="lower right",fontsize=9)
    bottom=fig.add_subplot(grid[2,:]);bottom.eventplot(first_moves,lineoffsets=.5,linelengths=.5,color="black")
    bottom.set(xlim=(0,25),ylim=(0,1),yticks=[.5],yticklabels=["Partner move observed"],xlabel="Blue pursued throughout")
    bottom.spines[["top","right","left"]].set_visible(False)
    fig.suptitle(run["representation_label"]+" · held-out v1_matched seed1, profile(1,7), repetition1",fontsize=12)
    for ext in ("png","pdf"):fig.savefig(out / f"example_partner_waiting.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)


def submit_fits(path):
    suite = full.read(path); root = Path(suite["root"])
    full.require("recovery" in suite["jobs"], "Submit recovery first")
    full.require("fit" not in suite["jobs"], "Fit jobs already submitted")
    environment = {key:value for key,value in os.environ.items() if not key.startswith("SLURM_")}
    recovery = suite["jobs"]["recovery"]["job_id"]
    stages = [("fit_"+name,"worker","0-29%4","8G","02:00:00","aftercorr:"+recovery,
               root / name / "analysis_run.json",root / name / "source/representation_probe_suite.py") for name in REPRESENTATIONS]
    stages += [("aggregate","aggregate-suite","0-7%4","16G","02:00:00",None,path.resolve(),root / "source/representation_probe_suite.py"),
               ("finalize","finalize",None,"8G","00:30:00",None,path.resolve(),root / "source/representation_probe_suite.py")]
    fit_jobs = []; previous = None
    for stage,mode,array,memory,limit,dependency,manifest,producer in stages:
        if dependency is None:
            dependency = "afterok:" + (":".join(fit_jobs) if stage=="aggregate" else previous)
        script = root / "jobs" / (stage + ".sh")
        command = [PYTHON,"-u",str(producer),mode,str(manifest)]
        import shlex
        line = shlex.join(command) + (' --index "${SLURM_ARRAY_TASK_ID}"' if array else "")
        script.write_text("\n".join(["#!/bin/bash","set -euo pipefail",
            "export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1",
            "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1",
            "export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=",'export XLA_FLAGS=""',
            "export MPLCONFIGDIR="+shlex.quote(str(root / "mpl_cache")),"exec "+line])+"\n")
        subprocess.run(["bash","-n",str(script)],check=True)
        command = ["sbatch","--parsable","--account=nlp","--partition=sphinx","--exclude=sphinx9",
                   "--cpus-per-task=1","--mem="+memory,"--time="+limit,"--job-name=repr_"+stage,
                   "--dependency="+dependency,"--kill-on-invalid-dep=yes",
                   "--output="+str(root / "logs" / (stage+"_%A_%a.out")),
                   "--error="+str(root / "logs" / (stage+"_%A_%a.err"))]
        if array: command += ["--array="+array]
        command.append(str(script))
        job = subprocess.check_output(command,text=True,env=environment).strip().split(";")[0]
        full.require(job.isdigit(),"Invalid scheduler job ID")
        suite["jobs"][stage] = dict(job_id=job,command=command)
        if stage.startswith("fit_"): fit_jobs.append(job)
        previous = job
        suite["status"] = "submitted"
        full.write(path,suite)
        print("[submitted]",stage,job,flush=True)
    suite["jobs"]["fit"] = dict(job_ids=fit_jobs,dependency="Corresponding recovery task; fits start as soon as each policy is recovered.")
    full.write(path,suite)


def finalize(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    suite = full.read(path); root = Path(suite["root"]); tables = []; audits = []
    for name in REPRESENTATIONS:
        directory = root / name
        audit = full.read(directory / "completion_audit.json")
        full.require(audit["status"]=="passed", "Incomplete representation " + name)
        full.require(full.read(directory / "analysis_run.json")["status"]=="completed", "Incomplete run manifest")
        frame = pd.read_csv(directory / "category_averages/category_probe_summary.csv")
        frame["representation"] = name; frame["feature_dimension"] = REPRESENTATIONS[name][0]
        tables.append(frame); audits.append(audit)
    table = pd.concat(tables,ignore_index=True)
    table.to_csv(root / "representation_comparison.csv",index=False)
    table[table.env_step.isin([0,5,10,20,50,100,200,400])].to_csv(root / "representation_comparison_selected_steps.csv",index=False)
    colors = ("#777777","#b18435","#7862cc","#d55870","#219e8f","#2673b8","#b349a0","#4e7d25")
    for condition in suite["conditions"]:
        for end,suffix in ((60,"first60"),(400,"all400")):
            fig,axes = plt.subplots(4,3,figsize=(17,13),sharex=True,sharey=True)
            for row,category in enumerate(full.CATEGORY_COLORS):
                for col,version in enumerate(full.VERSIONS):
                    ax = axes[row,col]
                    for name,color in zip(REPRESENTATIONS,colors):
                        g = table[(table.representation==name)&(table.condition==condition)&(table.version==version)&(table.category==category)].sort_values("env_step")
                        ax.plot(g.env_step,g.accuracy_mean,label=REPRESENTATIONS[name][1],color=color,lw=1.4)
                    ax.axhline(.5,color="gray",ls="--",lw=.7)
                    ax.set_facecolor(full.CATEGORY_COLORS[category]);ax.spines[["top","right"]].set_visible(False)
                    ax.set(title=version+" · "+full.CATEGORY_LABELS[category],xlim=(0,end),ylim=(.2,1.02),xlabel="Observation index",ylabel="Held-out accuracy")
                    ax.title.set_fontsize(9)
            fig.suptitle("Representation comparison · "+full.LABELS[full.CONDITIONS.index(condition)]+"\nSame 24-profile fitting set and four held-out episodes for all90 profiles; mean five policy seeds",fontsize=12)
            fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=4,frameon=False,fontsize=9)
            fig.tight_layout(rect=(0,.06,1,.94))
            for ext in ("png","pdf"):fig.savefig(root / f"representation_comparison_{condition}_{suffix}.{ext}",dpi=170)
            plt.close(fig)
    recoveries = []
    for record in suite["policies"]:
        for name in ("train","test","additional"):
            audit = full.read(cache_directory(suite,record) / (name+".json"))
            full.require(audit["status"]=="passed", "Missing recovery audit")
            recoveries.append(audit)
    full.write(root / "completion_audit.json",dict(status="passed",completed_utc=full.now(),representations=8,
               policies_per_representation=30,profiles=90,timepoints=401,fit_examples_per_policy=384,test_examples_per_policy=360,
               independent_probes=8*30*401,per_profile_pngs=8*90,recovered_slices=len(recoveries),
               maximum_recovery_probability_error=max(a["probability_max_error"] for a in recoveries),
               recovered_used_decision_disagreements=sum(a["used_prediction_disagreements"] for a in recoveries),
               original_result_directory_unchanged=True,policy_retraining=False,representation_audits=audits))
    lines = ["# Matched representation probe suite\n", "Eight representations; identical 24-profile x16-episode fitting set and all90 x4 held-out testing set. "
             "Separate affine classifiers at each step0..400 for thirty frozen policies. Recovered inputs share the original episode IDs and RNG schedules.\n"]
    lines += [f"- [{label}]({name}/README.md): {dimension} input features." for name,(dimension,label) in REPRESENTATIONS.items()]
    lines += ["\nEach directory contains per-policy readouts/partitions, ninety individual profile plots, a multipage PDF, "
              "combined all-profile figures, category averages, and recomputed early-behavior/motion/correlation diagnostics.\n",
              "Comparison CSVs and figures are in this directory. completion_audit.json records the final verification.\n"]
    (root / "README.md").write_text("\n".join(lines))
    suite.update(status="completed",completed_utc=full.now())
    full.write(path,suite)
    print("[suite completed]",root,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "recover", "initialize", "worker", "fit", "aggregate", "aggregate-suite", "submit", "finalize"))
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_OUT / "suite_run.json")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--only", choices=("train", "test", "additional"))
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare": prepare(args.base, args.out)
    elif args.mode == "recover": recover(args.path, args.index, args.only, args.diagnostic)
    elif args.mode == "initialize": initialize_runs(args.path)
    elif args.mode == "worker": worker(args.path,args.index)
    elif args.mode == "aggregate": aggregate(args.path)
    elif args.mode == "submit": submit_fits(args.path)
    elif args.mode == "finalize": finalize(args.path)
    else:
        suite = full.read(args.path); root = Path(suite["root"])
        names = tuple(REPRESENTATIONS)
        if args.mode == "fit":
            representation = names[args.index // len(suite["policies"])]
            worker(root / representation / "analysis_run.json",args.index % len(suite["policies"]))
        else:
            aggregate(root / names[args.index] / "analysis_run.json")


if __name__ == "__main__":
    main()
