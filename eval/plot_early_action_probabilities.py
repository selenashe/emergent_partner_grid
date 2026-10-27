"""Reconstruct frozen actor probabilities and plot recorded rewards at t=0..49.

Uses exactly the held-out episodes of the training-capability decoding study.
No environment transitions, policy training, or probe fitting are performed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np
import pandas as pd
from safetensors.numpy import load_file

import full_capability_evaluation as full
from analyze_instantaneous_decoding import bootstrap_seeds

STEPS = np.arange(50)
MOVES = ("Up", "Down", "Right", "Left", "Stay")
ALLOCATIONS = ("None", "Red", "Blue")
CATEGORIES = tuple(full.CATEGORY_COLORS)


def action_mask(version, condition, is_t0, ego_alloc_action):
    """Exact frozen protocol masks, including supplied forced assignments.

    v2 and v3/no-influence force the supplied allocation at round t=0.
    The recorded executed allocation therefore identifies that supplied value.
    All other initial choices are free between stay/red and stay/blue.
    """
    is_t0 = np.asarray(is_t0, dtype=bool)
    alloc = np.asarray(ego_alloc_action)
    ids = np.arange(15)
    if version == "v1_matched":
        legal = np.broadcast_to(ids % 3 == 0, is_t0.shape + (15,)).copy()
    else:
        legal = np.broadcast_to(ids % 3 != 0, is_t0.shape + (15,)).copy()
    forced = version == "v2" or (version == "v3" and condition == "rnn_diverse_noinfluence")
    if forced:
        full.require(np.isin(alloc[is_t0], [1, 2]).all(), "Invalid forced initial allocation")
        initial = ids == (12 + alloc[..., None])
    else:
        initial = np.isin(ids, [13, 14])
    return np.where(is_t0[..., None], initial, legal)


def actor_probabilities(hidden, params, legal):
    x = np.maximum(hidden @ params["params,Dense_0,kernel"] + params["params,Dense_0,bias"], 0)
    logits = x @ params["params,Dense_1,kernel"] + params["params,Dense_1,bias"]
    logits = np.where(legal, logits, -np.inf)
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities


def joint_marginals(probabilities):
    p = probabilities.reshape(probabilities.shape[:-1] + (5, 3))
    return p.sum(axis=-1), p.sum(axis=-2)


def read_episodes(record):
    archive = Path(record["probe_directory"]) / "training_capability_probes.npz"
    done = full.read(archive.with_name("completion.json"))
    full.require(full.sha(archive) == done["archive_sha256"], "Changed decoding episode archive")
    with np.load(archive) as probe:
        selected = probe["test"]
        reference = {k: probe[k][selected].copy() for k in ("pairs", "repetitions", "lengths")}
    batches = []; schemas = []
    paths = list(record["original_rollouts"].values()) + [str(Path(record["directory"]) / "additional_profiles.h5")]
    for path in paths:
        with h5py.File(path, "r") as saved:
            pairs = saved["capability"][:, 0, :]
            counters = {}; reps = []
            for pair in map(tuple, pairs):
                reps.append(counters.get(pair, 0)); counters[pair] = reps[-1] + 1
            full.require(set(counters.values()) == {20}, "Expected 20 saved episodes per profile")
            reps = np.array(reps)
            chosen = np.isin(reps, [1, 9, 14, 15])
            # Contiguous time slices avoid reading full 2000-step hidden arrays.
            item = {k: saved[k][:, :50][chosen] for k in ("hidden_state", "rewards", "is_t0", "ego_alloc_action", "dones")}
            full.require(not item["dones"].any(), "Episode termination inside the requested window")
            item.update(pairs=pairs[chosen], repetitions=reps[chosen])
            batches.append(item)
            schemas.append(dict(path=str(path), stats=full.stats(path), datasets=list(saved.keys())))
    data = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
    order = np.lexsort((data["repetitions"], data["pairs"][:, 1], data["pairs"][:, 0]))
    data = {k: v[order] for k, v in data.items()}
    for k in ("pairs", "repetitions"):
        np.testing.assert_array_equal(data[k], reference[k])
    full.require(len(data["pairs"]) == 360 and reference["lengths"].min() > 50, "Incorrect held-out cohort")
    full.require(np.isfinite(data["hidden_state"]).all() and np.isfinite(data["rewards"]).all(), "Nonfinite inputs")
    data["lengths"] = reference["lengths"]
    return data, schemas


def extract_policy(task):
    record, out = task
    out = Path(out) / "policies" / record["version"] / f"{record['condition']}_seed{record['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "completion.json").exists():
        return full.read(out / "completion.json")
    print("[read saved actor inputs]", full.identity(record), flush=True)
    for k in ("checkpoint", "config"):
        full.require(full.sha(record[k]) == record[k + "_sha256"], "Changed policy " + k)
    for name, path in record["original_rollouts"].items():
        full.require(full.stats(path) == record["original_stats"][name], "Changed original rollout")
    extra = full.read(Path(record["directory"]) / "evaluation_completion.json")
    full.require(full.stats(Path(record["directory"]) / "additional_profiles.h5") == extra["rollout_stats"], "Changed additional rollout")
    config = full.read(record["config"])
    full.require(config["GRID_ENCODER"] == "flat" and config["FC_DIM_SIZE"] == 128, "Unexpected actor architecture")
    data, schemas = read_episodes(record)
    legal = action_mask(record["version"], record["condition"], data["is_t0"], data["ego_alloc_action"])
    p = actor_probabilities(data.pop("hidden_state"), load_file(record["checkpoint"]), legal)
    full.require(p.shape == (360, 50, 15) and np.isfinite(p).all(), "Invalid action probabilities")
    np.testing.assert_allclose(p.sum(axis=-1), 1, atol=1e-6, rtol=0)
    full.require((p[~legal] == 0).all(), "Illegal action has positive probability")
    data.update(action_probabilities=p, env_steps=STEPS)
    archive = out / "episode_action_probabilities_rewards.npz"
    np.savez_compressed(archive, **data)
    audit = dict(status="completed", **full.identity(record), archive=str(archive), archive_sha256=full.sha(archive),
        episodes=360, profiles=90, steps=50, test_repetitions=[1, 9, 14, 15],
        reward_source="original saved HDF5 rewards", probability_source="frozen actor head applied to original saved h_t with exact phase/action mask",
        checkpoint_sha256=record["checkpoint_sha256"], input_rollouts=schemas,
        probabilities_normalized=True, normalization_max_absolute_error=float(np.max(np.abs(p.sum(axis=-1)-1))),
        illegal_probability_zero=True, episode_padding=False,
        policy_retraining=False, probe_retraining=False, new_rollouts=False)
    full.write(out / "completion.json", audit)
    print("[extracted]", full.identity(record), flush=True)
    return audit


def reference_check(run_path, index, out):
    """Compare NumPy reconstruction with the original frozen Flax network.

    Feed recorded h_t directly into its ScannedRNN output. Dummy grids are
    irrelevant after this interception; original legality masks still execute.
    """
    run = full.read(run_path); record = run["policies"][index]
    evaluator = full.load_evaluator(record)
    jax, jnp = evaluator.jax, evaluator.jnp
    config = full.read(record["config"])
    data, _ = read_episodes(record)
    indices = np.arange(0, 360, 17)
    times = np.array([0, 1, 5, 10, 20, 35, 49])
    hidden = data["hidden_state"][indices[:, None], times[None, :]].reshape(-1, 128)
    initial = data["is_t0"][indices[:, None], times[None, :]].reshape(-1)
    alloc = data["ego_alloc_action"][indices[:, None], times[None, :]].reshape(-1)
    legal = action_mask(record["version"], record["condition"], initial, alloc)
    wanted = actor_probabilities(hidden, load_file(record["checkpoint"]), legal)
    # Forced t0 assignments are observed; v3 influence starts unset. Non-t0
    # last_allocation never enters the mask. Match both cases explicitly.
    forced = record["version"] == "v2" or (record["version"] == "v3" and not config["INFLUENCE"])
    observed_alloc = np.where(initial & forced, alloc, 0)
    n = len(hidden)
    obs = dict(grid=jnp.zeros((1, n, 7, 7, 5)),
        last_allocation=jax.nn.one_hot(jnp.asarray(observed_alloc)[None, :], 3),
        is_t0=jnp.asarray(initial, dtype=jnp.float32)[None, :])
    network, init_hidden, _ = evaluator.trainer_mod.build_network(config, 15)
    params = evaluator.load_params(record["checkpoint"])
    def intercept(next_function, args, kwargs, context):
        if isinstance(context.module, evaluator.trainer_mod.ScannedRNN) and context.method_name == "__call__":
            return jnp.asarray(hidden), jnp.asarray(hidden)
        return next_function(*args, **kwargs)
    with evaluator.trainer_mod.nn.intercept_methods(intercept):
        _, pi, _ = network.apply(params, init_hidden(n), (obs, jnp.zeros((1, n), dtype=bool)))
    actual = np.asarray(pi.probs).reshape(n, 15)
    error = float(np.max(np.abs(actual - wanted)))
    np.testing.assert_allclose(actual, wanted, atol=3e-6, rtol=2e-5)
    full.write(out, dict(status="passed", **full.identity(record), compared_states=n, max_absolute_probability_error=error,
        reference="original frozen Flax network.apply / Distrax probabilities; saved post-observation state injected at ScannedRNN output",
        environment_rollouts=False, original_gpu_action_probabilities_were_not_recorded=True))
    print("[original actor reference passed]", full.identity(record), error, flush=True)


def summarize(run, out):
    base = full.read(run["input_manifest"])
    profiles = base["profiles"]; conditions = run["conditions"]
    pairs = np.array([(p["red_delay"], p["blue_delay"]) for p in profiles])
    probability = np.full((3, len(conditions), 5, 90, 50, 15), np.nan, dtype=np.float32)
    reward = np.full(probability.shape[:-1], np.nan, dtype=np.float32)
    audits = []
    for record in run["policies"]:
        d = out / "policies" / record["version"] / f"{record['condition']}_seed{record['seed']}"
        audit = full.read(d / "completion.json")
        full.require(full.sha(audit["archive"]) == audit["archive_sha256"], "Changed reconstructed archive")
        with np.load(audit["archive"]) as data:
            v = full.VERSIONS.index(record["version"]); c = conditions.index(record["condition"]); s = record["seed"] - 1
            for k, pair in enumerate(pairs):
                selected = (data["pairs"] == pair).all(axis=-1)
                full.require(selected.sum() == 4, "Profile has incorrect test episode count")
                probability[v, c, s, k] = data["action_probabilities"][selected].mean(axis=0)
                reward[v, c, s, k] = data["rewards"][selected].mean(axis=0)
        audits.append(audit)
    full.require(np.isfinite(probability).all() and np.isfinite(reward).all(), "Incomplete policy coverage")
    cat = np.array([p["category"] for p in profiles])
    category_probability = np.stack([probability[:, :, :, cat == k].mean(axis=3) for k in CATEGORIES], axis=3)
    category_reward = np.stack([reward[:, :, :, cat == k].mean(axis=3) for k in CATEGORIES], axis=3)
    common = dict(versions=np.array(full.VERSIONS), conditions=np.array(conditions), env_steps=STEPS)
    np.savez_compressed(out / "profile_summary_per_seed.npz", action_probabilities=probability, reward=reward, pairs=pairs, category=cat, **common)
    np.savez_compressed(out / "category_summary_per_seed.npz", action_probabilities=category_probability, reward=category_reward, categories=np.array(CATEGORIES), **common)
    tables = []
    for scope, probs, rewards, names in (("profile", probability, reward, range(90)), ("category", category_probability, category_reward, CATEGORIES)):
        mean_reward, lo, hi = bootstrap_seeds(rewards.transpose(2, 0, 1, 3, 4))
        means = probs.mean(axis=2)
        movement, allocation = joint_marginals(means)
        rows = []
        for v, version in enumerate(full.VERSIONS):
            for c, condition in enumerate(conditions):
                for k, name in enumerate(names):
                    for t in STEPS:
                        row = dict(version=version, condition=condition, env_step=int(t),
                            category=profiles[k]["category"] if scope == "profile" else name,
                            n_profiles=1 if scope == "profile" else int((cat == name).sum()),
                            n_test_episodes_per_seed=4 if scope == "profile" else 4 * int((cat == name).sum()),
                            n_seeds=5, reward_mean=float(mean_reward[v, c, k, t]),
                            reward_ci_low=float(lo[v, c, k, t]), reward_ci_high=float(hi[v, c, k, t]))
                        if scope == "profile": row.update(d_R=int(pairs[k, 0]), d_B=int(pairs[k, 1]))
                        row.update({f"p_action_{a:02d}": float(means[v, c, k, t, a]) for a in range(15)})
                        row.update({"p_move_" + move.lower(): float(movement[v, c, k, t, a]) for a, move in enumerate(MOVES)})
                        row.update({"p_alloc_" + alloc.lower(): float(allocation[v, c, k, t, a]) for a, alloc in enumerate(ALLOCATIONS)})
                        rows.append(row)
        pd.DataFrame(rows).to_csv(out / f"{scope}_summary.csv.gz", index=False, compression="gzip")
        tables.append((means, mean_reward, lo, hi))
    return profiles, tables, audits


def plot(run, out, profiles, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    # Movement is a hue family; allocation is a distinct shade within it.
    colors = ("#6baed6", "#2171b5", "#bdd7e7", "#fdae6b", "#d94801", "#fdd0a2",
              "#74c476", "#238b45", "#c7e9c0", "#fb6a4a", "#cb181d", "#fcbba1",
              "#9e9ac8", "#6a51a3", "#dadaeb")
    labels = [f"{m} / {a}" for m in MOVES for a in ("—", "R", "B")]
    handles = [Patch(color=color, label=label) for color, label in zip(colors, labels)]
    conditions = run["conditions"]
    def hist(ax, p):
        bottom = np.zeros(50)
        for a, color in enumerate(colors):
            if np.max(p[:, a]) > 0:
                top = bottom + p[:, a]
                vertices = np.stack([np.c_[STEPS-.47,bottom], np.c_[STEPS+.47,bottom],
                                     np.c_[STEPS+.47,top], np.c_[STEPS-.47,top]], axis=1)
                ax.add_collection(PolyCollection(vertices, facecolors=color, edgecolors="none", rasterized=True))
            bottom += p[:, a]
        ax.set(xlim=(-.7, 49.7), ylim=(0, 1), yticks=[0, .5, 1], ylabel="Action probability")
        ax.tick_params(labelbottom=False); ax.spines[["top", "right"]].set_visible(False)
    def reward_line(ax, mean, lo, hi, ymax):
        ax.fill_between(STEPS, lo, hi, color="#333333", alpha=.15, lw=0)
        ax.plot(STEPS, mean, color="#222222", lw=1.35)
        ax.axhline(0, color="#888888", lw=.6)
        ax.set(xlim=(-.7, 49.7), ylim=(-.025, ymax), xticks=[0, 10, 20, 30, 40, 49], ylabel="Mean step reward", xlabel="Environment step t")
        ax.spines[["top", "right"]].set_visible(False)
    def footer(fig, y=.025):
        fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(.5, y), frameon=False,
                   title="Joint action = movement / requested ego allocation (—: none; R: red; B: blue)")
        fig.text(.5, .006, "Probabilities reconstructed from saved hₜ + frozen actor; rewards from saved transitions. 4 test episodes/profile/seed × 5 seeds. Reward shading: 95% seed bootstrap CI. No time smoothing.", ha="center", fontsize=8)
    # Each page/profile: 3 experiments × 2 conditions × histogram/reward.
    probability, reward, low, high = tables[0]
    (out / "per_profile").mkdir(exist_ok=True)
    with PdfPages(out / "action_probabilities_rewards_all_profiles.pdf") as pdf:
        for k, profile in enumerate(profiles):
            fig = plt.figure(figsize=(16.5, 9.8))
            grid = fig.add_gridspec(4, 3, height_ratios=[1, .65, 1, .65], hspace=.38, wspace=.23, left=.065, right=.985, bottom=.17, top=.91)
            ymax = max(.025, float(high[:, :, k].max()) * 1.07)
            for c, condition in enumerate(conditions):
                for v, version in enumerate(full.VERSIONS):
                    ax = fig.add_subplot(grid[2*c, v]); hist(ax, probability[v, c, k])
                    ax.set_title(version + " · " + full.LABELS[full.CONDITIONS.index(condition)])
                    ax.set_facecolor(full.CATEGORY_COLORS[profile["category"]])
                    ax2 = fig.add_subplot(grid[2*c+1, v]); reward_line(ax2, reward[v, c, k], low[v, c, k], high[v, c, k], ymax)
                    if c < len(conditions)-1: ax2.set_xlabel("")
            fig.suptitle(f"Capability (d_R, d_B) = ({profile['red_delay']}, {profile['blue_delay']})  ·  {full.CATEGORY_LABELS[profile['category']]}", fontsize=15)
            footer(fig)
            target = out / "per_profile" / f"action_probabilities_rewards_dR{profile['red_delay']}_dB{profile['blue_delay']}.png"
            fig.savefig(target, dpi=125); pdf.savefig(fig, dpi=110); plt.close(fig)
            if k % 15 == 0: print("[plotted profiles]", k+1, "/ 90", flush=True)
    # All four exposure categories together, including the blue category.
    probability, reward, low, high = tables[1]
    for c, condition in enumerate(conditions):
        fig = plt.figure(figsize=(17, 16.8))
        grid = fig.add_gridspec(8, 3, height_ratios=[1, .6]*4, hspace=.4, wspace=.23, left=.065, right=.985, bottom=.10, top=.945)
        ymax = max(.025, float(high[:, c].max()) * 1.07)
        for k, category in enumerate(CATEGORIES):
            count = sum(p["category"] == category for p in profiles)
            for v, version in enumerate(full.VERSIONS):
                ax = fig.add_subplot(grid[2*k, v]); hist(ax, probability[v, c, k])
                ax.set_title(f"{version} · {full.CATEGORY_LABELS[category]} ({count})", fontsize=9)
                ax.set_facecolor(full.CATEGORY_COLORS[category])
                ax2 = fig.add_subplot(grid[2*k+1, v]); reward_line(ax2, reward[v, c, k], low[v, c, k], high[v, c, k], ymax)
                ax2.set_facecolor(full.CATEGORY_COLORS[category])
                if k < len(CATEGORIES)-1: ax2.set_xlabel("")
        fig.suptitle("Category averages · " + full.LABELS[full.CONDITIONS.index(condition)] + " · steps 0–49", fontsize=16)
        footer(fig, y=.018)
        suffix = "multi_partner" if condition == "rnn_diverse_influence" else "no_influence"
        fig.savefig(out / f"action_probabilities_rewards_category_averages_{suffix}.png", dpi=140)
        fig.savefig(out / f"action_probabilities_rewards_category_averages_{suffix}.pdf", dpi=110)
        plt.close(fig)
        print("[plotted categories]", condition, flush=True)
    pd.DataFrame([dict(action_id=a, movement=MOVES[a//3], requested_ego_allocation=ALLOCATIONS[a%3], color=colors[a]) for a in range(15)]).to_csv(out / "action_key.csv", index=False)


def main(path, out, workers):
    path, out = Path(path).resolve(), Path(out).resolve()
    run = full.read(path)
    full.require(run["status"] == "completed" and len(run["policies"]) == 30, "Expected completed 30-policy decoding study")
    out.mkdir(parents=True, exist_ok=True); (out / "source").mkdir(exist_ok=True)
    source_names = (Path(__file__).name, "full_capability_evaluation.py", "analyze_instantaneous_decoding.py")
    for name in source_names:
        source = Path(__file__).with_name(name); target = out / "source" / name
        if source.resolve() != target.resolve(): shutil.copyfile(source, target)
    manifest = dict(status="running", prepared_utc=full.now(), decoding_manifest=str(path), decoding_manifest_sha256=full.sha(path),
        conditions=run["conditions"], versions=list(full.VERSIONS), seeds=[1,2,3,4,5], steps=STEPS.tolist(), profiles=90,
        test_repetitions=[1,9,14,15], episodes_per_profile_per_seed=4,
        probability_method="float32 saved post-observation h_t -> Dense_0 -> ReLU -> Dense_1 -> exact frozen phase/initial-allocation mask -> softmax",
        reward_method="recorded reward on transition t following action at saved h_t; not cumulative",
        averaging="4 episodes/profile/seed, then 5 seeds; categories average profiles equally within seed before averaging seeds",
        uncertainty="Reward 95% percentile bootstrap over 5 seeds; 10000 draws, RNG30000", temporal_smoothing=False,
        policy_retraining=False, probe_retraining=False, new_rollouts=False,
        source_sha256={name:full.sha(out/"source"/name) for name in source_names})
    full.write(out / "analysis_run.json", manifest)
    full.verify(full.read(run["input_manifest"]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(extract_policy, [(record, str(out)) for record in run["policies"]]))
    # Six original-network checks cover every protocol × condition and source variant.
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", XLA_FLAGS="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    def check_reference(index):
        target = out / f"original_actor_reference_{index:02d}.json"
        if not target.exists():
            allowed = sorted(os.sched_getaffinity(0))
            core = allowed[(index//5) % len(allowed)]
            subprocess.run(["taskset", "-c", str(core), full.EVAL_PYTHON, str(out/"source"/Path(__file__).name), "reference", str(path), "--index", str(index), "--out", str(target)], env=env, check=True)
        full.require(full.read(target)["status"] == "passed", "Frozen actor reference failed")
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(check_reference, (0,5,10,15,20,25)))
    profiles, tables, audits = summarize(run, out)
    plot(run, out, profiles, tables)
    reference_audits = [full.read(out / f"original_actor_reference_{i:02d}.json") for i in (0,5,10,15,20,25)]
    full.require(len(list((out/"per_profile").glob("*.png"))) == 90, "Missing profile plot")
    for name, digest in manifest["source_sha256"].items():
        full.require(full.sha(out/"source"/name) == digest, "Analysis source changed")
    full.require(full.sha(path) == manifest["decoding_manifest_sha256"], "Decoding manifest changed")
    full.write(out / "completion_audit.json", dict(status="passed", completed_utc=full.now(), policies=30, profiles=90,
        test_episodes=10800, episode_steps=540000, profile_pngs=90, category_pngs=2,
        original_actor_reference=reference_audits, reconstructed_probabilities=True, original_recorded_rewards=True,
        all_probabilities_normalized=True, illegal_actions_zero=True, all_episodes_match_decoding_test_split=True,
        no_terminal_padding=True, policy_retraining=False, probe_retraining=False, new_rollouts=False))
    manifest.update(status="completed", completed_utc=full.now()); full.write(out/"analysis_run.json", manifest)
    (out / "README.md").write_text(
        "# Action probabilities and recorded step rewards, first 50 steps\n\n"
        "Evaluation rewards, post-observation recurrent states h_t, is_t0, and executed allocation actions were saved. Full policy action probability vectors and sampled joint movement/allocation actions were not in the original HDF5 logs. These plots reconstruct probabilities from the final frozen policy actor head and the actual saved h_t; rewards are read directly from the original evaluation logs. No RL or probe fitting and no new environment trajectories. Reconstructed probabilities may differ from the original unsaved GPU computation at floating-point rounding precision. Six original frozen Flax network checks cover all three protocols and both conditions.\n\n"
        "The plotted 50 decisions are t=0..49. Each distribution is conditional on the observation at t (saved post-observation h_t); its paired reward is the original reward from executing the corresponding action at t. Rewards are per-step, not cumulative returns.\n\n"
        "Exactly the latest decoding cohort: v1_matched/v2/v3, multi-partner and no-influence RNNs, seeds 1–5, all 90 unequal ordered delay pairs. Four held-out repetition IDs 1,9,14,15 (zero-based) per profile/seed. Single-partner policies are absent because they are absent from that decoding study. Every selected episode remains active throughout the first50 steps.\n\n"
        "Histogram: one stacked bar per timestep, 15 discrete joint action bins; action ID=3*movement+allocation. R/B request the EGO's red/blue role; when influence applies the partner takes the complementary goal. NONE is a reserved code in v2/v3 and has zero probability. Movement directions are absolute grid directions. These are averaged policy probabilities, not a histogram of sampled action counts.\n\n"
        "Masks: v1 at round t0 allows only Stay/R and Stay/B, and later only movement/None. v2 at t0 forces the supplied random initial assignment; v3 influence permits both Stay/R and Stay/B, whereas v3 no-influence forces the supplied assignment. Later v2/v3 allow movement×R/B. Supplied t0 allocations are recovered from their recorded executed allocation, which must equal the forced default. Original frozen network masks were checked independently.\n\n"
        "Average probabilities and rewards over the four episodes within each profile and seed, then average five policy seeds. Category averages give each profile equal weight within each seed before averaging seeds. Categories contain 24 green, 18 blue, 42 yellow, and 6 pink profiles. There is no smoothing or averaging across time. Reward bands are 95% percentile bootstrap intervals over five seed means (10,000 draws, RNG30000). Histogram uncertainty is not displayed.\n\n"
        "Files: per_profile/ contains90 individual PNGs, each showing all three experiments and both conditions. action_probabilities_rewards_all_profiles.pdf has the same90 pages. The category_averages_multi_partner/no_influence PNG/PDF files compare all four categories across experiments. profile_summary.csv.gz and category_summary.csv.gz contain all15 joint probabilities, movement/allocation marginals, rewards and intervals. Per-seed NPZ summaries preserve seeds; policies/ preserves episode-level reconstructed probabilities and original rewards. action_key.csv maps every action ID. completion_audit.json records checks.\n")
    print("[completed]", out, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("run", "reference")); p.add_argument("path", type=Path)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--index", type=int); p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    if args.command == "reference": reference_check(args.path, args.index, args.out)
    else: main(args.path, args.out, args.workers)
