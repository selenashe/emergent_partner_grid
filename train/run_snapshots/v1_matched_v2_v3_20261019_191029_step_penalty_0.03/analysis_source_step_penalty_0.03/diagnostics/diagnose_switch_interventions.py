"""Small paired closed-loop evaluation of frozen policies with switching held.

Diagnostic intervention, not retraining or a replacement experimental condition.
Uses exact frozen environment/network code in a separate process per version.
Every next observation and recurrent state reflects the action actually applied.
"""

import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "v2": ROOT / "train/run_snapshots/counterbalanced1096_20261002_235609/v2_balanced_training/source",
    "v3": ROOT / "train/run_snapshots/counterbalanced1096_20261008_035848_v3/v3_balanced_training/source",
}
POLICIES = {
    "v2": (("rnn_diverse_influence", 4), ("rnn_diverse_influence", 1),
           ("rnn_single_influence", 8), ("rnn_single_influence", 1)),
    "v3": (("rnn_diverse_influence", 4), ("rnn_diverse_influence", 1),
           ("rnn_single_influence", 4), ("rnn_single_influence", 6)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=tuple(SOURCES), required=True)
    parser.add_argument("--episodes-per-profile", type=int, default=2)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "train/seed_diagnostics/v1_matched_v2_v3")
    args = parser.parse_args()
    source = SOURCES[args.version]
    sys.path.insert(0, str(ROOT))
    from repo_paths import resolve_path, resolve_record_paths
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "baselines/IPPO"))
    import jax
    import jax.numpy as jnp
    import numpy as np
    trainer = importlib.import_module("ippo_rnn_coordination_grid")
    from jaxmarl.environments.coordination_grid import CoordinationGrid, TEST_CAPABILITY_PAIRS
    from jaxmarl.wrappers.baselines import load_params
    if Path(trainer.__file__).resolve() != (source / "baselines/IPPO/ippo_rnn_coordination_grid.py").resolve():
        raise ValueError("Wrong trainer loaded")
    metadata = json.loads((ROOT / "/juice6/u/jshe/emergent_partner_grid/train/training_curves/v1_matched_v2_v3_20261019_191029_step_penalty_0.03/summary.json").read_text())
    records = [next(r for r in metadata["runs"] if
        (r["version"], r["condition"], r["seed"]) == (args.version, condition, seed))
        for condition, seed in POLICIES[args.version]]
    config = resolve_record_paths(json.loads(resolve_path(records[0]["config"]).read_text()))
    capabilities = np.array([[1, 4]] + [list(c) for c in TEST_CAPABILITY_PAIRS], dtype=np.int32)
    kwargs = dict(config["ENV_KWARGS"])
    kwargs.update(layouts_dir=str(resolve_path(kwargs["layouts_dir"])), augment_symmetries=False,
                  partner_capability_pairs=capabilities.tolist(), influence=True)
    if args.version == "v3":
        kwargs["allocation_protocol"] = "online_v3"
    env = CoordinationGrid(**kwargs)
    network, initial_memory, _ = trainer.build_network(config, env.n_ego_actions)
    repetitions = args.episodes_per_profile
    batch = len(capabilities) * repetitions
    cap_per_episode = jnp.repeat(jnp.array(capabilities), repetitions, axis=0)
    key_reset, key_step = jax.random.split(jax.random.PRNGKey(args.eval_seed))
    obs, state = jax.vmap(env.reset)(jax.random.split(key_reset, batch))
    state = state.replace(capability=cap_per_episode)
    obs = jax.vmap(env.get_obs)(state)

    def run(params, hold):
        def step(carry, unused):
            obs, state, memory, prior_done, ended, key = carry
            inp = jax.tree_util.tree_map(lambda x: x[jnp.newaxis], obs["agent_0"])
            memory, pi, _ = network.apply(params, memory, (inp, prior_done[jnp.newaxis]))
            key, action_key, step_key = jax.random.split(key, 3)
            requested = pi.sample(seed=action_key).squeeze(0)
            # Permit initialization and the first movement-phase request, then
            # hold the effective assignment until the next round begins.
            applied = jnp.where(hold & (state.time >= 2),
                                3 * (requested // 3) + state.partner_assignment, requested)
            obs2, state2, reward, done, info = jax.vmap(
                env.step_env, in_axes=(0, 0, {"agent_0": 0}))(
                    jax.random.split(step_key, batch), state, {"agent_0": applied})
            valid = ~ended
            stats = (reward["agent_0"] * valid, info["success"] * valid,
                     info["round_done"] * valid, info["assignment_changed"] * valid,
                     valid, (applied != requested) * valid)
            return (obs2, state2, memory, done["__all__"], ended | done["__all__"], key), stats
        final, stats = jax.lax.scan(step, (obs, state, initial_memory(batch),
            jnp.zeros(batch, bool), jnp.zeros(batch, bool), key_step),
            None, length=env.rounds_per_episode * env.max_steps)
        return final[4], jax.tree_util.tree_map(lambda x: x.sum(axis=0), stats)

    run = jax.jit(run)
    rows, episodes, provenance = [], [], []
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for record in records:
        cfg = resolve_record_paths(json.loads(resolve_path(record["config"]).read_text()))
        checkpoint = resolve_path(cfg["SAVE_PARAMS_PATH"])
        params = load_params(checkpoint)
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(params)):
            raise ValueError("Nonfinite checkpoint")
        results = {}
        for name, hold in (("baseline", False), ("hold_after_first_request", True)):
            ended, stats = run(params, jnp.bool_(hold))
            ended, stats = np.asarray(ended), [np.asarray(x) for x in stats]
            if not ended.all() or not (stats[2] == 20).all():
                raise ValueError("Incomplete intervention episode")
            results[name] = stats
            for i in range(batch):
                episodes.append(dict(version=args.version, condition=record["condition"], seed=record["seed"],
                    intervention=name, profile=i // repetitions, repetition=i % repetitions,
                    red_delay=int(capabilities[i//repetitions, 0]), blue_delay=int(capabilities[i//repetitions, 1]),
                    episode_return=float(stats[0][i]), successes=int(stats[1][i]), rounds=int(stats[2][i]),
                    switches=int(stats[3][i]), steps=int(stats[4][i]), overridden_actions=int(stats[5][i])))
        for population, sl in (("actual_training_profile", slice(0, repetitions)), ("novel", slice(repetitions, batch))):
            base, held = results["baseline"], results["hold_after_first_request"]
            for name, stats in results.items():
                rows.append(dict(version=args.version, condition=record["condition"], seed=record["seed"],
                    population=population, intervention=name, n_episodes=len(stats[0][sl]),
                    episode_return=float(stats[0][sl].mean()), success=float(stats[1][sl].mean()/20),
                    switches_per_round=float(stats[3][sl].mean()/20), mean_episode_steps=float(stats[4][sl].mean()),
                    overridden_actions=float(stats[5][sl].mean()),
                    paired_return_difference_from_baseline=float((stats[0][sl]-base[0][sl]).mean())))
        provenance.append(dict(condition=record["condition"], seed=record["seed"], checkpoint=str(checkpoint),
                               checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
        # Commit each completed policy's evidence even if a later run fails.
        for name, data in (("summary", rows), ("episodes", episodes)):
            with (out / f"switch_intervention_{args.version}_{name}.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
        print(json.dumps(rows[-4:]), flush=True)
    (out / f"switch_intervention_{args.version}_provenance.json").write_text(json.dumps(dict(
        version=args.version, environment_source=str(source / "jaxmarl/environments/coordination_grid/coordination_grid.py"),
        trainer_source=str(trainer.__file__), eval_seed=args.eval_seed, episodes_per_profile=repetitions,
        rounds_per_episode=20, profiles=capabilities.tolist(), policies=provenance,
        caveat="Exploratory final-policy intervention on two paired episodes per profile; no training-cause inference or independent learner replications.",
        intervention="Keep sampled movement; allow t0 and first movement request, hold subsequent effective allocation; recompute observations/memory every step."), indent=2)+"\n")


if __name__ == "__main__":
    main()
