"""Audit the return oracle against exhaustive BFS and frozen JAX environments."""

import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_environment(path, name):
    module_name = "jaxmarl.environments.coordination_grid.oracle_validation_" + name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", choices=("train", "test"), default="train")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    from eval.optimal_training_return import geometry, inputs, round_return, shortest_round, transition, frozen_environment_sources
    directory = "training_return_ceiling" if args.population == "train" else "heldout_return_ceiling"
    output_root = ROOT / "eval" / directory / "v1_matched_v2_v3"
    out = args.out or output_root / "validation.json"
    paths, capabilities, _, sources = inputs(args.population)
    profiles = ((1, 4), (3, 9), (7, 2)) if args.population == "train" else ((0, 5), (3, 6), (9, 0))
    indices = [0, 1, 35, 200, 548, 1095]
    # Include a case that cannot succeed within the horizon under a fixed goal.
    per_case = output_root / "per_layout_profile.csv"
    if per_case.exists():
        import csv
        with per_case.open() as file:
            for row in csv.DictReader(file):
                if (int(row["d_R"]), int(row["d_B"])) in profiles and (
                    int(row["fixed_partner_red_steps"]) > 100 or int(row["fixed_partner_blue_steps"]) > 100
                ):
                    index = int(row["layout_index"])
                    if index not in indices:
                        indices.append(index)
                    break
    selected_paths = [paths[i] for i in indices]
    geometries = [geometry(p) for p in selected_paths]
    rng = np.random.default_rng(20261012)
    bfs_checks, trajectory_checks, transition_checks, timeout_checks = 0, 0, 0, 0
    variants = []
    variant_sources = frozen_environment_sources()
    for version in ("v1_matched", "v2", "v3"):
        for influence in (True, False):
            module = load_environment(variant_sources[(version,influence)], version + str(influence))
            kwargs = dict(layout_paths=[str(p) for p in selected_paths], influence=influence,
                          rounds_per_episode=1, max_steps=100, step_penalty=.01,
                          success_reward=1., augment_symmetries=False,
                          partner_capability_pairs=capabilities)
            if version == "v3":
                kwargs["allocation_protocol"] = "online_" + version
            env = module.CoordinationGrid(**kwargs)
            variants.append((version, influence, env))
    # Test A* optimality against a search with zero heuristic on the same graph.
    for _, moves, pn, distances, goals, starts in geometries:
        for profile in profiles:
            for initial_goal in (0, 1):
                for online in (False, True):
                    args_search = (moves, pn, distances, goals, *starts,
                                   np.array(profile), initial_goal, online)
                    a = shortest_round(*args_search, heuristic=True)[0]
                    b = shortest_round(*args_search, heuristic=False)[0]
                    if a != b:
                        raise AssertionError(f"A*/BFS mismatch: {a} != {b}")
                    bfs_checks += 1
    print(f"A* equals exhaustive BFS on {bfs_checks} layout/profile/assignment cases", flush=True)

    for version, influence, env in variants:
        online = version != "v1_matched" and influence
        # Compare arbitrary movement states in a vectorized call to frozen step_env.
        n_cases = 250
        local_layouts = rng.integers(0, len(geometries), n_cases)
        caps = np.array([profiles[i] for i in rng.integers(0, len(profiles), n_cases)], dtype=np.int32)
        keys = jax.random.split(jax.random.PRNGKey(42), n_cases)
        _, states = jax.vmap(env.reset_to_layout)(keys, jnp.array(local_layouts), jnp.array(caps))
        old_goals = rng.integers(0, 2, n_cases)
        counters = np.array([rng.integers(0, max(c) + 1) for c in caps])
        proposed_goals = rng.integers(0, 2, n_cases)
        move_actions = rng.integers(0, 5, n_cases)
        positions, expected = [], []
        for i in range(n_cases):
            _, moves, pn, distances, _, _ = geometries[local_layouts[i]]
            free = np.flatnonzero(np.all(distances >= 0, axis=0))
            ego, partner = rng.choice(free, 2, replace=False)
            positions.append([[ego % env.width, ego // env.width],
                              [partner % env.width, partner // env.width]])
            expected.append(transition(ego, partner, old_goals[i], counters[i], move_actions[i],
                                       proposed_goals[i], caps[i].astype(np.int64), moves, pn, online))
        states = states.replace(agent_pos=jnp.array(positions, dtype=jnp.int32),
                                time=jnp.ones(n_cases, dtype=jnp.int32),
                                partner_goal=jnp.array(old_goals + 1, dtype=jnp.int32),
                                partner_move_ctr=jnp.array(counters, dtype=jnp.int32),
                                partner_assignment=jnp.array(2-old_goals, dtype=jnp.int32),
                                last_ego_allocation=jnp.array(2-old_goals, dtype=jnp.int32))
        alloc = 2-proposed_goals if version != "v1_matched" else np.zeros(n_cases, dtype=np.int32)
        actions = jnp.array(3 * move_actions + alloc, dtype=jnp.int32)
        _, new, _, _, _ = jax.jit(jax.vmap(lambda k, s, a: env.step_env(k, s, {"agent_0": a})))(keys, states, actions)
        positions_new = np.asarray(new.agent_pos)
        actual = np.column_stack((positions_new[:, 0, 1] * env.width + positions_new[:, 0, 0],
                                  positions_new[:, 1, 1] * env.width + positions_new[:, 1, 0],
                                  np.asarray(new.partner_goal)-1, np.asarray(new.partner_move_ctr)))
        np.testing.assert_array_equal(actual, np.array(expected))
        transition_checks += n_cases

        # Replay optimal action sequences through the actual frozen environment.
        def replay(state, actions, assignment_key):
            def step(carry, action):
                def active(s):
                    _, next_state, rewards, _, info = env.step_env(assignment_key, s,
                                                                 {"agent_0": action})
                    return next_state, (rewards["agent_0"], info["success"])
                return jax.lax.cond(action >= 0, active,
                                    lambda s: (s, (jnp.float32(0), jnp.bool_(False))), carry)
            return jax.lax.scan(step, state, actions)
        replay_jit = jax.jit(replay)
        for li, (_, moves, pn, distances, goals, starts) in enumerate(geometries):
            for profile in profiles:
                for initial_goal in (0, 1):
                    steps, actions, _ = shortest_round(moves, pn, distances, goals, *starts,
                                                       np.array(profile), initial_goal, online, trace=True)
                    if steps > 100:
                        timeout_checks += 1
                        continue
                    _, state = env.reset_to_layout(jax.random.PRNGKey(42), jnp.int32(li),
                                                   jnp.array(profile, dtype=jnp.int32))
                    assignment_key = jax.random.PRNGKey(0)
                    if version == "v1_matched":
                        if not influence:
                            for key_seed in range(128):
                                candidate = jax.random.PRNGKey(key_seed)
                                assigned = env.step_env(candidate, state, {"agent_0": actions[0]})[1]
                                if int(assigned.partner_assignment) == 2-initial_goal:
                                    assignment_key = candidate
                                    break
                            else:
                                raise AssertionError("Could not exercise both initial assignments")
                    else:
                        state = state.replace(partner_goal=jnp.int32(initial_goal+1),
                                              partner_assignment=jnp.int32(2-initial_goal),
                                              last_ego_allocation=jnp.int32(2-initial_goal))
                    padded = np.full(100, -1, dtype=np.int32)
                    padded[:len(actions)] = actions
                    final_state, (rewards, successes) = replay_jit(state, jnp.array(padded), assignment_key)
                    success_indices = np.flatnonzero(np.asarray(successes))
                    if success_indices.tolist() != [steps-1] or int(final_state.time) != steps:
                        raise AssertionError(f"Incorrect completion: {version}, {influence}, {li}, {profile}, {initial_goal}")
                    if not np.isclose(np.asarray(rewards).sum(), round_return(steps), atol=2e-6):
                        raise AssertionError("Incorrect return/reward timing")
                    trajectory_checks += 1
        print(f"Frozen {version}, influence={influence}: transitions and optimal trajectories passed", flush=True)
    result = {
        "population": args.population,
        "passed": True, "layout_indices": list(indices), "profiles": [list(c) for c in profiles],
        "astar_matches_exhaustive_bfs": bfs_checks,
        "frozen_environment_transition_checks": transition_checks,
        "frozen_environment_optimal_trajectory_checks": trajectory_checks,
        "timeout_cases_without_successful_trajectory": timeout_checks,
        "versions": ["v1_matched", "v2", "v3"], "both_influence_settings": True,
        "reward_tolerance": 2e-6,
        "environment_variants": {
            v + ('_influence' if influence else '_noinfluence'): {
                'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            for (v,influence),p in variant_sources.items()},
        "validator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
