"""Full-information return ceilings under the frozen training task dynamics.

An exact shortest-path search controls only ego. The partner remains the
executed deterministic BFS navigator, including movement delays, collision
blocking, swap blocking, and destination-delay loading on reassignment.
Capabilities are supplied to the oracle: these are task ceilings, not a claim
about the optimum of a policy with hidden capabilities or a given architecture.
"""

import argparse
from collections import deque
import csv
import hashlib
import heapq
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_path, resolve_record_paths

DELTAS = ((0, -1), (0, 1), (1, 0), (-1, 0), (0, 0))


def geometry(path):
    raw = json.loads(Path(path).read_text())
    grid = np.asarray(raw["grid"])
    height, width = grid.shape
    n = grid.size
    moves = np.empty((n, 5), dtype=np.int64)
    for pos in range(n):
        y, x = divmod(pos, width)
        for action, (dx, dy) in enumerate(DELTAS):
            nx, ny = x + dx, y + dy
            moves[pos, action] = (ny * width + nx if 0 <= nx < width and 0 <= ny < height
                                  and grid[ny, nx] != 1 else pos)
    goals = np.array([raw[name][0] * width + raw[name][1] for name in ("red_goal", "blue_goal")])
    distances = np.full((2, n), -1, dtype=np.int64)
    partner_next = np.full((2, n), 4, dtype=np.int64)
    for goal, goal_pos in enumerate(goals):
        distances[goal, goal_pos] = 0
        queue = deque([int(goal_pos)])
        while queue:
            pos = queue.popleft()
            for nxt in moves[pos, :4]:
                if distances[goal, nxt] < 0:
                    distances[goal, nxt] = distances[goal, pos] + 1
                    queue.append(int(nxt))
        for pos in range(n):
            if grid.flat[pos] == 1 or distances[goal, pos] <= 0:
                continue
            for action in range(4):
                if distances[goal, moves[pos, action]] == distances[goal, pos] - 1:
                    partner_next[goal, pos] = action
                    break
    starts = [raw[name][0] * width + raw[name][1] for name in ("ego_start", "partner_start")]
    return raw, moves, partner_next, distances, goals, starts


@njit(cache=True)
def transition(ego, partner, goal, counter, move, requested_goal, delays, moves, partner_next, online):
    next_goal = requested_goal if online else goal
    ctr = delays[next_goal] if online and next_goal != goal else counter
    next_partner = moves[partner, partner_next[next_goal, partner]] if ctr == 0 else partner
    next_counter = delays[next_goal] if ctr == 0 else ctr - 1
    next_ego = moves[ego, move]
    if next_ego == next_partner or (next_ego == partner and next_partner == ego):
        next_ego, next_partner = ego, partner
    return next_ego, next_partner, next_goal, next_counter


@njit(cache=True)
def lower_bound(ego, partner, goal, counter, delays, distances, online):
    # Optimistically give an online partner its fastest delay for every move,
    # ignore collisions and goal-direction restrictions. Fixed-goal navigation
    # retains its exact cadence. Both relaxations underestimate remaining time.
    delay = min(delays[0], delays[1]) if online else delays[goal]
    first_wait = min(counter, delay) if online else counter
    result = 1000000
    for partner_goal in range(2):
        pd = distances[partner_goal, partner]
        ed = distances[1 - partner_goal, ego]
        if pd < 0 or ed < 0:
            continue
        pt = first_wait + 1 + (pd - 1) * (delay + 1) if pd > 0 else 0
        result = min(result, max(ed, pt))
    return result


@njit(cache=True)
def shortest_round(moves, partner_next, distances, goals, ego_start, partner_start,
                   delays, initial_goal, online, trace=False, heuristic=True, horizon=100):
    """A* with reopening; setting heuristic=False gives exhaustive BFS ordering.

    State contains both positions, current partner goal and delay counter.
    The first (stationary initialization) transition is counted explicitly.
    A missing successful path returns horizon+1 and has timeout reward -1.
    """
    n = moves.shape[0]
    n_counter = max(delays[0], delays[1]) + 1
    n_states = n * n * 2 * n_counter
    costs = np.full(n_states, horizon + 1, dtype=np.int16)
    parents = np.full(n_states if trace else 0, -1, dtype=np.int64)
    parent_actions = np.full(n_states if trace else 0, -1, dtype=np.int64)
    start = ((ego_start * n + partner_start) * 2 + initial_goal) * n_counter
    costs[start] = 1
    h = lower_bound(ego_start, partner_start, initial_goal, 0, delays, distances, online) if heuristic else 0
    queue = [(int(1 + h), int(1), int(start))]
    popped = 0
    while queue:
        priority, cost, state_id = heapq.heappop(queue)
        if cost != costs[state_id]:
            continue
        if priority > horizon:
            break
        popped += 1
        counter = state_id % n_counter
        encoded = state_id // n_counter
        goal = encoded % 2
        encoded //= 2
        partner = encoded % n
        ego = encoded // n
        success = ((ego == goals[0] and partner == goals[1]) or
                   (ego == goals[1] and partner == goals[0]))
        if success:
            actions = np.empty(cost if trace else 0, dtype=np.int64)
            if trace:
                actions[0] = 12 + (2 - initial_goal)
                current = state_id
                for idx in range(cost - 1, 0, -1):
                    actions[idx] = parent_actions[current]
                    current = parents[current]
            return cost, actions, popped
        if cost >= horizon:
            continue
        for choice in range(2 if online else 1):
            requested_goal = choice if online else goal
            for move in range(5):
                ne, npartner, ng, nc = transition(ego, partner, goal, counter, move,
                                                requested_goal, delays, moves, partner_next, online)
                next_id = ((ne * n + npartner) * 2 + ng) * n_counter + nc
                next_cost = cost + 1
                if next_cost >= costs[next_id]:
                    continue
                nh = lower_bound(ne, npartner, ng, nc, delays, distances, online) if heuristic else 0
                if next_cost + nh > horizon:
                    continue
                costs[next_id] = next_cost
                if trace:
                    parents[next_id] = state_id
                    parent_actions[next_id] = 3 * move + ((2 - ng) if online else 0)
                heapq.heappush(queue, (int(next_cost + nh), int(next_cost), int(next_id)))
    return horizon + 1, np.empty(0, dtype=np.int64), popped


def round_return(steps):
    return 1.0 - .03 * (steps - 1) if steps <= 100 else -3.0


def inputs(population="train"):
    manifests = {
        "v1_matched": ROOT / "train/manifests/sbatch_v1_matched_20261009_182845.json",
        "v2": ROOT / "train/manifests/sbatch_counterbalanced1096_20261002_235609.json",
        "v3": ROOT / "train/manifests/sbatch_counterbalanced1096_20261008_035848_v3.json",
    }
    records = {v: resolve_record_paths(json.loads(p.read_text())) for v, p in manifests.items()}
    config_paths = {v: Path(r["versions"][v]["checkpoint_root"]) / "rnn_diverse_influence_seed1_config.json"
                    for v, r in records.items()}
    configs = {v: resolve_record_paths(json.loads(p.read_text())) for v, p in config_paths.items()}
    paths = sorted(Path(configs["v1_matched"]["ENV_KWARGS"]["layouts_dir"]).glob("*.json"))
    if len(paths) != 1096:
        raise ValueError("Expected the frozen 1,096-layout training corpus")
    for version, config in configs.items():
        kwargs = config["ENV_KWARGS"]
        if kwargs["step_penalty"] != .03 or kwargs["success_reward"] != 1 or kwargs["max_steps"] != 100:
            raise ValueError(f"Unexpected reward or horizon: {version}")
        if kwargs["rounds_per_episode"] != 20 or kwargs["augment_symmetries"]:
            raise ValueError(f"Unexpected episode/layout setting: {version}")
        if sorted(Path(kwargs["layouts_dir"]).glob("*.json")) != paths:
            raise ValueError(f"Different layout corpus: {version}")
    capabilities = configs["v1_matched"]["ENV_KWARGS"]["partner_capability_pairs"]
    if any(c["ENV_KWARGS"]["partner_capability_pairs"] != capabilities for c in configs.values()):
        raise ValueError("Different capability populations")
    sources = {v: Path(r["versions"][v]["frozen_source_root"]) /
               "jaxmarl/environments/coordination_grid/coordination_grid.py" for v, r in records.items()}
    if population == "test":
        populations = {v: [list(c) for c in runpy.run_path(str(p.with_name("capability_populations.py")))["TEST_CAPABILITY_PAIRS"]]
                       for v, p in sources.items()}
        capabilities = populations["v1_matched"]
        if len(capabilities) != 22 or any(c != capabilities for c in populations.values()):
            raise ValueError("Expected the same 22 held-out profiles in every frozen version")
    elif population != "train":
        raise ValueError(f"Unknown population: {population}")
    return paths, capabilities, config_paths, sources


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, population="train"):
    summaries = []
    for version, field in (("v1_matched", "fixed_best_return"),
                           ("v2", "online_random_initial_return"),
                           ("v3", "online_chosen_initial_return")):
        for condition in ("mlp_diverse_influence", "rnn_diverse_influence",
                          "rnn_single_influence", "rnn_diverse_noinfluence"):
            relevant = ([r for r in rows if (r["d_R"], r["d_B"]) == (1, 4)]
                        if population == "train" and condition == "rnn_single_influence" else rows)
            target = "fixed_random_assignment_return" if condition == "rnn_diverse_noinfluence" else field
            values = np.array([r[target] for r in relevant])
            summaries.append({"version": version, "condition": condition,
                              "n_layout_profile_pairs": len(relevant),
                              "oracle_mean_round_return": float(values.mean()),
                              "oracle_mean_episode_return": float(20 * values.mean()),
                              "oracle_min_round_return": float(values.min()),
                              "oracle_max_round_return": float(values.max())})
    return summaries


def initial_information_bounds(rows):
    """Choose initialization before seeing capability; reveal it for navigation.

    V1_matched's feedforward initialization has identical observations for every
    profile on the same layout. Maximizing expected fixed-assignment return
    therefore bounds its return even with an oracle navigator thereafter.
    V3 has the same initial ambiguity but permits subsequent reassignment.
    """
    grouped = {}
    for row in rows:
        grouped.setdefault(row["layout_index"], []).append(row)
    results = []
    for index, group in grouped.items():
        fixed = [np.mean([round_return(r[f"fixed_partner_{g}_steps"]) for r in group])
                 for g in ("red", "blue")]
        online = [np.mean([round_return(r[f"online_initial_partner_{g}_steps"]) for r in group])
                  for g in ("red", "blue")]
        results.append({"layout_index": index, "layout_id": group[0]["layout_id"],
                        "fixed_blind_initial_upper_round_return": float(max(fixed)),
                        "online_blind_initial_upper_round_return": float(max(online))})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", choices=("train", "test"), default="train")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--limit-layouts", type=int, help="Development subset; never call this the full-corpus result")
    args = parser.parse_args()
    paths, capabilities, config_paths, sources = inputs(args.population)
    if args.limit_layouts:
        paths = paths[:args.limit_layouts]
    directory = "training_return_ceiling" if args.population == "train" else "heldout_return_ceiling"
    out_dir = (args.out_dir or Path('/juice6/u/jshe/emergent_partner_grid') / "eval" / directory / 'v1_matched_v2_v3_20261019_191029_step_penalty_0.03').resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, node_count = [], 0
    started = time.monotonic()
    for layout_idx, path in enumerate(paths):
        raw, moves, partner_next, distances, goals, starts = geometry(path)
        # A fixed partner goal depends only on that goal's delay; reuse across profiles.
        fixed = {}
        for goal in range(2):
            for delay in sorted({c[goal] for c in capabilities}):
                d = np.array([delay, delay], dtype=np.int64)
                steps, _, nodes = shortest_round(moves, partner_next, distances, goals,
                                                 *starts, d, goal, False)
                fixed[(goal, delay)] = int(steps)
                node_count += nodes
        for dr, db in capabilities:
            delays = np.array([dr, db], dtype=np.int64)
            times = []
            for goal in range(2):
                steps, _, nodes = shortest_round(moves, partner_next, distances, goals,
                                                 *starts, delays, goal, True)
                times.append(int(steps))
                node_count += nodes
            fixed_red, fixed_blue = fixed[(0, dr)], fixed[(1, db)]
            rows.append({
                "layout_index": layout_idx, "layout_id": raw["layout_id"], "d_R": dr, "d_B": db,
                "fixed_partner_red_steps": fixed_red, "fixed_partner_blue_steps": fixed_blue,
                "online_initial_partner_red_steps": times[0], "online_initial_partner_blue_steps": times[1],
                "fixed_best_return": max(round_return(fixed_red), round_return(fixed_blue)),
                "fixed_random_assignment_return": .5 * (round_return(fixed_red) + round_return(fixed_blue)),
                "online_random_initial_return": .5 * (round_return(times[0]) + round_return(times[1])),
                "online_chosen_initial_return": max(round_return(times[0]), round_return(times[1])),
            })
        if (layout_idx + 1) % 32 == 0 or layout_idx == 0 or layout_idx + 1 == len(paths):
            print(f"Solved {layout_idx+1}/{len(paths)} layouts; {len(rows)} profile/layout pairs; "
                  f"{time.monotonic()-started:.1f}s; {node_count} searched states", flush=True)
    summaries = summarize(rows, args.population)
    information_bounds = initial_information_bounds(rows)
    write_csv(out_dir / "per_layout_profile.csv", rows)
    write_csv(out_dir / "ceiling_summary.csv", summaries)
    write_csv(out_dir / "initial_information_bounds.csv", information_bounds)
    report = {
        "population": args.population,
        "capability_profiles": capabilities,
        "method": "Exact A* shortest successful completion with an admissible distance/cadence relaxation and state reopening; only ego actions are optimized.",
        "information": "Oracle knows the true partner capabilities at episode start. This is a physical task ceiling, not an architecture-constrained or partially observed policy optimum.",
        "averaging": ("Uniform weight across all training layouts and the condition's training profiles. "
                      if args.population == "train" else
                      "Uniform weight across the same familiar layouts and all 22 held-out profiles, including for the single-partner-trained policy. ")
                     + "No-influence averages both fixed partner goals equally; v2 influence averages both random initial goals equally. Multiply mean round return by 20.",
        "episode_return_formula": "Sum over 20 rounds of 1 - 0.03*(optimal_round_steps-1), or -3 for a timeout.",
        "absolute_reward_bound": 20, "layouts": len(paths), "profile_layout_pairs": len(rows),
        "complete_corpus": len(paths) == 1096, "searched_states": node_count,
        "elapsed_seconds": time.monotonic() - started,
        "summaries": summaries,
        "mlp_blind_initialization_upper_bounds": {
            "v1_matched": float(20 * np.mean([r["fixed_blind_initial_upper_round_return"] for r in information_bounds])),
            "v3": float(20 * np.mean([r["online_blind_initial_upper_round_return"] for r in information_bounds])),
            "interpretation": f"For each layout choose the initial partner goal maximizing expected return across the {len(capabilities)} profiles, then allow fully informed optimal navigation. These bound MLP performance, without establishing an attainable feedforward policy optimum.",
        },
        "layout_records": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        "config_records": {v: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for v, p in config_paths.items()},
        "environment_sources": {v: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for v, p in sources.items()},
        "population_sources": {v: {"path": str(p.with_name("capability_populations.py")),
                                   "sha256": hashlib.sha256(p.with_name("capability_populations.py").read_bytes()).hexdigest()}
                               for v, p in sources.items()},
        "solver_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "Exact ceilings apply to a fully informed oracle; a learned policy cannot directly observe capability.",
            "MLP and diverse RNN share the same task ceiling; this does not imply equal attainable performance under their information/memory constraints.",
            ("The final logged-update dot plot samples its own completed-episode mix; these uniform-corpus ceilings are not exact ceilings for that unrecorded sample mix."
             if args.population == "train" else
             "The held-out rollout files sample layout sequences; these uniform-corpus ceilings are population references, not exact ceilings for each recorded sequence."),
            "V1_matched no-influence uses round parity; its uniform-corpus expectation matches an equal goal mixture, but a particular 20-layout episode must retain its actual round parity.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in summaries:
        print(f"{row['version']} {row['condition']}: {row['oracle_mean_episode_return']:.6f}")


if __name__ == "__main__":
    main()
