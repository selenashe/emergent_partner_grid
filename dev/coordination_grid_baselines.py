"""Non-communicating baselines run through the real ``CoordinationGrid``.

Each baseline is an ego-only policy: the partner is scripted inside the env.
This guarantees that collision rules, wall blocking, the t=0 stationary step,
the 50/50 partner-goal sample under NONE, the deterministic UP>DOWN>RIGHT>LEFT
tie-break for greedy navigation, and the step horizon are all identical to
what the training environment uses.

Public interface:

    run_baselines(envs, trials, seed, *, max_steps=15, wait_k=2, partner_z=0.5)
        -> pandas.DataFrame indexed by layout_id with columns
        {name}_success, {name}_steps_to_success for each baseline.

    summarize_baselines(envs, trials, ..., seed)
        -> dict aggregate matching the shape ``analyze_grids_lite`` used
        to write.

An ``envs`` value must be a dict with at least ``"_path"`` set to the JSON's
absolute path (``_load_jsons`` in ``analyze_grids_lite`` populates that).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

from jaxmarl.environments.coordination_grid import (
    CoordinationGrid,
    encode_ego,
    Actions,
    Messages,
    GOAL_UNSET, GOAL_RED, GOAL_BLUE,
)

# Movement / message shortcuts (mirror the env's IntEnums).
UP, DOWN, RIGHT, LEFT, STAY = 0, 1, 2, 3, 4
NONE, M0, M1 = 0, 1, 2

# Matches the env's DIR_TO_VEC order for indices 0..3 (UP, DOWN, RIGHT, LEFT).
_MOVE_DELTAS = ((0, -1), (0, 1), (1, 0), (-1, 0))


# --------------------------------------------------------------------------- #
# Env cache + shortest-path helpers                                           #
# --------------------------------------------------------------------------- #

_ENV_CACHE: Dict[Tuple[str, int, float], CoordinationGrid] = {}


def _get_env(layout_path: str, max_steps: int, partner_z: float) -> CoordinationGrid:
    key = (layout_path, int(max_steps), float(partner_z))
    if key not in _ENV_CACHE:
        _ENV_CACHE[key] = CoordinationGrid(
            layout_path=layout_path,
            max_steps=int(max_steps),
            partner_z=float(partner_z),
        )
    return _ENV_CACHE[key]


def _bfs_distances(wall_np: np.ndarray, gx: int, gy: int) -> np.ndarray:
    """Grid-shaped int array of shortest-path lengths from every cell to
    (gx, gy). -1 for walls / unreachable cells.
    """
    h, w = wall_np.shape
    dist = -np.ones((h, w), dtype=np.int32)
    if not (0 <= gx < w and 0 <= gy < h) or wall_np[gy, gx]:
        return dist
    dist[gy, gx] = 0
    q = deque([(gx, gy)])
    while q:
        cx, cy = q.popleft()
        for dx, dy in _MOVE_DELTAS:
            nx, ny = cx + dx, cy + dy
            if (
                0 <= nx < w and 0 <= ny < h
                and not wall_np[ny, nx]
                and dist[ny, nx] < 0
            ):
                dist[ny, nx] = dist[cy, cx] + 1
                q.append((nx, ny))
    return dist


def _greedy_move_toward(env: CoordinationGrid, xy, goal_name: str) -> int:
    """Reuse the env's own precomputed BFS next-action tables so that the
    ego's tie-break is identical to the scripted partner's.
    """
    x, y = int(xy[0]), int(xy[1])
    if goal_name == "RED":
        return int(env.next_action_toward_red[0, y, x])
    if goal_name == "BLUE":
        return int(env.next_action_toward_blue[0, y, x])
    return STAY


# --------------------------------------------------------------------------- #
# Rollout skeleton                                                            #
# --------------------------------------------------------------------------- #

def _rollout(env: CoordinationGrid, policy: Callable, seed: int
             ) -> Tuple[bool, int]:
    """Run one episode. ``policy(state, obs) -> ego_action_int``.
    Returns (success, steps_used). On failure, steps_used == env.max_steps.
    """
    key = jax.random.PRNGKey(int(seed))
    obs, state = env.reset(key)
    success = False
    steps = 0
    for _ in range(env.max_steps):
        a = int(policy(state, obs))
        # Reuse the same rng key each step: env.step_env only uses it for the
        # partner-goal sample, which commits at t=1. Passing the same key across
        # steps is fine — no aliasing to worry about.
        obs, state, r, done, info = env.step_env(key, state, {"agent_0": a})
        steps += 1
        if bool(done["__all__"]):
            success = bool(info["success"])
            break
    return success, steps


# --------------------------------------------------------------------------- #
# Ego policies                                                                #
# --------------------------------------------------------------------------- #

class _WalkToPolicy:
    """Every step, take the greedy BFS-next-action toward a fixed goal."""

    def __init__(self, env: CoordinationGrid, goal_name: str):
        self.env = env
        self.goal_name = goal_name

    def __call__(self, state, obs) -> int:
        move = _greedy_move_toward(self.env, state.agent_pos[0], self.goal_name)
        return encode_ego(move, NONE)


class _NearestPolicy(_WalkToPolicy):
    """Walk toward whichever of RED/BLUE is BFS-closer to ego_start."""

    def __init__(self, env: CoordinationGrid, rng: random.Random):
        wall_np = np.array(env.wall_map)
        rx, ry = int(env.red_goal[0]), int(env.red_goal[1])
        bx, by = int(env.blue_goal[0]), int(env.blue_goal[1])
        d_red = _bfs_distances(wall_np, rx, ry)
        d_blue = _bfs_distances(wall_np, bx, by)
        ex, ey = int(env.ego_start[0]), int(env.ego_start[1])
        dr, db = int(d_red[ey, ex]), int(d_blue[ey, ex])
        if dr < db:
            goal_name = "RED"
        elif db < dr:
            goal_name = "BLUE"
        else:
            goal_name = "RED" if rng.random() < 0.5 else "BLUE"
        super().__init__(env, goal_name)


class _RandomGoalPolicy(_WalkToPolicy):
    """Pick a goal uniformly at policy-construction time."""

    def __init__(self, env: CoordinationGrid, rng: random.Random):
        goal_name = "RED" if rng.random() < 0.5 else "BLUE"
        super().__init__(env, goal_name)


class _WaitReactPolicy:
    """STAY for ``k`` env-steps observing the partner; then commit to the
    goal the partner is NOT moving toward, inferred by which of {RED,BLUE}
    the partner's BFS distance has decreased more toward.

    k >= 1 so we actually get a partner motion sample (env's t=0 is free,
    the partner also stays; useful information starts at t=1).
    """

    def __init__(self, env: CoordinationGrid, rng: random.Random, k: int = 2):
        self.env = env
        self.k = int(k)
        self.rng = rng
        wall_np = np.array(env.wall_map)
        self.d_red = _bfs_distances(
            wall_np, int(env.red_goal[0]), int(env.red_goal[1])
        )
        self.d_blue = _bfs_distances(
            wall_np, int(env.blue_goal[0]), int(env.blue_goal[1])
        )
        self.px0, self.py0 = (
            int(env.partner_start[0]), int(env.partner_start[1])
        )
        self.target: str | None = None

    def _infer_and_pick(self, state) -> str:
        px = int(state.agent_pos[1, 0]); py = int(state.agent_pos[1, 1])
        dr_delta = int(self.d_red[self.py0, self.px0]) - int(self.d_red[py, px])
        db_delta = int(self.d_blue[self.py0, self.px0]) - int(self.d_blue[py, px])
        if dr_delta > db_delta:
            inferred = "RED"
        elif db_delta > dr_delta:
            inferred = "BLUE"
        else:
            inferred = "RED" if self.rng.random() < 0.5 else "BLUE"
        return "BLUE" if inferred == "RED" else "RED"

    def __call__(self, state, obs) -> int:
        t = int(state.time)
        if t < self.k:
            return encode_ego(STAY, NONE)
        if self.target is None:
            self.target = self._infer_and_pick(state)
        move = _greedy_move_toward(self.env, state.agent_pos[0], self.target)
        return encode_ego(move, NONE)


class _OraclePolicy:
    """Wait until the partner has committed (state.partner_goal != UNSET),
    then walk to the complementary goal. Env commits at t=1, so at t=2 the
    goal is visible. Diagnostic upper bound (no message needed).
    """

    def __init__(self, env: CoordinationGrid):
        self.env = env
        self.target: str | None = None

    def __call__(self, state, obs) -> int:
        if self.target is None:
            pg = int(state.partner_goal)
            if pg == GOAL_RED:
                self.target = "BLUE"
            elif pg == GOAL_BLUE:
                self.target = "RED"
            else:
                return encode_ego(STAY, NONE)
        move = _greedy_move_toward(self.env, state.agent_pos[0], self.target)
        return encode_ego(move, NONE)


# --------------------------------------------------------------------------- #
# Baseline registry                                                           #
# --------------------------------------------------------------------------- #

def _make_baseline_fns(env: CoordinationGrid, wait_k: int):
    """Return {name: build_policy(rng) -> callable}. Wrapping in a builder
    lets stateful policies (wait_react) get a fresh state per trial.
    """
    return {
        "nearest":    lambda rng: _NearestPolicy(env, rng),
        "fixed_RB":   lambda rng: _WalkToPolicy(env, "RED"),
        "fixed_BR":   lambda rng: _WalkToPolicy(env, "BLUE"),
        "random":     lambda rng: _RandomGoalPolicy(env, rng),
        "wait_react": lambda rng: _WaitReactPolicy(env, rng, k=wait_k),
        "oracle":     lambda rng: _OraclePolicy(env),
    }


NON_COMM_BASELINES: List[str] = ["nearest", "wait_react", "random"]
ALL_BASELINES: List[str] = [
    "nearest", "fixed_RB", "fixed_BR", "wait_react", "random", "oracle",
]


# --------------------------------------------------------------------------- #
# Public runners                                                              #
# --------------------------------------------------------------------------- #

def _resolve_path(env_dict: dict) -> str:
    if "_path" not in env_dict:
        raise KeyError(
            "env_dict is missing '_path'. Load with the updated _load_jsons "
            "in analyze_grids_lite (which stores the source path)."
        )
    return env_dict["_path"]


def run_baselines(envs: Dict[str, dict],
                  trials: int = 100,
                  seed: int = 0,
                  *,
                  max_steps: int = 15,
                  wait_k: int = 2,
                  partner_z: float = 0.5,
                  baselines: List[str] | None = None,
                  ) -> pd.DataFrame:
    """Per-layout baseline results.

    Returns one row per layout with columns:
        <name>_success           fraction of trials that ended in success
        <name>_steps_to_success  mean episode length over SUCCESSFUL trials
                                 (NaN if a baseline never succeeded).
    """
    names = list(baselines) if baselines is not None else ALL_BASELINES
    master = random.Random(seed)
    rows: List[dict] = []
    for layout_id, env_dict in envs.items():
        env = _get_env(_resolve_path(env_dict), max_steps, partner_z)
        builders = _make_baseline_fns(env, wait_k)
        row: Dict[str, float] = {"layout_id": layout_id}
        for name in names:
            build = builders[name]
            rng = random.Random(master.randrange(2**31))
            n_succ = 0
            step_sum = 0
            for trial_i in range(trials):
                policy = build(rng)
                trial_seed = rng.randrange(2**31)
                succ, steps = _rollout(env, policy, trial_seed)
                if succ:
                    n_succ += 1
                    step_sum += steps
            row[f"{name}_success"] = n_succ / trials
            row[f"{name}_steps_to_success"] = (
                (step_sum / n_succ) if n_succ else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_baselines(envs: Dict[str, dict],
                        trials: int,
                        max_steps: int,
                        wait_k: int,
                        seed: int,
                        partner_z: float = 0.5,
                        ) -> Dict[str, dict]:
    """Aggregate over layouts. Shape matches the old analyze_grids_lite output."""
    df = run_baselines(
        envs, trials=trials, seed=seed,
        max_steps=max_steps, wait_k=wait_k, partner_z=partner_z,
        baselines=NON_COMM_BASELINES,
    )
    print(
        f"[3] Non-communicating baselines through CoordinationGrid "
        f"(mean over layouts; {trials} trials/layout, "
        f"max_steps={max_steps}, wait_k={wait_k}, partner_z={partner_z})"
    )
    print(
        f"    {'baseline':<12}  {'mean_success':>13}  "
        f"{'mean_cost':>10}  {'n_cost':>6}"
    )
    out: Dict[str, dict] = {
        "config": {
            "trials": trials, "max_steps": max_steps,
            "wait_k": wait_k, "seed": seed, "partner_z": partner_z,
        },
        "baselines": {},
    }
    for name in NON_COMM_BASELINES:
        succ_col = df[f"{name}_success"].to_numpy(dtype=float)
        cost_col = df[f"{name}_steps_to_success"].to_numpy(dtype=float)
        mean_succ = float(succ_col.mean())
        ok = ~np.isnan(cost_col)
        mean_cost = float(cost_col[ok].mean()) if ok.any() else float("nan")
        n_cost = int(ok.sum())
        print(
            f"    {name:<12}  {mean_succ:13.3f}  "
            f"{mean_cost:10.3f}  {n_cost:6d}"
        )
        out["baselines"][name] = {
            "mean_success": mean_succ,
            "mean_cost": mean_cost if not np.isnan(mean_cost) else None,
            "n_cost": n_cost,
        }
    return out
