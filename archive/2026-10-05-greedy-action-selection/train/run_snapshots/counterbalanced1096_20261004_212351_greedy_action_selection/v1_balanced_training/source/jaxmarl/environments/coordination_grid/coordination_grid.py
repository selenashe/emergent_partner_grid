"""
CoordinationGrid: 20-round 7×7 grid coordination task with a scripted
partner whose *latent capability profile* ``(d_R, d_B)`` controls
per-goal movement delay. Ego (agent_0) is the only learned agent;
partner (agent_1) is scripted.

Capability semantics (matches the Overcooked reference)
-------------------------------------------------------
Each partner has a pair of per-goal delays

    (d_R, d_B)   with d_X ∈ {0, 1, ...}

where d_X is the number of wait steps inserted after a partner move
while pursuing goal X. So

    d = 0  -> partner moves every step   (fastest)
    d = 1  -> partner moves every 2 steps
    d = k  -> partner moves every (k + 1) steps

The training and test capability populations live in
``capability_populations.py`` — the training pool is a specialist set
(fast on one task, slow on the other), and the test pool includes
scalar delay values never seen at training (0, 5, 6).

The capability is fixed across the 20-round partner episode, never
appears in the ego's observation, and only affects the partner. The
ego must infer ``(d_R, d_B)`` from partner movement timing and use it
to choose the better of two possible role allocations for this round.

Action encoding
---------------
Ego action is a joint (move, alloc) pair, flat-encoded as an int in [0, 15):
    a = 3 * move + alloc
    move  ∈ {0:UP, 1:DOWN, 2:RIGHT, 3:LEFT, 4:STAY}
    alloc ∈ {0:NONE, 1:ALLOC_RED, 2:ALLOC_BLUE}

Use ``encode_ego(move, alloc)`` / ``decode_ego(a)``.

Legal-mask summary
------------------
At t=0 the ego MUST commit to a role:
    legal = { STAY+ALLOC_RED, STAY+ALLOC_BLUE }   (ids 13, 14)
At t>=1 the ego just navigates:
    legal = { {UP,DOWN,RIGHT,LEFT,STAY} + NONE }  (ids 0, 3, 6, 9, 12)

Note: this ``alloc`` channel is NOT a partner-dependent symbolic message.
Its meaning is fixed and universal (RED / BLUE goal), so it functions as
a task-level role assignment, not as communication about who the partner
is.  The ego still has to *infer* the partner's capability from motion.

Allocation → partner goal
-------------------------
At t=1 the partner deterministically takes the *opposite* goal to the
ego's t=0 allocation:
    ego alloc = ALLOC_RED  → partner_goal = BLUE
    ego alloc = ALLOC_BLUE → partner_goal = RED
There is no other stochastic goal-commit rule.

Partner movement + delay
------------------------
Partner navigation still uses the precomputed BFS shortest-path tables
toward its committed goal.  On each round-local step t>=1:
    * if ``partner_move_ctr == 0`` and the partner has a committed goal,
      the partner takes one BFS step and ``partner_move_ctr`` resets to
      ``d``, where ``d = d_R`` if goal == RED else ``d_B``.
    * otherwise the partner STAYs and ``partner_move_ctr`` decrements
      (floored at 0).
So d=0 → partner moves every step (fastest); d=k → partner moves once
every k+1 steps. ``partner_move_ctr`` resets to 0 at every round
boundary; nothing else about partner navigation depends on capability.

Allocation influence
--------------------
When ``influence=True`` (default), the partner's assigned goal at t=1
is the complement of the ego's t=0 alloc — so the ego's allocation
choice controls the partner. When ``influence=False``, the partner's
goal at t=1 is instead determined by a partner-independent balanced
rule (round_idx parity), regardless of the ego's t=0 action; the ego
still selects an alloc from the legal t=0 set, but that alloc is
ignored for partner assignment. This is the no-influence control.

Reward and termination are unchanged:
    success (agents on distinct goals) → +1.0
    otherwise                          → -0.01
    round ends on success or max_steps
Twenty rounds per partner episode; done['__all__'] only fires on the
last round's terminal step; GRU carries across intermediate rounds.

Observation for each agent is a dict:
    {"grid": (H, W, 5), "last_allocation": (3,), "is_t0": scalar float32}
"""

import glob
import json
import os
from collections import deque
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import lax

from jaxmarl.environments import MultiAgentEnv


# Directional offsets in (dx, dy). Index into DIR_TO_VEC by move action id.
DIR_TO_VEC = jnp.array(
    [
        (0, -1),  # 0: UP    (NORTH)
        (0, 1),   # 1: DOWN  (SOUTH)
        (1, 0),   # 2: RIGHT (EAST)
        (-1, 0),  # 3: LEFT  (WEST)
        (0, 0),   # 4: STAY
    ],
    dtype=jnp.int32,
)


class Actions(IntEnum):
    up = 0
    down = 1
    right = 2
    left = 3
    stay = 4


class Allocations(IntEnum):
    """Ego's t=0 allocation channel.

    ``none`` is used at t>=1 (no allocation happens outside t=0).
    ``red`` / ``blue`` are the two legal t=0 commitments: the partner
    takes the *complementary* goal.
    """
    none = 0
    red = 1
    blue = 2


N_MOVES = 5
N_ALLOCATIONS = 3
N_EGO_ACTIONS = N_MOVES * N_ALLOCATIONS  # 15

# Backward-compat aliases (older callers still say "messages"; the flat
# encoding is unchanged).
N_MESSAGES = N_ALLOCATIONS
Messages = Allocations

# Legality of the flat ego actions per environment phase.
#   t=0: STAY + {ALLOC_RED, ALLOC_BLUE}   -> {13, 14}
#   t>=1: {UP,DOWN,RIGHT,LEFT,STAY} + NONE -> {0, 3, 6, 9, 12}
LEGAL_ACTION_IDS_T0 = tuple(
    3 * int(Actions.stay) + int(a) for a in (Allocations.red, Allocations.blue)
)
LEGAL_ACTION_IDS_TGEQ1 = tuple(
    3 * mv + int(Allocations.none) for mv in range(N_MOVES)
)
ACTION_MASK_T0 = np.zeros(N_EGO_ACTIONS, dtype=np.bool_)
ACTION_MASK_T0[list(LEGAL_ACTION_IDS_T0)] = True
ACTION_MASK_TGEQ1 = np.zeros(N_EGO_ACTIONS, dtype=np.bool_)
ACTION_MASK_TGEQ1[list(LEGAL_ACTION_IDS_TGEQ1)] = True

# ---------------------------------------------------------------------------
# Communication condition
# ---------------------------------------------------------------------------
# The only supported condition is ``action_only`` in the sense that the ego
# has no partner-dependent symbolic message channel.  The t=0 alloc action
# is a fixed-meaning role commitment (RED vs BLUE), not a token whose
# meaning depends on the partner's capability.
COMM_ACTION_ONLY = "action_only"
COMM_CONDITIONS = (COMM_ACTION_ONLY,)

# Partner-goal state encoding.
GOAL_UNSET = 0
GOAL_RED = 1
GOAL_BLUE = 2


# ---------------------------------------------------------------------------
# Capability profile constants — reference-style specialist populations.
# Authoritative source: capability_populations.py
# ---------------------------------------------------------------------------
from .capability_populations import (  # noqa: E402
    TRAIN_CAPABILITY_PAIRS,
    TEST_CAPABILITY_PAIRS,
    DEFAULT_SINGLE_PARTNER,
    FAST_DELAYS,
    SLOW_DELAYS,
)


def encode_ego(move, alloc) -> int:
    """(move, alloc) -> flat ego action id in [0, 15). Pure Python; JAX-safe."""
    return int(3) * int(move) + int(alloc)


def decode_ego(action):
    """Flat action -> (move, alloc). JAX-compatible; accepts scalar or array."""
    a = jnp.asarray(action, dtype=jnp.int32)
    move = a // 3
    alloc = a % 3
    return move, alloc


DEFAULT_LAYOUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "dev", "grids", "layouts", "train", "train_0000.json",
)


@struct.dataclass
class State:
    # Round-local geometry (sampled fresh at each round boundary within a
    # partner episode).
    agent_pos: chex.Array       # (2, 2) int32 -- [x, y] per agent
    wall_map: chex.Array        # (H, W) bool
    red_goal: chex.Array        # (2,) int32 [x, y]
    blue_goal: chex.Array       # (2,) int32 [x, y]
    time: chex.Array            # scalar int32 -- round-local time
    terminal: chex.Array        # scalar bool

    # Partner-related state.
    capability: chex.Array          # (2,) int32   -- [d_R, d_B]; FIXED across
                                    #    all rounds of a partner episode.
    partner_goal: chex.Array        # scalar int32 (0=UNSET, 1=RED, 2=BLUE)
    partner_move_ctr: chex.Array    # scalar int32 -- delay counter for
                                    #    partner navigation; reset per round.
    # ----- Split allocation fields -----
    # OBSERVATION-ONLY: what the ego picked on the most recent step. In both
    # influence and no-influence conditions this reflects the ego's action
    # channel exactly (NONE at t>=1 by legality mask). Never used to drive
    # the partner. This is what obs["last_allocation"] reports.
    last_ego_allocation: chex.Array  # scalar int32 (Allocations.*)
    # INTERNAL: the allocation that actually determines the partner's goal.
    # Written once per round (at t=0) and preserved until the round resets.
    #   influence=True  : partner_assignment = ego's t=0 alloc.
    #   influence=False : partner_assignment = round-parity forced alloc
    #                     (independent of ego action).
    # Not exposed in the observation under either condition.
    partner_assignment: chex.Array   # scalar int32 (Allocations.*)
    layout_idx: chex.Array          # scalar int32; addresses stacked layouts.
    round_idx: chex.Array           # scalar int32; which round (0-indexed).
    episode_layout_seq: chex.Array  # (rounds_per_episode,) int32


def _load_layout(layout_path: str) -> dict:
    """Parse a coordination-grid JSON into concrete arrays.

    Coordinates in the JSON are [row, col]; we convert to (x, y).
    """
    with open(layout_path, "r") as f:
        raw = json.load(f)

    grid = np.array(raw["grid"], dtype=np.int32)
    h, w = grid.shape
    wall_map = (grid == 1)

    def rc_to_xy(rc):
        return np.array([rc[1], rc[0]], dtype=np.int32)

    return {
        "_path": os.path.abspath(layout_path),
        "height": int(h),
        "width": int(w),
        "wall_map": jnp.asarray(wall_map, dtype=jnp.bool_),
        "wall_map_np": wall_map,
        "ego_start": jnp.asarray(rc_to_xy(raw["ego_start"]), dtype=jnp.int32),
        "partner_start": jnp.asarray(rc_to_xy(raw["partner_start"]), dtype=jnp.int32),
        "red_goal": jnp.asarray(rc_to_xy(raw["red_goal"]), dtype=jnp.int32),
        "blue_goal": jnp.asarray(rc_to_xy(raw["blue_goal"]), dtype=jnp.int32),
        "red_goal_np": rc_to_xy(raw["red_goal"]),
        "blue_goal_np": rc_to_xy(raw["blue_goal"]),
    }


def _bfs_next_actions(wall_map_np: np.ndarray, goal_xy_np: np.ndarray) -> np.ndarray:
    """Numpy multi-source BFS from goal. Returns (H, W) int array of the move
    to take from each cell to reduce distance to goal by 1. Walls and
    unreachable cells get STAY. Tie-break in the fixed order
    UP, DOWN, RIGHT, LEFT so partner navigation is deterministic.
    """
    h, w = wall_map_np.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    gx, gy = int(goal_xy_np[0]), int(goal_xy_np[1])

    if not (0 <= gx < w and 0 <= gy < h) or wall_map_np[gy, gx]:
        return np.full((h, w), int(Actions.stay), dtype=np.int32)

    dist[gy, gx] = 0
    q = deque([(gx, gy)])
    NEIGHBORS = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # (dx, dy) UP,DOWN,RIGHT,LEFT
    while q:
        cx, cy = q.popleft()
        for dx, dy in NEIGHBORS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and not wall_map_np[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = dist[cy, cx] + 1
                q.append((nx, ny))

    next_action = np.full((h, w), int(Actions.stay), dtype=np.int32)
    ACTION_DELTAS = [
        (int(Actions.up),    0, -1),
        (int(Actions.down),  0,  1),
        (int(Actions.right), 1,  0),
        (int(Actions.left), -1,  0),
    ]
    for cy in range(h):
        for cx in range(w):
            if wall_map_np[cy, cx] or dist[cy, cx] <= 0:
                # goal cell -> STAY; wall / unreachable -> STAY (default).
                continue
            target = dist[cy, cx] - 1
            for a, dx, dy in ACTION_DELTAS:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and dist[ny, nx] == target:
                    next_action[cy, cx] = a
                    break
    return next_action


def bfs_distance_map(wall_map_np: np.ndarray, goal_xy_np: np.ndarray) -> np.ndarray:
    """Numpy multi-source BFS from goal. Returns (H, W) int distance-to-goal
    (-1 for wall/unreachable). Used by the pre-PPO validation script to
    compute per-cell shortest-path lengths.
    """
    h, w = wall_map_np.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    gx, gy = int(goal_xy_np[0]), int(goal_xy_np[1])
    if not (0 <= gx < w and 0 <= gy < h) or wall_map_np[gy, gx]:
        return dist
    dist[gy, gx] = 0
    q = deque([(gx, gy)])
    NEIGHBORS = [(0, -1), (0, 1), (1, 0), (-1, 0)]
    while q:
        cx, cy = q.popleft()
        for dx, dy in NEIGHBORS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and not wall_map_np[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = dist[cy, cx] + 1
                q.append((nx, ny))
    return dist


SYMMETRY_NAMES = (
    "identity",         # 0
    "rot90_cw",         # 1
    "rot180",           # 2
    "rot270_cw",        # 3
    "flip_lr",          # 4  mirror across the vertical axis (left <-> right)
    "flip_ud",          # 5  mirror across the horizontal axis (top <-> bottom)
    "transpose",        # 6  mirror across the main diagonal
    "anti_transpose",   # 7  mirror across the anti-diagonal
)
N_SYMMETRIES = len(SYMMETRY_NAMES)


def _sym_position(g: int, xy: np.ndarray, n: int) -> np.ndarray:
    """Apply D4 symmetry ``g`` to a single (x, y) position on an n×n grid."""
    x = int(xy[0]); y = int(xy[1])
    if g == 0:
        nx, ny = x, y
    elif g == 1:
        nx, ny = n - 1 - y, x
    elif g == 2:
        nx, ny = n - 1 - x, n - 1 - y
    elif g == 3:
        nx, ny = y, n - 1 - x
    elif g == 4:
        nx, ny = n - 1 - x, y
    elif g == 5:
        nx, ny = x, n - 1 - y
    elif g == 6:
        nx, ny = y, x
    elif g == 7:
        nx, ny = n - 1 - y, n - 1 - x
    else:
        raise ValueError(f"unknown symmetry idx {g}")
    return np.array([nx, ny], dtype=np.int32)


def _sym_wall(g: int, wall_np: np.ndarray) -> np.ndarray:
    """Apply D4 symmetry ``g`` to a (H, W) wall array (indexed [y, x])."""
    if g == 0:
        return wall_np.copy()
    if g == 1:
        return np.rot90(wall_np, k=-1).copy()
    if g == 2:
        return np.rot90(wall_np, k=2).copy()
    if g == 3:
        return np.rot90(wall_np, k=1).copy()
    if g == 4:
        return np.fliplr(wall_np).copy()
    if g == 5:
        return np.flipud(wall_np).copy()
    if g == 6:
        return wall_np.T.copy()
    if g == 7:
        return wall_np[::-1, ::-1].T.copy()
    raise ValueError(f"unknown symmetry idx {g}")


def _augment_layout(base: dict, g: int, n: int) -> dict:
    wall_np = _sym_wall(g, base["wall_map_np"])
    ego = _sym_position(g, np.asarray(base["ego_start"], dtype=np.int32), n)
    partner = _sym_position(g, np.asarray(base["partner_start"], dtype=np.int32), n)
    red = _sym_position(g, np.asarray(base["red_goal"], dtype=np.int32), n)
    blue = _sym_position(g, np.asarray(base["blue_goal"], dtype=np.int32), n)
    return {
        "height": base["height"],
        "width": base["width"],
        "wall_map": jnp.asarray(wall_np, dtype=jnp.bool_),
        "wall_map_np": wall_np,
        "ego_start": jnp.asarray(ego, dtype=jnp.int32),
        "partner_start": jnp.asarray(partner, dtype=jnp.int32),
        "red_goal": jnp.asarray(red, dtype=jnp.int32),
        "blue_goal": jnp.asarray(blue, dtype=jnp.int32),
        "red_goal_np": red,
        "blue_goal_np": blue,
        "_sym_source_path": base.get("_path", None),
        "_sym_idx": int(g),
    }


def _resolve_layout_paths(
    layout_path: Optional[str],
    layout_paths: Optional[Sequence[str]],
    layouts_dir: Optional[str],
) -> List[str]:
    """Collapse the three ways to specify layouts into a concrete list."""
    if layout_paths is not None and len(list(layout_paths)) > 0:
        return list(layout_paths)
    if layouts_dir is not None:
        paths = sorted(glob.glob(os.path.join(layouts_dir, "*.json")))
        if not paths:
            raise FileNotFoundError(f"No *.json under {layouts_dir}")
        return paths
    if layout_path is not None:
        return [layout_path]
    raise ValueError(
        "Provide exactly one of layout_path=..., layout_paths=[...], "
        "or layouts_dir=..."
    )


def _canonicalize_capability_pairs(
    pairs: Optional[Sequence[Sequence[int]]],
    fallback: Sequence[Sequence[int]] = ((1, 1),),
) -> np.ndarray:
    """Return an (K, 2) int32 numpy array of capability *delay* pairs.

    Delay semantics: entry ``d_X >= 0`` = number of wait steps between
    consecutive partner moves while pursuing goal X (d=0 = every step).
    """
    if pairs is None or len(list(pairs)) == 0:
        pairs = fallback
    arr = np.asarray([[int(a), int(b)] for a, b in pairs], dtype=np.int32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"capability pairs must be shape (K, 2); got {arr.shape}"
        )
    if (arr < 0).any():
        raise ValueError(
            "capability delays must be >= 0 (d=0 means 'move every step')"
        )
    return arr


class CoordinationGrid(MultiAgentEnv):
    """CoordinationGrid with capability-vector scripted partner.

    The env still supports a pool of layouts; per-layout geometry and BFS
    tables are stacked along a leading K axis and indexed by
    ``state.layout_idx``. Layouts must all be the same H×W.
    """

    def __init__(
        self,
        layout_path: Optional[str] = None,
        layout_paths: Optional[Sequence[str]] = None,
        layouts_dir: Optional[str] = None,
        max_steps: int = 15,
        step_penalty: float = 0.01,
        success_reward: float = 1.0,
        partner_capability_pairs: Optional[Sequence[Sequence[int]]] = None,
        rounds_per_episode: int = 1,
        augment_symmetries: bool = False,
        hide_partner_until_time: int = 0,
        communication_condition: str = COMM_ACTION_ONLY,
        influence: bool = True,
        # legacy kwargs (accepted for backward compat, ignored otherwise):
        partner_z: Optional[float] = None,
        partner_z_values: Optional[Sequence[float]] = None,
    ):
        super().__init__(num_agents=2)

        if partner_z is not None or partner_z_values is not None:
            # These were the pre-capability scalar knobs; they are ignored now
            # but accepted to avoid crashing existing configs that still pass
            # them.  Warn loudly the first time.
            print(
                "[CoordinationGrid] Ignoring legacy partner_z / partner_z_values "
                "kwargs; use partner_capability_pairs instead.",
                flush=True,
            )

        if layout_path is None and layout_paths is None and layouts_dir is None:
            layout_path = DEFAULT_LAYOUT_PATH
        paths = _resolve_layout_paths(layout_path, layout_paths, layouts_dir)
        loaded_base = [_load_layout(p) for p in paths]
        self.layout_paths: List[str] = list(paths)
        self.n_base_layouts: int = len(loaded_base)

        h0, w0 = loaded_base[0]["height"], loaded_base[0]["width"]
        for i, ld in enumerate(loaded_base):
            if (ld["height"], ld["width"]) != (h0, w0):
                raise ValueError(
                    f"layout {paths[i]} shape {(ld['height'], ld['width'])} != "
                    f"first layout shape {(h0, w0)}"
                )
        if augment_symmetries and h0 != w0:
            raise ValueError(
                f"augment_symmetries=True requires square grids; got {h0}x{w0}"
            )
        self.height = h0
        self.width = w0

        self.augment_symmetries: bool = bool(augment_symmetries)
        self.symmetries_per_layout: int = (
            N_SYMMETRIES if self.augment_symmetries else 1
        )
        if self.augment_symmetries:
            loaded: List[dict] = []
            self.layout_source_paths: List[str] = []
            self.layout_sym_idx: List[int] = []
            for base in loaded_base:
                for g in range(N_SYMMETRIES):
                    loaded.append(_augment_layout(base, g, h0))
                    self.layout_source_paths.append(base["_path"])
                    self.layout_sym_idx.append(g)
        else:
            loaded = loaded_base
            self.layout_source_paths = [ld["_path"] for ld in loaded_base]
            self.layout_sym_idx = [0] * len(loaded_base)

        self.n_layouts: int = len(loaded)

        wall_maps_np = np.stack([np.asarray(ld["wall_map_np"], dtype=np.bool_)
                                  for ld in loaded], axis=0)
        ego_starts_np = np.stack([np.asarray(ld["ego_start"], dtype=np.int32)
                                   for ld in loaded], axis=0)
        partner_starts_np = np.stack([np.asarray(ld["partner_start"], dtype=np.int32)
                                       for ld in loaded], axis=0)
        red_goals_np = np.stack([np.asarray(ld["red_goal"], dtype=np.int32)
                                  for ld in loaded], axis=0)
        blue_goals_np = np.stack([np.asarray(ld["blue_goal"], dtype=np.int32)
                                   for ld in loaded], axis=0)

        self.wall_maps = jnp.asarray(wall_maps_np, dtype=jnp.bool_)
        self.ego_starts = jnp.asarray(ego_starts_np, dtype=jnp.int32)
        self.partner_starts = jnp.asarray(partner_starts_np, dtype=jnp.int32)
        self.red_goals = jnp.asarray(red_goals_np, dtype=jnp.int32)
        self.blue_goals = jnp.asarray(blue_goals_np, dtype=jnp.int32)

        self.wall_map = self.wall_maps[0]
        self.ego_start = self.ego_starts[0]
        self.partner_start = self.partner_starts[0]
        self.red_goal = self.red_goals[0]
        self.blue_goal = self.blue_goals[0]

        self.grid_shape = (self.height, self.width, 5)
        self.obs_shape = self.grid_shape
        # kept as ``msg_shape`` for backward compat with older code that
        # inspects that attribute; the payload is now the allocation one-hot.
        self.msg_shape = (N_ALLOCATIONS,)
        self.allocation_shape = (N_ALLOCATIONS,)

        self.agents = ["agent_0", "agent_1"]

        self.action_set = jnp.array(
            [Actions.up, Actions.down, Actions.right, Actions.left, Actions.stay],
            dtype=jnp.int32,
        )
        self.n_ego_actions = N_EGO_ACTIONS
        self.n_moves = N_MOVES
        self.n_allocations = N_ALLOCATIONS
        self.n_messages = N_ALLOCATIONS  # backward-compat alias

        self.max_steps = int(max_steps)
        self.step_penalty = float(step_penalty)
        self.success_reward = float(success_reward)

        # ---- Partner capability pool ----
        cap_np = _canonicalize_capability_pairs(
            partner_capability_pairs,
            fallback=TRAIN_CAPABILITY_PAIRS,
        )
        self.partner_capability_pairs = jnp.asarray(cap_np, dtype=jnp.int32)  # (K, 2)
        self.partner_capability_pairs_np = cap_np
        self.n_partner_capability: int = int(cap_np.shape[0])
        # Convenience scalar handle (first pair) for BC.
        self.partner_capability_default = tuple(int(v) for v in cap_np[0].tolist())

        self.rounds_per_episode = int(rounds_per_episode)
        if self.rounds_per_episode < 1:
            raise ValueError("rounds_per_episode must be >= 1")

        # Info-structure knob preserved for compatibility. K=0 (default) —
        # partner always visible. K>0 hides the partner-position channel
        # at round-local time < K.
        self.hide_partner_until_time = int(hide_partner_until_time)
        if self.hide_partner_until_time < 0:
            raise ValueError("hide_partner_until_time must be >= 0")

        if communication_condition not in COMM_CONDITIONS:
            raise ValueError(
                f"communication_condition must be one of {COMM_CONDITIONS}, "
                f"got {communication_condition!r}"
            )
        self.communication_condition: str = communication_condition
        self.influence: bool = bool(influence)
        self.t0_action_mask = jnp.asarray(ACTION_MASK_T0, dtype=jnp.bool_)
        self.t0_action_mask_np = np.asarray(ACTION_MASK_T0, dtype=np.bool_)

        # Precompute BFS next-action tables per layout, per goal color.
        next_red_np = np.stack(
            [_bfs_next_actions(ld["wall_map_np"], ld["red_goal_np"])
             for ld in loaded], axis=0,
        )
        next_blue_np = np.stack(
            [_bfs_next_actions(ld["wall_map_np"], ld["blue_goal_np"])
             for ld in loaded], axis=0,
        )
        self.next_action_toward_red = jnp.asarray(next_red_np, dtype=jnp.int32)
        self.next_action_toward_blue = jnp.asarray(next_blue_np, dtype=jnp.int32)

    # ------------------------------------------------------------------- reset
    def _build_state_for(
        self,
        layout_idx: chex.Array,
        capability: Optional[chex.Array] = None,
        round_idx: Optional[chex.Array] = None,
        episode_layout_seq: Optional[chex.Array] = None,
    ) -> "State":
        idx = layout_idx.astype(jnp.int32)
        wall_map = self.wall_maps[idx]
        ego_start = self.ego_starts[idx]
        partner_start = self.partner_starts[idx]
        red_goal = self.red_goals[idx]
        blue_goal = self.blue_goals[idx]
        agent_pos = jnp.stack([ego_start, partner_start], axis=0).astype(jnp.int32)
        if capability is None:
            cap_val = jnp.asarray(self.partner_capability_default, dtype=jnp.int32)
        else:
            cap_val = jnp.asarray(capability, dtype=jnp.int32)
        round_val = jnp.int32(0) if round_idx is None else jnp.asarray(round_idx, dtype=jnp.int32)
        if episode_layout_seq is None:
            eps_seq = jnp.full((self.rounds_per_episode,), idx, dtype=jnp.int32)
        else:
            eps_seq = jnp.asarray(episode_layout_seq, dtype=jnp.int32)
        return State(
            agent_pos=agent_pos,
            wall_map=wall_map,
            red_goal=red_goal,
            blue_goal=blue_goal,
            time=jnp.int32(0),
            terminal=jnp.bool_(False),
            capability=cap_val,
            partner_goal=jnp.int32(GOAL_UNSET),
            partner_move_ctr=jnp.int32(0),
            last_ego_allocation=jnp.int32(Allocations.none),
            partner_assignment=jnp.int32(Allocations.none),
            layout_idx=idx,
            round_idx=round_val,
            episode_layout_seq=eps_seq,
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], "State"]:
        """Full partner-episode reset: sample capability from the pool AND a
        fresh 20-layout sequence, and start at round 0.
        """
        key, k_layout, k_cap = jax.random.split(key, 3)
        layout_seq = jax.random.randint(
            k_layout, shape=(self.rounds_per_episode,),
            minval=0, maxval=self.n_layouts, dtype=jnp.int32,
        )
        cap_idx = jax.random.randint(
            k_cap, shape=(), minval=0,
            maxval=self.n_partner_capability, dtype=jnp.int32,
        )
        capability = self.partner_capability_pairs[cap_idx]  # (2,)
        state = self._build_state_for(
            layout_seq[0], capability=capability, round_idx=jnp.int32(0),
            episode_layout_seq=layout_seq,
        )
        obs = self.get_obs(state)
        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def reset_from_schedule(
        self, capability: chex.Array, layout_seq: chex.Array,
    ) -> Tuple[Dict[str, chex.Array], "State"]:
        """Deterministic reset from a pre-scheduled (capability, layout_seq).

        Used by the training scheduler to enforce a balanced sweep: every
        capability profile sees every training layout exactly once per sweep.
        """
        layout_seq = jnp.asarray(layout_seq, dtype=jnp.int32)
        state = self._build_state_for(
            layout_seq[0],
            capability=jnp.asarray(capability, dtype=jnp.int32),
            round_idx=jnp.int32(0),
            episode_layout_seq=layout_seq,
        )
        obs = self.get_obs(state)
        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def reset_to_layout(
        self, key: chex.PRNGKey, layout_idx: chex.Array,
        capability: Optional[chex.Array] = None,
    ) -> Tuple[Dict[str, chex.Array], "State"]:
        """Deterministic reset to a specific layout — for eval loops that need
        to visit every val/test layout N times. Also accepts an explicit
        ``capability`` so per-profile eval loops can force it.
        """
        state = self._build_state_for(
            jnp.asarray(layout_idx, dtype=jnp.int32),
            capability=(None if capability is None
                        else jnp.asarray(capability, dtype=jnp.int32)),
            round_idx=jnp.int32(0),
        )
        obs = self.get_obs(state)
        return lax.stop_gradient(obs), lax.stop_gradient(state)

    # --------------------------------------------------- partner navigation
    def _partner_next_move(
        self, partner_xy: chex.Array, goal: chex.Array, layout_idx: chex.Array
    ) -> chex.Array:
        """goal ∈ {UNSET, RED, BLUE}. Returns int32 move ∈ {0..4}.

        UNSET → STAY. This is the *single* point where the BFS tables are
        consulted; the cooldown mechanic wraps this in step_env.
        """
        x = partner_xy[0]
        y = partner_xy[1]
        move_red = self.next_action_toward_red[layout_idx, y, x]
        move_blue = self.next_action_toward_blue[layout_idx, y, x]
        is_red = goal == GOAL_RED
        is_blue = goal == GOAL_BLUE
        return jnp.where(
            is_red, move_red,
            jnp.where(is_blue, move_blue, jnp.int32(Actions.stay)),
        )

    # ------------------------------------------------------------- step_env
    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        ego_action = jnp.asarray(actions["agent_0"], dtype=jnp.int32)
        ego_move, ego_alloc = decode_ego(ego_action)  # scalars

        is_time0 = state.time == 0
        is_time1 = state.time == 1

        # RNG kept for API stability; commit / cooldown are deterministic.
        key, _k_unused = jax.random.split(key, 2)

        # ------- Partner goal commitment (once, at t=1, from partner_assignment) -------
        # partner_assignment was written at t=0 (see block below): under
        # influence=True it holds the ego's t=0 alloc; under influence=False
        # it holds the round-parity forced alloc. Same commit rule applies:
        # ALLOC_RED → partner takes BLUE; ALLOC_BLUE → RED. NONE (never
        # legal at t=0) → UNSET.
        partner_assignment = state.partner_assignment
        alloc_derived_goal = jnp.where(
            partner_assignment == int(Allocations.red),
            jnp.int32(GOAL_BLUE),
            jnp.where(
                partner_assignment == int(Allocations.blue),
                jnp.int32(GOAL_RED),
                jnp.int32(GOAL_UNSET),
            ),
        )
        should_commit = is_time1 & (state.partner_goal == GOAL_UNSET)
        new_partner_goal = jnp.where(should_commit, alloc_derived_goal, state.partner_goal)

        # ------- Delay mechanic (reference-style) -------
        # At t>=1, if partner has a committed goal AND move counter == 0,
        # partner takes one BFS step and counter resets to d (the delay for
        # the committed goal color). Otherwise partner STAYs and the counter
        # decrements (floored at 0). At t=0 the partner always STAYs.
        # d=0 -> moves every step; d=k -> moves once every (k+1) steps.
        d_for_goal = jnp.where(
            new_partner_goal == GOAL_RED,
            state.capability[0],
            jnp.where(
                new_partner_goal == GOAL_BLUE,
                state.capability[1],
                jnp.int32(0),          # UNSET: d doesn't matter
            ),
        )
        partner_has_goal = new_partner_goal != GOAL_UNSET
        can_move_now = (state.partner_move_ctr == 0) & partner_has_goal & (~is_time0)
        partner_greedy = self._partner_next_move(
            state.agent_pos[1], new_partner_goal, state.layout_idx
        )
        partner_effective_move = jnp.where(
            can_move_now, partner_greedy, jnp.int32(Actions.stay)
        )
        next_partner_move_ctr = jnp.where(
            can_move_now,
            d_for_goal,
            jnp.maximum(state.partner_move_ctr - jnp.int32(1), jnp.int32(0)),
        )

        # ------- Ego effective move -------
        # t=0: ego STAYs (mask enforces this at policy level; enforce here
        # too so hand-crafted tests can't sneak movement at t=0).
        ego_effective_move = jnp.where(
            is_time0, jnp.int32(Actions.stay), ego_move
        )

        # ------- Movement transition (walls, boundary, collision, swap) -------
        acts = self.action_set.take(
            indices=jnp.array([ego_effective_move, partner_effective_move])
        )
        deltas = DIR_TO_VEC[acts]
        proposed = state.agent_pos.astype(jnp.int32) + deltas

        in_bounds = (
            (proposed[:, 0] >= 0) & (proposed[:, 0] < self.width)
            & (proposed[:, 1] >= 0) & (proposed[:, 1] < self.height)
        )
        proposed_clipped = jnp.stack(
            [
                jnp.clip(proposed[:, 0], 0, self.width - 1),
                jnp.clip(proposed[:, 1], 0, self.height - 1),
            ],
            axis=1,
        )
        wall_hits = state.wall_map[proposed_clipped[:, 1], proposed_clipped[:, 0]]
        bounced = (~in_bounds) | wall_hits
        new_pos = jnp.where(
            bounced[:, None], state.agent_pos, proposed_clipped
        ).astype(jnp.int32)

        collision = jnp.all(new_pos[0] == new_pos[1])
        alice_pos = jnp.where(collision, state.agent_pos[0], new_pos[0])
        bob_pos = jnp.where(collision, state.agent_pos[1], new_pos[1])
        swap = jnp.all(new_pos[0] == state.agent_pos[1]) & jnp.all(
            new_pos[1] == state.agent_pos[0]
        )
        alice_pos = jnp.where((~collision) & swap, state.agent_pos[0], alice_pos)
        bob_pos = jnp.where((~collision) & swap, state.agent_pos[1], bob_pos)
        agent_pos = jnp.stack([alice_pos, bob_pos], axis=0).astype(jnp.int32)

        # ------- Time, success, reward, termination -------
        new_time = state.time + 1

        a0 = agent_pos[0]
        a1 = agent_pos[1]
        red = state.red_goal
        blue = state.blue_goal
        a0_red = jnp.all(a0 == red)
        a0_blue = jnp.all(a0 == blue)
        a1_red = jnp.all(a1 == red)
        a1_blue = jnp.all(a1 == blue)
        success = (a0_red & a1_blue) | (a0_blue & a1_red)

        reward = jnp.where(
            success,
            jnp.float32(self.success_reward),
            jnp.float32(-self.step_penalty),
        )
        round_done = success | (new_time >= self.max_steps)

        is_final_round = state.round_idx >= (self.rounds_per_episode - 1)
        intermediate_round = round_done & (~is_final_round)
        partner_episode_done = round_done & is_final_round

        # ------- Allocation write -------
        # partner_assignment (internal, drives the partner):
        #   influence=True  : written at t=0 from ego's alloc; preserved after.
        #   influence=False : written at t=0 from round-parity forced alloc,
        #                     independent of ego action; preserved after.
        # last_ego_allocation (observation only, identical schema in both
        # conditions): mirrors the ego's action channel every step, so the
        # ego always sees its own t=0 alloc at t=1 and NONE thereafter
        # (NONE is enforced by the legality mask at t>=1).
        forced_alloc = jnp.where(
            (state.round_idx % jnp.int32(2)) == jnp.int32(0),
            jnp.int32(Allocations.red),
            jnp.int32(Allocations.blue),
        )
        assignment_source_at_t0 = jnp.where(
            jnp.bool_(self.influence), ego_alloc, forced_alloc,
        )
        partner_assignment_written = jnp.where(
            is_time0, assignment_source_at_t0, state.partner_assignment,
        )
        last_ego_allocation_written = ego_alloc

        step_state = state.replace(
            agent_pos=agent_pos,
            time=new_time,
            terminal=partner_episode_done,
            partner_goal=new_partner_goal,
            partner_move_ctr=next_partner_move_ctr,
            last_ego_allocation=last_ego_allocation_written,
            partner_assignment=partner_assignment_written,
            # round_idx and capability unchanged by a plain step.
        )

        # Intermediate-round transition: next layout from the episode's
        # pre-scheduled sequence, KEEP capability, reset per-round fields.
        next_round_local = jnp.minimum(
            state.round_idx + 1, jnp.int32(self.rounds_per_episode - 1)
        )
        next_layout_idx = state.episode_layout_seq[next_round_local]
        next_round_state = self._build_state_for(
            next_layout_idx,
            capability=state.capability,
            round_idx=state.round_idx + 1,
            episode_layout_seq=state.episode_layout_seq,
        )

        new_state = jax.tree_util.tree_map(
            lambda a, b: jnp.where(intermediate_round, b, a),
            step_state, next_round_state,
        )

        obs = self.get_obs(new_state)

        rewards = {"agent_0": reward, "agent_1": reward}
        dones = {
            "agent_0": partner_episode_done,
            "agent_1": partner_episode_done,
            "__all__": partner_episode_done,
        }
        info = {
            "success": success,
            "round_done": round_done,
            "round_idx": state.round_idx,
            "layout_idx": state.layout_idx,
            # POST-transition round-local time. Use this — not state.time
            # (which is the pre-step time) — when recording the step at
            # which a round completes, so empirical completion times align
            # with the analytical convention in capability_selection.completion_time.
            "round_time": new_time,
            "capability": state.capability,           # (2,) int32 per env, (d_R, d_B)
            "capability_d_r": state.capability[0],
            "capability_d_b": state.capability[1],
            "partner_goal": new_partner_goal,
            # what actually drives the partner this step (equals ego alloc
            # under influence=True; equals the round-parity forced alloc
            # under influence=False). Persists across the round.
            "partner_assignment": partner_assignment_written,
            # what the ego selected this step. NONE at t>=1 by legality mask.
            "ego_alloc_action": ego_alloc,
            # Alias kept for callers written against the pre-split API.
            "pending_allocation": partner_assignment_written,
            "ego_move_effective": ego_effective_move,
            "partner_move_effective": partner_effective_move,
        }
        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(new_state),
            rewards,
            dones,
            info,
        )

    # ---------------------------------------------------------------- get_obs
    def get_obs(self, state: State) -> Dict[str, Dict[str, chex.Array]]:
        """Per-agent obs is a dict:
            {"grid": (H, W, 5), "last_allocation": (3,), "is_t0": ()}
        "last_allocation" is a one-hot over {NONE, RED, BLUE} for the
        allocation ego sent on the immediately preceding step
        (== state.pending_allocation).
        Capability is deliberately absent.
        """
        h, w = self.height, self.width
        walls = state.wall_map.astype(jnp.float32)

        def one_hot(pos):
            g = jnp.zeros((h, w), dtype=jnp.float32)
            return g.at[pos[1], pos[0]].set(1.0)

        red_layer = one_hot(state.red_goal)
        blue_layer = one_hot(state.blue_goal)
        ego_layer = one_hot(state.agent_pos[0])
        partner_layer = one_hot(state.agent_pos[1])

        partner_visible = (state.time >= jnp.int32(self.hide_partner_until_time))
        partner_layer = partner_layer * partner_visible.astype(jnp.float32)

        grid = jnp.stack(
            [walls, red_layer, blue_layer, ego_layer, partner_layer], axis=-1
        )

        # Observation carries the EGO'S action channel only, never the
        # internal partner_assignment, so the schema is identical under
        # influence=True and influence=False.
        last_allocation = jax.nn.one_hot(
            state.last_ego_allocation, N_ALLOCATIONS
        ).astype(jnp.float32)

        is_t0 = (state.time == 0).astype(jnp.float32)

        obs_i = {"grid": grid, "last_allocation": last_allocation, "is_t0": is_t0}
        return {"agent_0": obs_i, "agent_1": obs_i}
