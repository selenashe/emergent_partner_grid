"""
CoordinationGrid: single-round 7x7 grid coordination task with a scripted,
message-conditioned partner.

Ego (agent_0) action is a *joint* (move, message) pair, encoded as a single
discrete int in [0, 15):
    a = 3 * move + msg
    move ∈ {0:UP, 1:DOWN, 2:RIGHT, 3:LEFT, 4:STAY}
    msg  ∈ {0:NONE, 1:M0, 2:M1}
Use ``encode_ego(move, msg)`` / ``decode_ego(a)``.

Partner (agent_1) is scripted:
    * Latent partner type z ∈ [0, 1] is fixed per env (i.e. per partner).
    * At timestep 0 neither agent moves; ego only communicates.
    * At timestep 1, if partner_goal is UNSET, partner commits to a goal
      using the message ego sent at t=0 and z:
          P(RED | M0, z) = z,  P(RED | M1, z) = 1 - z
          P(RED | NONE) = 0.5  (neutral prior)
      This commit happens ONCE; later messages do not change the goal.
    * From t ≥ 1 the partner navigates greedily along a precomputed
      shortest-path table toward its committed goal. Navigation does NOT
      depend on z.

Observation for each agent is a dict:
    {"grid": (H, W, 5), "last_message": (3,), "is_t0": scalar float32}
where "last_message" is a one-hot over {NONE, M0, M1} for the message ego
sent on the immediately preceding step (NONE at t=0). This is what the ego
uses to correlate its own utterances with partner responses when inferring z.
``is_t0`` is 1.0 when ``state.time == 0`` (the free-comm step) and 0.0 else;
policies use it to mask their action space so movement is only sampled at
t>=1 and communication is only sampled at t=0.
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


class Messages(IntEnum):
    none = 0
    m0 = 1
    m1 = 2


N_MOVES = 5
N_MESSAGES = 3
N_EGO_ACTIONS = N_MOVES * N_MESSAGES  # 15

# Legality of the flat ego actions per environment phase. At t=0 the env
# forces movement=STAY, so only the three (STAY, msg) actions are behaviorally
# distinct. At t>=1 the scripted partner ignores the message channel (it
# committed at t=1 and never revisits), so only the five (move, NONE) actions
# are behaviorally distinct. Flat encoding is ``a = 3 * move + msg``.
#   t=0 legal ids: STAY(4)*3 + NONE/M0/M1 -> {12, 13, 14}
#   t>=1 legal ids: {UP,DOWN,RIGHT,LEFT,STAY}*3 + NONE -> {0, 3, 6, 9, 12}
LEGAL_ACTION_IDS_T0 = tuple(3 * int(Actions.stay) + m for m in range(N_MESSAGES))
LEGAL_ACTION_IDS_TGEQ1 = tuple(3 * mv + int(Messages.none) for mv in range(N_MOVES))
ACTION_MASK_T0 = np.zeros(N_EGO_ACTIONS, dtype=np.bool_)
ACTION_MASK_T0[list(LEGAL_ACTION_IDS_T0)] = True
ACTION_MASK_TGEQ1 = np.zeros(N_EGO_ACTIONS, dtype=np.bool_)
ACTION_MASK_TGEQ1[list(LEGAL_ACTION_IDS_TGEQ1)] = True

# ---------------------------------------------------------------------------
# Communication conditions.
# ---------------------------------------------------------------------------
# Three experimental settings that differ *only* in the causal role of the
# t=0 message channel (what messages mean, and whether they exist at all).
# The environment API (15-way flat action, obs dict, timing, rewards, nav
# dynamics, multi-round + z-per-episode sampling, D4 augmentation, mask-based
# policy) is otherwise identical across conditions.
#
#   action_only     : no explicit message channel. Partner samples
#                     uniformly regardless of the ego's t=0 action.
#                     Only ``STAY+NONE`` is legal at t=0.
#   universal       : messages have globally fixed meanings, independent of
#                     the partner's z. z is still sampled+stored (so the
#                     ego cannot detect the condition via obs), but the
#                     decoder ignores it.
#                         P(RED | NONE) = 0.5
#                         P(RED | M0)   = 1.0
#                         P(RED | M1)   = 0.0
#   partner_specific: messages are z-conditional (current behaviour).
#                         P(RED | NONE)     = 0.5
#                         P(RED | M0, z)    = z
#                         P(RED | M1, z)    = 1 - z
COMM_ACTION_ONLY = "action_only"
COMM_UNIVERSAL = "universal"
COMM_PARTNER_SPECIFIC = "partner_specific"
COMM_CONDITIONS = (COMM_ACTION_ONLY, COMM_UNIVERSAL, COMM_PARTNER_SPECIFIC)

# t=0 legal action set is condition-specific. t>=1 is always the movement-only
# 5-way mask, in all conditions.
_LEGAL_T0_ACTION_ONLY = (3 * int(Actions.stay) + int(Messages.none),)   # {12}
ACTION_MASK_T0_ACTION_ONLY = np.zeros(N_EGO_ACTIONS, dtype=np.bool_)
ACTION_MASK_T0_ACTION_ONLY[list(_LEGAL_T0_ACTION_ONLY)] = True
ACTION_MASK_T0_VERBAL = ACTION_MASK_T0   # {12, 13, 14}
LEGAL_ACTION_IDS_T0_ACTION_ONLY = _LEGAL_T0_ACTION_ONLY
LEGAL_ACTION_IDS_T0_VERBAL = LEGAL_ACTION_IDS_T0

_T0_ACTION_MASKS_BY_CONDITION = {
    COMM_ACTION_ONLY: ACTION_MASK_T0_ACTION_ONLY,
    COMM_UNIVERSAL: ACTION_MASK_T0_VERBAL,
    COMM_PARTNER_SPECIFIC: ACTION_MASK_T0_VERBAL,
}

# Partner-goal state encoding.
GOAL_UNSET = 0
GOAL_RED = 1
GOAL_BLUE = 2


def encode_ego(move, msg) -> int:
    """(move, msg) -> flat ego action id in [0, 15). Pure Python; JAX-safe."""
    return int(3) * int(move) + int(msg)


def decode_ego(action):
    """Flat action -> (move, msg). JAX-compatible; accepts scalar or array."""
    a = jnp.asarray(action, dtype=jnp.int32)
    move = a // 3
    msg = a % 3
    return move, msg


DEFAULT_LAYOUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "dev", "grids", "layouts", "train", "train_0000.json",
)


@struct.dataclass
class State:
    # Round-local geometry (sampled fresh at each round boundary within a partner episode).
    agent_pos: chex.Array       # (2, 2) int32 -- [x, y] per agent
    wall_map: chex.Array        # (H, W) bool
    red_goal: chex.Array        # (2,) int32 [x, y]
    blue_goal: chex.Array       # (2,) int32 [x, y]
    time: chex.Array            # scalar int32   -- round-local time (resets on new round)
    terminal: chex.Array        # scalar bool

    # Partner-related state.
    z: chex.Array               # scalar float32 in [0, 1]; FIXED across all rounds of a
                                # partner episode; resampled only on full env reset.
    partner_goal: chex.Array    # scalar int32 (0=UNSET, 1=RED, 2=BLUE); reset each round.
    pending_message: chex.Array # scalar int32 (0=NONE, 1=M0, 2=M1) -- ego's msg from LAST step
    layout_idx: chex.Array      # scalar int32; addresses the stacked layout / BFS tables.
    round_idx: chex.Array       # scalar int32; which round of the partner episode we're in.
                                # 0-indexed; ranges over [0, rounds_per_episode).
    episode_layout_seq: chex.Array   # (rounds_per_episode,) int32 — sequence of
                                # layout_idx values the partner episode will visit
                                # across its rounds. On the intermediate-round
                                # transition inside step_env, the next layout is
                                # ``episode_layout_seq[round_idx + 1]`` instead of
                                # a fresh random sample. reset(key) fills this
                                # with a random draw for BC; the trainer's
                                # scheduler uses reset_from_schedule to fill it
                                # with a pre-scheduled sequence.


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
    UP, DOWN, RIGHT, LEFT so partner navigation is deterministic across z.
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
    """Apply D4 symmetry ``g`` to a single (x, y) position on an n×n grid.
    Returns a new np.int32 array of shape (2,).
    """
    x = int(xy[0]); y = int(xy[1])
    if g == 0:                                     # identity
        nx, ny = x, y
    elif g == 1:                                   # rot 90 CW
        nx, ny = n - 1 - y, x
    elif g == 2:                                   # rot 180
        nx, ny = n - 1 - x, n - 1 - y
    elif g == 3:                                   # rot 270 CW (= 90 CCW)
        nx, ny = y, n - 1 - x
    elif g == 4:                                   # flip left <-> right
        nx, ny = n - 1 - x, y
    elif g == 5:                                   # flip top <-> bottom
        nx, ny = x, n - 1 - y
    elif g == 6:                                   # transpose (main diag)
        nx, ny = y, x
    elif g == 7:                                   # anti-transpose
        nx, ny = n - 1 - y, n - 1 - x
    else:
        raise ValueError(f"unknown symmetry idx {g}")
    return np.array([nx, ny], dtype=np.int32)


def _sym_wall(g: int, wall_np: np.ndarray) -> np.ndarray:
    """Apply D4 symmetry ``g`` to a (H, W) wall array (indexed [y, x])."""
    if g == 0:
        return wall_np.copy()
    if g == 1:
        return np.rot90(wall_np, k=-1).copy()       # CW 90
    if g == 2:
        return np.rot90(wall_np, k=2).copy()
    if g == 3:
        return np.rot90(wall_np, k=1).copy()        # CCW 90
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
    """Return a new layout dict obtained by applying D4 symmetry ``g`` to
    ``base``. Positions and walls are transformed; BFS tables are NOT copied
    from ``base`` — the caller re-runs BFS on the transformed geometry so
    action-direction remaps can't get out of sync.
    """
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


class CoordinationGrid(MultiAgentEnv):
    """CoordinationGrid with scripted, message-conditioned partner.

    Supports a *pool* of layouts. On each ``reset`` a layout is sampled
    uniformly from the pool (or the pool has size 1 for the single-layout
    case, which is fully backward compatible). All per-layout geometry
    (wall_map, starts, goals) plus precomputed BFS next-action tables are
    stacked along a leading ``K`` axis; ``state.layout_idx`` records which
    slice the current episode uses, and ``_partner_next_move`` indexes into
    the stacked BFS tables using it. Layouts must all be the same H×W.
    """

    def __init__(
        self,
        layout_path: Optional[str] = None,
        layout_paths: Optional[Sequence[str]] = None,
        layouts_dir: Optional[str] = None,
        max_steps: int = 15,
        step_penalty: float = 0.01,
        success_reward: float = 1.0,
        partner_z: Optional[float] = None,
        partner_z_values: Optional[Sequence[float]] = None,
        rounds_per_episode: int = 1,
        augment_symmetries: bool = False,
        hide_partner_until_time: int = 0,
        communication_condition: str = COMM_PARTNER_SPECIFIC,
    ):
        super().__init__(num_agents=2)

        if layout_path is None and layout_paths is None and layouts_dir is None:
            layout_path = DEFAULT_LAYOUT_PATH
        paths = _resolve_layout_paths(layout_path, layout_paths, layouts_dir)
        loaded_base = [_load_layout(p) for p in paths]
        self.layout_paths: List[str] = list(paths)
        self.n_base_layouts: int = len(loaded_base)

        # H×W must match across all base layouts (network / obs shape depends
        # on it). Grids must also be *square* to admit the D4 group without
        # changing shape.
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

        # Expand the base layouts through the D4 group, or not. When augmented,
        # the effective pool is 8x larger; ``reset`` samples uniformly from it.
        # We recompute BFS on each transformed geometry rather than remapping
        # action IDs — same result, half the failure modes.
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

        # Stack per-layout geometry along a leading axis (K, ...). Cast/stack
        # via numpy first so we do everything in one shot and hand JAX a
        # concrete dtype.
        wall_maps_np = np.stack([np.asarray(ld["wall_map_np"], dtype=np.bool_)
                                  for ld in loaded], axis=0)  # (K, H, W)
        ego_starts_np = np.stack([np.asarray(ld["ego_start"], dtype=np.int32)
                                   for ld in loaded], axis=0)                       # (K, 2)
        partner_starts_np = np.stack([np.asarray(ld["partner_start"], dtype=np.int32)
                                       for ld in loaded], axis=0)                    # (K, 2)
        red_goals_np = np.stack([np.asarray(ld["red_goal"], dtype=np.int32)
                                  for ld in loaded], axis=0)                         # (K, 2)
        blue_goals_np = np.stack([np.asarray(ld["blue_goal"], dtype=np.int32)
                                   for ld in loaded], axis=0)                        # (K, 2)

        self.wall_maps = jnp.asarray(wall_maps_np, dtype=jnp.bool_)
        self.ego_starts = jnp.asarray(ego_starts_np, dtype=jnp.int32)
        self.partner_starts = jnp.asarray(partner_starts_np, dtype=jnp.int32)
        self.red_goals = jnp.asarray(red_goals_np, dtype=jnp.int32)
        self.blue_goals = jnp.asarray(blue_goals_np, dtype=jnp.int32)

        # For single-layout callers who reach in directly for these fields.
        self.wall_map = self.wall_maps[0]
        self.ego_start = self.ego_starts[0]
        self.partner_start = self.partner_starts[0]
        self.red_goal = self.red_goals[0]
        self.blue_goal = self.blue_goals[0]

        # Observation shapes for downstream code that inspects .obs_shape /
        # .msg_shape (e.g. policy heads). obs_shape stays the grid shape for
        # backward compat with any consumer that reads it.
        self.grid_shape = (self.height, self.width, 5)
        self.obs_shape = self.grid_shape
        self.msg_shape = (N_MESSAGES,)

        self.agents = ["agent_0", "agent_1"]

        self.action_set = jnp.array(
            [Actions.up, Actions.down, Actions.right, Actions.left, Actions.stay],
            dtype=jnp.int32,
        )
        self.n_ego_actions = N_EGO_ACTIONS
        self.n_moves = N_MOVES
        self.n_messages = N_MESSAGES

        self.max_steps = int(max_steps)
        self.step_penalty = float(step_penalty)
        self.success_reward = float(success_reward)

        # ---- Partner-z sampling pool ----
        # `partner_z_values` (list) is the primary knob for Stage C+D — a
        # partner-episode reset samples one z uniformly from this pool and
        # holds it for all `rounds_per_episode` rounds.
        # Backward compat: if only the old scalar `partner_z` is given, we
        # treat that as a length-1 pool (matches Stage A/B behaviour).
        if partner_z_values is not None:
            zs = [float(v) for v in partner_z_values]
            if len(zs) == 0:
                raise ValueError("partner_z_values must be non-empty")
        elif partner_z is not None:
            zs = [float(partner_z)]
        else:
            zs = [0.5]
        self.partner_z_values = jnp.asarray(zs, dtype=jnp.float32)   # (Z,)
        self.n_partner_z: int = len(zs)
        # Keep .partner_z as a scalar for BC (some callers look at this
        # directly). Points at the first entry of the pool.
        self.partner_z = float(zs[0])

        self.rounds_per_episode = int(rounds_per_episode)
        if self.rounds_per_episode < 1:
            raise ValueError("rounds_per_episode must be >= 1")

        # Information-structure intervention: zero the partner-position
        # channel of the grid observation while ``state.time <
        # hide_partner_until_time`` (round-local time). K=0 (default) is
        # backward-compatible — partner is always visible. K=3 hides partner
        # at t=0, 1, 2 (i.e. through the first two movement steps) and
        # reveals it at t=3 onward. This forces the ego to rely on the
        # t=0 message (and therefore on knowing z) rather than on cheap
        # behavioural inference from partner motion.
        self.hide_partner_until_time = int(hide_partner_until_time)
        if self.hide_partner_until_time < 0:
            raise ValueError("hide_partner_until_time must be >= 0")

        # Communication condition (see COMM_CONDITIONS above). Fixed per env
        # instance. Different conditions produce different p_red rules and a
        # different t=0 action-legality mask; nothing else in the env changes.
        if communication_condition not in COMM_CONDITIONS:
            raise ValueError(
                f"communication_condition must be one of {COMM_CONDITIONS}, "
                f"got {communication_condition!r}"
            )
        self.communication_condition: str = communication_condition
        # Convenient handle so downstream code (network mask, tests) can grab
        # the correct t=0 mask without re-implementing the switch.
        self.t0_action_mask = jnp.asarray(
            _T0_ACTION_MASKS_BY_CONDITION[self.communication_condition],
            dtype=jnp.bool_,
        )
        self.t0_action_mask_np = np.asarray(
            _T0_ACTION_MASKS_BY_CONDITION[self.communication_condition],
            dtype=np.bool_,
        )

        # Precompute BFS next-action tables per layout. Shape (K, H, W).
        # Partner navigation lookups always index by state.layout_idx, which
        # means later stages (layout×z sampling wrappers) only need to point
        # at these tables with a different idx; no other code path changes.
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
        z: Optional[chex.Array] = None,
        round_idx: Optional[chex.Array] = None,
        episode_layout_seq: Optional[chex.Array] = None,
    ) -> "State":
        """Build a fresh round-start State on the given layout.

        z / round_idx default to the pool's first z / 0 respectively — this
        is what a Stage A/B single-round env wants. For Stage C+D, the caller
        (``reset``, or the intermediate-round transition inside ``step_env``)
        threads in the appropriate z and round_idx explicitly.
        """
        idx = layout_idx.astype(jnp.int32)
        wall_map = self.wall_maps[idx]
        ego_start = self.ego_starts[idx]
        partner_start = self.partner_starts[idx]
        red_goal = self.red_goals[idx]
        blue_goal = self.blue_goals[idx]
        agent_pos = jnp.stack([ego_start, partner_start], axis=0).astype(jnp.int32)
        z_val = jnp.float32(self.partner_z) if z is None else jnp.asarray(z, dtype=jnp.float32)
        round_val = jnp.int32(0) if round_idx is None else jnp.asarray(round_idx, dtype=jnp.int32)
        if episode_layout_seq is None:
            # Fill remaining rounds with the current layout as a benign
            # default. Random-reset callers (`reset(key)`) override this
            # further down with a proper random sequence.
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
            z=z_val,
            partner_goal=jnp.int32(GOAL_UNSET),
            pending_message=jnp.int32(Messages.none),
            layout_idx=idx,
            round_idx=round_val,
            episode_layout_seq=eps_seq,
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], "State"]:
        # Full partner-episode reset: sample a fresh partner-type z from the
        # allowed pool AND a fresh sequence of ``rounds_per_episode`` random
        # layouts, and start at round 0.
        key, k_layout, k_z = jax.random.split(key, 3)
        # Uniform random layout for each round of the partner episode.
        layout_seq = jax.random.randint(
            k_layout, shape=(self.rounds_per_episode,),
            minval=0, maxval=self.n_layouts, dtype=jnp.int32,
        )
        z_idx = jax.random.randint(
            k_z, shape=(), minval=0, maxval=self.n_partner_z, dtype=jnp.int32
        )
        z = self.partner_z_values[z_idx]
        state = self._build_state_for(
            layout_seq[0], z=z, round_idx=jnp.int32(0),
            episode_layout_seq=layout_seq,
        )
        obs = self.get_obs(state)
        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def reset_from_schedule(
        self, z: chex.Array, layout_seq: chex.Array,
    ) -> Tuple[Dict[str, chex.Array], "State"]:
        """Deterministic reset from a pre-scheduled (z, layout_seq).

        Used by the training scheduler to enforce a balanced sweep: every
        z sees every training layout exactly once per sweep. Bypasses
        random sampling entirely; layout_seq must have length exactly
        ``rounds_per_episode``.
        """
        layout_seq = jnp.asarray(layout_seq, dtype=jnp.int32)
        state = self._build_state_for(
            layout_seq[0],
            z=jnp.asarray(z, dtype=jnp.float32),
            round_idx=jnp.int32(0),
            episode_layout_seq=layout_seq,
        )
        obs = self.get_obs(state)
        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def reset_to_layout(
        self, key: chex.PRNGKey, layout_idx: chex.Array,
        z: Optional[chex.Array] = None,
    ) -> Tuple[Dict[str, chex.Array], "State"]:
        """Deterministic reset to a specific layout — for eval loops that need
        to visit every val/test layout N times. Also accepts an explicit ``z``
        so per-z eval loops can force the partner type."""
        state = self._build_state_for(
            jnp.asarray(layout_idx, dtype=jnp.int32),
            z=(None if z is None else jnp.asarray(z, dtype=jnp.float32)),
            round_idx=jnp.int32(0),
        )
        obs = self.get_obs(state)
        return lax.stop_gradient(obs), lax.stop_gradient(state)

    # --------------------------------------------------- partner navigation
    def _partner_next_move(
        self, partner_xy: chex.Array, goal: chex.Array, layout_idx: chex.Array
    ) -> chex.Array:
        """goal ∈ {UNSET, RED, BLUE}. Returns int32 move ∈ {0..4}.
        UNSET (no commit yet) → STAY. This is the *single* point where the
        BFS tables are consulted; extending to multiple layouts later only
        requires stacking self.next_action_toward_* along axis 0 and passing
        the appropriate layout_idx.
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
        ego_move, ego_msg = decode_ego(ego_action)  # scalars

        is_time0 = state.time == 0
        is_time1 = state.time == 1

        # Split the step key up-front — we need three independent streams:
        # partner-goal sample, next-round layout sample (used only if this
        # step ends a non-final round), and one to thread out for future use.
        key, k_partner, k_next_layout = jax.random.split(key, 3)

        # ------- Partner goal commitment (once, at t=1, from t=0's msg) -------
        # p_red is the ONLY thing the communication condition changes.
        pending_msg = state.pending_message
        if self.communication_condition == COMM_ACTION_ONLY:
            # No message channel. Partner samples uniformly regardless of the
            # ego's t=0 action or z. If the ego somehow sends an M0/M1 anyway
            # (e.g. hand-written test with the mask bypassed) the message is
            # ignored — that's the entire point of this condition.
            p_red = jnp.float32(0.5)
        elif self.communication_condition == COMM_UNIVERSAL:
            # Universal fixed meanings. z is stored on state (so the ego can't
            # deduce the condition from obs), but the decoder ignores it.
            #   NONE -> 0.5, M0 -> RED w.p. 1, M1 -> BLUE w.p. 1.
            p_red = jnp.where(
                pending_msg == Messages.none, jnp.float32(0.5),
                jnp.where(
                    pending_msg == Messages.m0, jnp.float32(1.0),
                    jnp.float32(0.0),
                ),
            )
        else:  # COMM_PARTNER_SPECIFIC (current)
            p_red = jnp.where(
                pending_msg == Messages.none, jnp.float32(0.5),
                jnp.where(
                    pending_msg == Messages.m0, state.z,
                    jnp.float32(1.0) - state.z,
                ),
            )
        sample_red = jax.random.bernoulli(k_partner, p=p_red)
        sampled_goal = jnp.where(
            sample_red, jnp.int32(GOAL_RED), jnp.int32(GOAL_BLUE)
        )
        should_commit = is_time1 & (state.partner_goal == GOAL_UNSET)
        new_partner_goal = jnp.where(should_commit, sampled_goal, state.partner_goal)

        # ------- Effective moves this step -------
        # t=0: both agents STAY; ego msg is recorded for next step.
        # t>=1: ego moves as requested; partner navigates toward committed goal.
        ego_effective_move = jnp.where(
            is_time0, jnp.int32(Actions.stay), ego_move
        )
        partner_greedy = self._partner_next_move(
            state.agent_pos[1], new_partner_goal, state.layout_idx
        )
        partner_effective_move = jnp.where(
            is_time0, jnp.int32(Actions.stay), partner_greedy
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
        # A round ends on either success or hitting the per-round horizon.
        round_done = success | (new_time >= self.max_steps)

        # Stage C+D: a partner episode contains `rounds_per_episode` rounds.
        # Only the final round triggers done["__all__"] (which in turn triggers
        # JaxMARL's autoreset and a fresh partner-type z on the next call).
        # Intermediate round ends produce a fresh in-episode round: new layout,
        # positions/goals/partner_goal/pending_message reset, time=0, SAME z,
        # round_idx incremented.
        is_final_round = state.round_idx >= (self.rounds_per_episode - 1)
        intermediate_round = round_done & (~is_final_round)
        partner_episode_done = round_done & is_final_round

        # State if the step is "just another step" (round continues or the
        # final round just ended — autoreset handles the latter externally).
        step_state = state.replace(
            agent_pos=agent_pos,
            time=new_time,
            terminal=partner_episode_done,
            partner_goal=new_partner_goal,
            pending_message=ego_msg,
            # round_idx and z unchanged by a plain step.
        )

        # State if an intermediate round just ended: pull the pre-scheduled
        # layout for the NEXT round from state.episode_layout_seq, keep z,
        # keep the full sequence, bump round_idx by 1. If the sequence was
        # populated by ``reset(key)`` (random) the effect is a fresh random
        # layout; if it came from ``reset_from_schedule`` the effect is the
        # trainer's pre-planned sweep.
        next_round_local = jnp.minimum(
            state.round_idx + 1, jnp.int32(self.rounds_per_episode - 1)
        )
        next_layout_idx = state.episode_layout_seq[next_round_local]
        next_round_state = self._build_state_for(
            next_layout_idx,
            z=state.z,
            round_idx=state.round_idx + 1,
            episode_layout_seq=state.episode_layout_seq,
        )

        # Select which state to return. `intermediate_round` is a scalar bool.
        new_state = jax.tree_util.tree_map(
            lambda a, b: jnp.where(intermediate_round, b, a),
            step_state, next_round_state,
        )

        obs = self.get_obs(new_state)

        rewards = {"agent_0": reward, "agent_1": reward}
        # done["__all__"] is True ONLY on the terminal step of the final round.
        dones = {
            "agent_0": partner_episode_done,
            "agent_1": partner_episode_done,
            "__all__": partner_episode_done,
        }
        info = {
            "success": success,                       # coordination success this step
            "round_done": round_done,                 # this step ended a round
            "round_idx": state.round_idx,             # which round just ran (0-indexed)
            "z": state.z,                             # partner type for this episode
            "partner_goal": new_partner_goal,
            "pending_message": ego_msg,
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
            {"grid": (H, W, 5), "last_message": (3,), "is_t0": ()}
        "last_message" is a one-hot over {NONE, M0, M1} for the message ego
        sent on the immediately preceding step (== state.pending_message).
        "is_t0" is 1.0 iff state.time == 0 (the free-comm step) and 0.0
        otherwise; a masked-action policy uses it to disable movement at
        t=0 and messaging at t>=1.
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

        # Information-structure intervention: hide the partner-position
        # channel until state.time >= hide_partner_until_time. When K=0
        # (default) this is a no-op. This is applied round-locally because
        # state.time is round-local (resets to 0 at every round boundary),
        # so the intervention repeats within every round of a partner ep.
        partner_visible = (state.time >= jnp.int32(self.hide_partner_until_time))
        partner_layer = partner_layer * partner_visible.astype(jnp.float32)

        grid = jnp.stack(
            [walls, red_layer, blue_layer, ego_layer, partner_layer], axis=-1
        )  # (H, W, 5)

        last_message = jax.nn.one_hot(
            state.pending_message, N_MESSAGES
        ).astype(jnp.float32)  # (3,)

        is_t0 = (state.time == 0).astype(jnp.float32)                 # scalar

        obs_i = {"grid": grid, "last_message": last_message, "is_t0": is_t0}
        return {"agent_0": obs_i, "agent_1": obs_i}
