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
    {"grid": (H, W, 5), "last_message": (3,)}
where "last_message" is a one-hot over {NONE, M0, M1} for the message ego
sent on the immediately preceding step (NONE at t=0). This is what the ego
uses to correlate its own utterances with partner responses when inferring z.
"""

import json
import os
from collections import deque
from enum import IntEnum
from typing import Dict, Tuple

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
    # Geometry (currently per-episode; will be swapped for a stacked layout lookup later).
    agent_pos: chex.Array       # (2, 2) int32 -- [x, y] per agent
    wall_map: chex.Array        # (H, W) bool
    red_goal: chex.Array        # (2,) int32 [x, y]
    blue_goal: chex.Array       # (2,) int32 [x, y]
    time: chex.Array            # scalar int32
    terminal: chex.Array        # scalar bool

    # Partner-related state.
    z: chex.Array               # scalar float32 in [0, 1]
    partner_goal: chex.Array    # scalar int32 (0=UNSET, 1=RED, 2=BLUE)
    pending_message: chex.Array # scalar int32 (0=NONE, 1=M0, 2=M1) — ego's msg from LAST step
    layout_idx: chex.Array      # scalar int32; addresses BFS tables (future multi-layout)


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


class CoordinationGrid(MultiAgentEnv):
    """CoordinationGrid with scripted, message-conditioned partner."""

    def __init__(
        self,
        layout_path: str = DEFAULT_LAYOUT_PATH,
        max_steps: int = 15,
        step_penalty: float = 0.01,
        success_reward: float = 1.0,
        partner_z: float = 0.5,
    ):
        super().__init__(num_agents=2)

        layout = _load_layout(layout_path)
        self.height = layout["height"]
        self.width = layout["width"]
        self.wall_map = layout["wall_map"]
        self.ego_start = layout["ego_start"]
        self.partner_start = layout["partner_start"]
        self.red_goal = layout["red_goal"]
        self.blue_goal = layout["blue_goal"]

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
        self.partner_z = float(partner_z)

        # Precompute BFS next-action tables. Shape (K=1, H, W) — leading
        # dim anticipates a per-layout stack once multi-layout scaffolding
        # lands. Partner navigation lookups always index by layout_idx.
        next_red = _bfs_next_actions(layout["wall_map_np"], layout["red_goal_np"])
        next_blue = _bfs_next_actions(layout["wall_map_np"], layout["blue_goal_np"])
        self.next_action_toward_red = jnp.asarray(
            next_red[None, :, :], dtype=jnp.int32
        )   # (1, H, W)
        self.next_action_toward_blue = jnp.asarray(
            next_blue[None, :, :], dtype=jnp.int32
        )   # (1, H, W)

    # ------------------------------------------------------------------- reset
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        agent_pos = jnp.stack([self.ego_start, self.partner_start], axis=0).astype(
            jnp.int32
        )
        state = State(
            agent_pos=agent_pos,
            wall_map=self.wall_map,
            red_goal=self.red_goal,
            blue_goal=self.blue_goal,
            time=jnp.int32(0),
            terminal=jnp.bool_(False),
            z=jnp.float32(self.partner_z),
            partner_goal=jnp.int32(GOAL_UNSET),
            pending_message=jnp.int32(Messages.none),
            layout_idx=jnp.int32(0),
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

        # ------- Partner goal commitment (once, at t=1, from t=0's msg) -------
        pending_msg = state.pending_message
        p_red = jnp.where(
            pending_msg == Messages.none, jnp.float32(0.5),
            jnp.where(
                pending_msg == Messages.m0, state.z,
                jnp.float32(1.0) - state.z,
            ),
        )
        key, subkey = jax.random.split(key)
        sample_red = jax.random.bernoulli(subkey, p=p_red)
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
        done = success | (new_time >= self.max_steps)

        new_state = state.replace(
            agent_pos=agent_pos,
            time=new_time,
            terminal=done,
            partner_goal=new_partner_goal,
            pending_message=ego_msg,  # ego's msg THIS step becomes pending for NEXT
        )

        obs = self.get_obs(new_state)
        rewards = {"agent_0": reward, "agent_1": reward}
        dones = {"agent_0": done, "agent_1": done, "__all__": done}
        info = {
            "success": success,
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
            {"grid": (H, W, 5), "last_message": (3,)}
        "last_message" is a one-hot over {NONE, M0, M1} for the message ego
        sent on the immediately preceding step (== state.pending_message).
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

        grid = jnp.stack(
            [walls, red_layer, blue_layer, ego_layer, partner_layer], axis=-1
        )  # (H, W, 5)

        last_message = jax.nn.one_hot(
            state.pending_message, N_MESSAGES
        ).astype(jnp.float32)  # (3,)

        obs_i = {"grid": grid, "last_message": last_message}
        return {"agent_0": obs_i, "agent_1": obs_i}
