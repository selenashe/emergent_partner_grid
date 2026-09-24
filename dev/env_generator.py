"""Generate a train/val/test corpus of 7x7 two-agent, two-goal grid mazes.

Each generated environment satisfies:
  * both agents can reach both goals,
  * agent-start and goal locations are sampled without positional bias,
  * walls / shortest-path lengths vary,
  * both goal-assignments (ego->red, partner->blue) and (ego->blue, partner->red)
    are physically feasible.

Grid-cell encoding used in the saved numpy arrays and JSON `grid` field:
    0 = empty
    1 = wall
    2 = red goal
    3 = blue goal
    4 = ego start
    5 = partner start

Outputs (under --out_dir, default `dev/grids`):
    layouts/{split}/{layout_id}.json     - per-env JSON metadata
    renders/{split}/{layout_id}.png      - per-env PNG render
    {split}_layouts.npz                  - compiled (N, H, W) int8 array per split,
                                           key `layouts`

Example usage:
    # 200 envs, default 70/15/15 split, default WALL_DENSITY [0.15, 0.60]
    python dev/env_generator.py --n 200 --master_seed 0

    # Custom density and grid size, larger corpus, custom output dir
    python dev/env_generator.py \\
        --n 1000 \\
        --grid_size 7 \\
        --wall_density 0.20 0.35 \\
        --train_test_val_ratio 0.8 0.1 0.1 \\
        --master_seed 42 \\
        --out_dir dev/grids_v2
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle

# ----- cell codes for the compiled numpy representation -----
EMPTY = 0
WALL = 1
RED = 2
BLUE = 3
EGO = 4
PARTNER = 5


@dataclass
class GridEnv:
    """A single generated environment.

    `grid` stores only walls/empties (values EMPTY/WALL). Agents and goals are
    tracked as separate coordinates so we can render them on top and encode
    them into the compiled numpy array.
    """

    grid: np.ndarray
    ego_start: Tuple[int, int]
    partner_start: Tuple[int, int]
    red_goal: Tuple[int, int]
    blue_goal: Tuple[int, int]
    seed: int
    wall_density: float  # sampled wall probability (the parameter value)
    metadata: Optional[Dict] = None  # populated by compute_metrics(); saved in JSON


# --------------------------------------------------------------------------- #
# Reachability                                                                #
# --------------------------------------------------------------------------- #

def bfs_distances(grid: np.ndarray, src: Tuple[int, int]) -> np.ndarray:
    """4-connected shortest-path distances from `src`. Unreachable cells are -1."""
    H, W = grid.shape
    dist = -np.ones_like(grid, dtype=np.int32)
    dist[src] = 0
    q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == EMPTY and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _all_pairs_reachable(env: GridEnv) -> bool:
    for agent in (env.ego_start, env.partner_start):
        d = bfs_distances(env.grid, agent)
        for goal in (env.red_goal, env.blue_goal):
            if d[goal] < 0:
                return False
    return True


# --------------------------------------------------------------------------- #
# Structural + coordination metrics                                            #
# --------------------------------------------------------------------------- #

# Canonical neighbor order used everywhere path enumeration needs a tie-break.
_NBR_ORDER: Tuple[Tuple[int, int], ...] = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _traversable_neighbors(grid: np.ndarray, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
    H, W = grid.shape
    r, c = cell
    out = []
    for dr, dc in _NBR_ORDER:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == EMPTY:
            out.append((nr, nc))
    return out


def bfs_path_counts(grid: np.ndarray, src: Tuple[int, int]
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dist, count): count[v] = number of distinct shortest paths src->v."""
    H, W = grid.shape
    dist = -np.ones((H, W), dtype=np.int32)
    count = np.zeros((H, W), dtype=np.int64)
    dist[src] = 0
    count[src] = 1
    order: List[Tuple[int, int]] = [src]
    q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in _NBR_ORDER:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == EMPTY and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                order.append((nr, nc))
                q.append((nr, nc))
    # DP in BFS order: count[v] = sum count[u] over predecessors u with dist=dist[v]-1.
    for v in order[1:]:
        r, c = v
        s = 0
        for dr, dc in _NBR_ORDER:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == EMPTY and dist[nr, nc] == dist[r, c] - 1:
                s += int(count[nr, nc])
        count[r, c] = s
    return dist, count


def canonical_shortest_path(grid: np.ndarray,
                            src: Tuple[int, int],
                            dst: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Deterministic shortest path from src to dst (or [] if unreachable).

    Reconstructed backwards from `dst`; ties broken by `_NBR_ORDER`.
    """
    dist = bfs_distances(grid, src)
    if dist[dst] < 0:
        return []
    H, W = grid.shape
    path = [dst]
    cur = dst
    while cur != src:
        r, c = cur
        for dr, dc in _NBR_ORDER:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == EMPTY and dist[nr, nc] == dist[r, c] - 1:
                path.append((nr, nc))
                cur = (nr, nc)
                break
    path.reverse()
    return path


def _structural_counts(grid: np.ndarray) -> Tuple[int, int]:
    """(num_junctions, num_dead_ends) over traversable cells."""
    junctions = dead_ends = 0
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] == EMPTY:
                n = len(_traversable_neighbors(grid, (r, c)))
                if n >= 3:
                    junctions += 1
                elif n == 1:
                    dead_ends += 1
    return junctions, dead_ends


def _path_overlap(path_a: List[Tuple[int, int]],
                  path_b: List[Tuple[int, int]]) -> float:
    """Jaccard overlap on cell-sets of two paths."""
    if not path_a or not path_b:
        return 0.0
    sa, sb = set(path_a), set(path_b)
    union = len(sa | sb)
    return (len(sa & sb) / union) if union > 0 else 0.0


def _switching_cost(grid: np.ndarray,
                    start: Tuple[int, int],
                    first_goal: Tuple[int, int],
                    other_goal: Tuple[int, int],
                    k: int) -> Optional[float]:
    """Penalty for committing to `first_goal` for k steps then switching to
    `other_goal`:

        cost = k' + d(pos_after_k, other) - d(start, other)

    where k' = min(k, len(path_to_first_goal)-1) so we don't overshoot the goal
    when the first path is shorter than k.
    """
    d_start_other = int(bfs_distances(grid, start)[other_goal])
    if d_start_other < 0:
        return None
    path = canonical_shortest_path(grid, start, first_goal)
    if not path:
        return None
    steps = min(k, len(path) - 1)
    pos = path[steps]
    d_pos_other = int(bfs_distances(grid, pos)[other_goal])
    if d_pos_other < 0:
        return None
    return float(steps + d_pos_other - d_start_other)


def compute_metrics(env: GridEnv, switching_k: int) -> Dict[str, float]:
    grid = env.grid
    E, P, R, B = env.ego_start, env.partner_start, env.red_goal, env.blue_goal

    d_E = bfs_distances(grid, E)
    d_P = bfs_distances(grid, P)
    e_r, e_b = int(d_E[R]), int(d_E[B])
    p_r, p_b = int(d_P[R]), int(d_P[B])

    _, cnt_E = bfs_path_counts(grid, E)
    _, cnt_P = bfs_path_counts(grid, P)

    p_ER = canonical_shortest_path(grid, E, R)
    p_EB = canonical_shortest_path(grid, E, B)
    p_PR = canonical_shortest_path(grid, P, R)
    p_PB = canonical_shortest_path(grid, P, B)

    C1, C2 = e_r + p_b, e_b + p_r

    junctions, dead_ends = _structural_counts(grid)
    realized = float((grid == WALL).sum() / grid.size)

    # Four switching-cost cases; drop unreachable ones (shouldn't happen for
    # accepted envs, but be defensive).
    sc = [c for c in (
        _switching_cost(grid, E, R, B, switching_k),
        _switching_cost(grid, E, B, R, switching_k),
        _switching_cost(grid, P, R, B, switching_k),
        _switching_cost(grid, P, B, R, switching_k),
    ) if c is not None]
    sc_mean = float(np.mean(sc)) if sc else 0.0
    sc_min = float(np.min(sc)) if sc else 0.0

    return {
        "ego_to_red":     e_r,
        "ego_to_blue":    e_b,
        "partner_to_red": p_r,
        "partner_to_blue": p_b,
        "sampled_wall_probability": float(env.wall_density),
        "realized_wall_density":    realized,
        "num_junctions": junctions,
        "num_dead_ends": dead_ends,
        "num_shortest_paths_ego_red":     int(cnt_E[R]),
        "num_shortest_paths_ego_blue":    int(cnt_E[B]),
        "num_shortest_paths_partner_red": int(cnt_P[R]),
        "num_shortest_paths_partner_blue": int(cnt_P[B]),
        "shortest_path_overlap_assignment_1": _path_overlap(p_ER, p_PB),
        "shortest_path_overlap_assignment_2": _path_overlap(p_EB, p_PR),
        "assignment_cost_1": C1,
        "assignment_cost_2": C2,
        "assignment_cost_difference": abs(C1 - C2),
        "switching_cost_mean": sc_mean,
        "switching_cost_min":  sc_min,
    }


# --------------------------------------------------------------------------- #
# Sampling                                                                    #
# --------------------------------------------------------------------------- #

def _sample_grid(rng: random.Random, grid_size: int,
                 wall_density: Sequence[float]) -> Tuple[np.ndarray, float]:
    wall_prob = rng.uniform(*wall_density)
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    for r in range(grid_size):
        for c in range(grid_size):
            if rng.random() < wall_prob:
                grid[r, c] = WALL
    return grid, wall_prob


def _sample_env(rng: random.Random,
                seed: int,
                grid_size: int,
                wall_density: Sequence[float],
                min_agent_sep: int,
                min_goal_sep: int,
                max_assignment_gap: int,
                min_switching_cost: float,
                geometric_pref_threshold: int,
                switching_k: int,
                max_attempts: int) -> GridEnv:
    """Rejection-sample one valid environment; attach metrics to `env.metadata`."""
    for _ in range(max_attempts):
        grid, wall_prob = _sample_grid(rng, grid_size, wall_density)

        empties = [(r, c) for r in range(grid_size) for c in range(grid_size)
                   if grid[r, c] == EMPTY]
        if len(empties) < 4:
            continue

        # Pick four distinct cells and assign roles uniformly — no positional prior.
        cells = rng.sample(empties, 4)
        rng.shuffle(cells)
        ego, partner, red, blue = cells

        env = GridEnv(grid=grid, ego_start=ego, partner_start=partner,
                      red_goal=red, blue_goal=blue, seed=seed,
                      wall_density=wall_prob)

        if not _all_pairs_reachable(env):
            continue

        # Reject trivial layouts — we want meaningful navigation and varied
        # shortest-path lengths.
        d_from_ego = bfs_distances(grid, ego)
        d_from_red = bfs_distances(grid, red)
        if d_from_ego[partner] < min_agent_sep:
            continue
        if d_from_red[blue] < min_goal_sep:
            continue

        # Compute coordination + structural metrics; use them for the remaining
        # filters and stash them for the JSON.
        m = compute_metrics(env, switching_k=switching_k)

        # 1. Competitive assignments: the two goal-assignments must have
        #    comparable total cost, or one of them dominates trivially.
        if m["assignment_cost_difference"] > max_assignment_gap:
            continue

        # 2. Nontrivial switching cost: miscoordination must actually hurt.
        if m["switching_cost_mean"] < min_switching_cost:
            continue

        # 3. No obvious geometric assignment: if the two agents strictly prefer
        #    opposite goals with a big margin on both sides, the layout gives
        #    away the coordination answer.
        delta_E = abs(m["ego_to_red"] - m["ego_to_blue"])
        delta_P = abs(m["partner_to_red"] - m["partner_to_blue"])
        ego_prefers_red = m["ego_to_red"] < m["ego_to_blue"]
        partner_prefers_red = m["partner_to_red"] < m["partner_to_blue"]
        opposite_preferences = ego_prefers_red != partner_prefers_red
        if opposite_preferences and delta_E >= geometric_pref_threshold and delta_P >= geometric_pref_threshold:
            continue

        env.metadata = m
        return env

    raise RuntimeError(f"Failed to sample a valid env in {max_attempts} attempts.")


def generate_envs(n: int,
                  master_seed: int,
                  grid_size: int,
                  wall_density: Sequence[float],
                  min_agent_sep: int,
                  min_goal_sep: int,
                  max_assignment_gap: int,
                  min_switching_cost: float,
                  geometric_pref_threshold: int,
                  switching_k: int,
                  max_attempts: int) -> List[GridEnv]:
    master_rng = random.Random(master_seed)
    envs: List[GridEnv] = []
    for _ in range(n):
        env_seed = master_rng.randrange(2**31)
        env_rng = random.Random(env_seed)
        envs.append(_sample_env(env_rng, seed=env_seed,
                                grid_size=grid_size,
                                wall_density=wall_density,
                                min_agent_sep=min_agent_sep,
                                min_goal_sep=min_goal_sep,
                                max_assignment_gap=max_assignment_gap,
                                min_switching_cost=min_switching_cost,
                                geometric_pref_threshold=geometric_pref_threshold,
                                switching_k=switching_k,
                                max_attempts=max_attempts))
    return envs


# --------------------------------------------------------------------------- #
# Encoding + I/O                                                              #
# --------------------------------------------------------------------------- #

def encode_env(env: GridEnv) -> np.ndarray:
    """Compile a GridEnv into a single (H, W) int8 array with cell codes 0-5."""
    coded = env.grid.astype(np.int8).copy()
    coded[env.red_goal] = RED
    coded[env.blue_goal] = BLUE
    coded[env.ego_start] = EGO
    coded[env.partner_start] = PARTNER
    return coded


def env_metadata(env: GridEnv, switching_k: int = 2) -> Dict[str, float]:
    """Return the metrics dict, computing it lazily if not already attached."""
    if env.metadata is None:
        env.metadata = compute_metrics(env, switching_k=switching_k)
    return env.metadata


# Match the deepest lists (no nested `[`/`]` inside) so we can collapse rows
# like a grid row or a coordinate pair onto a single line while keeping the
# surrounding structure pretty-printed.
_INNERMOST_LIST = re.compile(r"\[[^\[\]]+\]")


def _dumps_compact_arrays(obj: dict, indent: int = 2) -> str:
    """`json.dumps` with `indent`, but with the innermost arrays inlined.

    Example:
        [
          [1, 2, 3],
          [4, 5, 6]
        ]
    instead of every scalar on its own line.
    """
    text = json.dumps(obj, indent=indent)

    def _collapse(m: re.Match) -> str:
        items = [t.strip() for t in m.group(0)[1:-1].split(",") if t.strip()]
        return "[" + ", ".join(items) + "]"

    return _INNERMOST_LIST.sub(_collapse, text)


def env_to_json_dict(env: GridEnv, layout_id: str) -> dict:
    return {
        "layout_id": layout_id,
        "seed": int(env.seed),
        "grid": encode_env(env).tolist(),
        "ego_start": list(env.ego_start),
        "partner_start": list(env.partner_start),
        "red_goal": list(env.red_goal),
        "blue_goal": list(env.blue_goal),
        "metadata": env_metadata(env),
    }


def render_env(env: GridEnv, path: Path) -> None:
    grid_size = env.grid.shape[0]
    fig, ax = plt.subplots(figsize=(4, 4))

    cmap = ListedColormap(["#ffffff", "#333333"])
    ax.imshow(env.grid, cmap=cmap, vmin=0, vmax=1)
    for k in range(grid_size + 1):
        ax.axhline(k - 0.5, color="#bbbbbb", lw=0.5)
        ax.axvline(k - 0.5, color="#bbbbbb", lw=0.5)

    for (r, c), color, label in [(env.red_goal, "#e74c3c", "R"),
                                  (env.blue_goal, "#3498db", "B")]:
        ax.add_patch(Rectangle((c - 0.45, r - 0.45), 0.9, 0.9,
                                facecolor=color, edgecolor="black", lw=1.2))
        ax.text(c, r, label, ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")

    for (r, c), color, label in [(env.ego_start, "#f1c40f", "E"),
                                  (env.partner_start, "#2ecc71", "P")]:
        ax.add_patch(Circle((c, r), 0.35, facecolor=color,
                             edgecolor="black", lw=1.2))
        ax.text(c, r, label, ha="center", va="center",
                fontsize=10, fontweight="bold")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Split + save                                                                #
# --------------------------------------------------------------------------- #

def _split_indices(n: int, ratio: Sequence[float]) -> Dict[str, range]:
    assert len(ratio) == 3, "TRAIN_TEST_VAL_RATIO must have three entries"
    s = sum(ratio)
    if not np.isclose(s, 1.0):
        raise ValueError(f"train/test/val ratios must sum to 1.0, got {s}")

    # Split by order in generation; deterministic given `n`.
    n_train = int(round(n * ratio[0]))
    n_val   = int(round(n * ratio[2]))
    n_test  = n - n_train - n_val  # absorb rounding remainder here
    return {
        "train": range(0, n_train),
        "test":  range(n_train, n_train + n_test),
        "val":   range(n_train + n_test, n),
    }


def save_split(envs: List[GridEnv], split: str, indices: range,
               out_dir: Path) -> None:
    layouts_dir = out_dir / "layouts" / split
    renders_dir = out_dir / "renders" / split
    layouts_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)

    width = max(4, len(str(max(1, len(indices)))))
    compiled = []
    for local_i, global_i in enumerate(indices):
        env = envs[global_i]
        layout_id = f"{split}_{local_i:0{width}d}"

        # JSON.
        with open(layouts_dir / f"{layout_id}.json", "w") as f:
            f.write(_dumps_compact_arrays(env_to_json_dict(env, layout_id)))

        # Render.
        render_env(env, renders_dir / f"{layout_id}.png")

        compiled.append(encode_env(env))

    # Compiled npz for JAX training. Empty splits get an (0, H, W) array so
    # downstream code can still load without special-casing.
    if compiled:
        arr = np.stack(compiled, axis=0)
    else:
        grid_size = envs[0].grid.shape[0] if envs else 0
        arr = np.zeros((0, grid_size, grid_size), dtype=np.int8)
    np.savez(out_dir / f"{split}_layouts.npz", layouts=arr)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100,
                   help="Total number of environments to generate.")
    p.add_argument("--master_seed", type=int, default=0,
                   help="Master seed for reproducible generation.")
    p.add_argument("--grid_size", type=int, default=7,
                   help="Grid side length (square).")
    p.add_argument("--wall_density", type=float, nargs=2, default=[0.15, 0.60],
                   metavar=("MIN", "MAX"),
                   help="Range [min max] of per-cell wall probability, sampled "
                        "uniformly per env. Pass the same value twice to fix it.")
    p.add_argument("--train_test_val_ratio", type=float, nargs=3,
                   default=[0.70, 0.15, 0.15],
                   metavar=("TRAIN", "TEST", "VAL"),
                   help="Split ratios (must sum to 1.0).")
    p.add_argument("--min_agent_sep", type=int, default=3,
                   help="Reject layouts where the two agent starts are closer "
                        "than this many steps (shortest path).")
    p.add_argument("--min_goal_sep", type=int, default=3,
                   help="Reject layouts where the two goals are closer than "
                        "this many steps (shortest path).")
    p.add_argument("--max_assignment_gap", type=int, default=5,
                   help="Reject if |C1 - C2| exceeds this, where "
                        "C1 = d(E,R)+d(P,B) and C2 = d(E,B)+d(P,R). "
                        "Keeps the two goal assignments competitive.")
    p.add_argument("--min_switching_cost", type=float, default=1.0,
                   help="Reject if the mean switching cost across the four "
                        "commit-then-switch cases is below this. Ensures "
                        "miscoordination actually hurts.")
    p.add_argument("--geometric_pref_threshold", type=int, default=2,
                   help="Reject layouts with 'opposite geometric preferences' "
                        "where the two agents strictly prefer opposite goals "
                        "and both individual distance advantages meet or "
                        "exceed this threshold.")
    p.add_argument("--switching_k", type=int, default=2,
                   help="Number of committed steps before switching, used by "
                        "the switching-cost metric.")
    p.add_argument("--max_attempts", type=int, default=500,
                   help="Max rejection-sampling attempts per env.")
    p.add_argument("--out_dir", type=Path, default=Path("dev/grids"),
                   help="Output root directory.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    envs = generate_envs(n=args.n,
                         master_seed=args.master_seed,
                         grid_size=args.grid_size,
                         wall_density=args.wall_density,
                         min_agent_sep=args.min_agent_sep,
                         min_goal_sep=args.min_goal_sep,
                         max_assignment_gap=args.max_assignment_gap,
                         min_switching_cost=args.min_switching_cost,
                         geometric_pref_threshold=args.geometric_pref_threshold,
                         switching_k=args.switching_k,
                         max_attempts=args.max_attempts)

    splits = _split_indices(args.n, args.train_test_val_ratio)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, indices in splits.items():
        save_split(envs, split, indices, args.out_dir)
        print(f"[{split:5s}] wrote {len(indices)} envs -> {args.out_dir}")

    print(f"Done. Total: {args.n} envs across {list(splits)}.")


if __name__ == "__main__":
    main()
