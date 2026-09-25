"""Build the final experiment layout corpus.

Steps:
    1. Generate 2000 unique base 7x7 layouts using env_generator's current
       criteria and defaults.
    2. Apply exactly ONE D4 symmetry per base layout, balanced across the
       corpus so each of the 8 symmetries is used 250 times (total 2000).
    3. Deterministic shuffle with a fixed seed.
    4. Split 1600 train / 200 val / 200 test.
    5. Save JSON files, PNG renders, and a compiled per-split .npz to a new
       final-experiment directory (default: ``dev/grids_final``).
    6. Record the applied symmetry in each JSON's ``metadata.sym_applied``
       for traceability.
    7. Assert train/val/test are disjoint and all 2000 layouts are unique
       (up to grid identity + agent/goal positions).

NOTE ON AUGMENTATION AT TRAIN TIME
This is *one transform per base layout*, not the previous 8x on-the-fly
augmentation. The training config must set ``augment_symmetries: false``
for train, val, and test — otherwise ``CoordinationGrid`` will expand the
already-transformed 2000 into 16000, which is not what we want.

Usage:
    python dev/build_final_corpus.py \\
        --n_base 2000 \\
        --master_seed 2026 \\
        --out_dir dev/grids_final
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_generator import (
    GridEnv,
    _dumps_compact_arrays,
    compute_metrics,
    encode_env,
    env_to_json_dict,
    generate_envs,
    render_env,
)

from jaxmarl.environments.coordination_grid.coordination_grid import (
    _sym_position,
    _sym_wall,
    SYMMETRY_NAMES,
    N_SYMMETRIES,
)


# --------------------------------------------------------------------------- #
# Symmetry application                                                        #
# --------------------------------------------------------------------------- #

def _rc_to_xy(rc: Sequence[int]) -> np.ndarray:
    return np.array([int(rc[1]), int(rc[0])], dtype=np.int32)


def _xy_to_rc(xy: np.ndarray) -> Tuple[int, int]:
    return (int(xy[1]), int(xy[0]))


def apply_sym_to_env(env: GridEnv, g: int, switching_k: int = 2) -> GridEnv:
    """Return a new GridEnv obtained by applying D4 symmetry ``g`` to ``env``.

    Env-generator positions are stored as (row, col) tuples; coordination_grid
    symmetry helpers use (x, y). We convert both ways at the boundary.
    """
    n = env.grid.shape[0]
    new_wall = _sym_wall(g, env.grid.astype(np.bool_)).astype(env.grid.dtype)

    new_ego     = _xy_to_rc(_sym_position(g, _rc_to_xy(env.ego_start),     n))
    new_partner = _xy_to_rc(_sym_position(g, _rc_to_xy(env.partner_start), n))
    new_red     = _xy_to_rc(_sym_position(g, _rc_to_xy(env.red_goal),      n))
    new_blue    = _xy_to_rc(_sym_position(g, _rc_to_xy(env.blue_goal),     n))

    out = GridEnv(
        grid=new_wall,
        ego_start=new_ego,
        partner_start=new_partner,
        red_goal=new_red,
        blue_goal=new_blue,
        seed=env.seed,
        wall_density=env.wall_density,
        metadata=None,
    )
    # Re-compute metrics on the transformed geometry — most are invariant
    # under D4, but the field write is cheap and keeps the JSON honest.
    out.metadata = compute_metrics(out, switching_k=switching_k)
    return out


def _env_fingerprint(env: GridEnv) -> str:
    """Stable byte hash of (walls, ego, partner, red, blue) — the fields that
    define the game state. Used only for uniqueness assertions."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(env.grid).tobytes())
    h.update(np.array([env.ego_start, env.partner_start,
                       env.red_goal, env.blue_goal], dtype=np.int32).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# I/O                                                                         #
# --------------------------------------------------------------------------- #

def _save_env(env: GridEnv, sym_idx: int, layouts_dir: Path, renders_dir: Path,
              layout_id: str) -> None:
    d = env_to_json_dict(env, layout_id)
    d.setdefault("metadata", {})
    d["metadata"]["sym_applied_idx"] = int(sym_idx)
    d["metadata"]["sym_applied_name"] = SYMMETRY_NAMES[sym_idx]
    with open(layouts_dir / f"{layout_id}.json", "w") as f:
        f.write(_dumps_compact_arrays(d))
    render_env(env, renders_dir / f"{layout_id}.png")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_base", type=int, default=2000)
    p.add_argument("--master_seed", type=int, default=2026)
    p.add_argument("--shuffle_seed", type=int, default=2026)
    p.add_argument("--n_train", type=int, default=1600)
    p.add_argument("--n_val",   type=int, default=200)
    p.add_argument("--n_test",  type=int, default=200)
    p.add_argument("--out_dir", type=Path,
                   default=Path("/juice6/u/jshe/emergent_partner_grid/dev/grids_final"))
    p.add_argument("--grid_size",   type=int, default=7)
    p.add_argument("--wall_density", type=float, nargs=2, default=[0.15, 0.60])
    p.add_argument("--min_agent_sep",           type=int, default=3)
    p.add_argument("--min_goal_sep",            type=int, default=3)
    p.add_argument("--max_assignment_gap",      type=int, default=5)
    p.add_argument("--min_switching_cost",      type=float, default=1.0)
    p.add_argument("--geometric_pref_threshold", type=int, default=2)
    p.add_argument("--switching_k",             type=int, default=2)
    p.add_argument("--max_attempts",            type=int, default=500)
    p.add_argument("--skip_render", action="store_true",
                   help="Skip PNG rendering — much faster for a large corpus.")
    args = p.parse_args()

    assert args.n_train + args.n_val + args.n_test == args.n_base, (
        f"Split counts {args.n_train}+{args.n_val}+{args.n_test} != {args.n_base}"
    )
    assert args.n_base % N_SYMMETRIES == 0, (
        f"n_base={args.n_base} must be divisible by {N_SYMMETRIES} for balanced "
        f"D4 assignment (got {args.n_base % N_SYMMETRIES} remainder)"
    )
    per_sym = args.n_base // N_SYMMETRIES

    # -------- Step 1: 2000 unique base layouts --------
    print(f"[1/6] Generating {args.n_base} base layouts "
          f"(master_seed={args.master_seed}) ...")
    base_envs = generate_envs(
        n=args.n_base, master_seed=args.master_seed,
        grid_size=args.grid_size, wall_density=args.wall_density,
        min_agent_sep=args.min_agent_sep, min_goal_sep=args.min_goal_sep,
        max_assignment_gap=args.max_assignment_gap,
        min_switching_cost=args.min_switching_cost,
        geometric_pref_threshold=args.geometric_pref_threshold,
        switching_k=args.switching_k, max_attempts=args.max_attempts,
    )
    base_prints = [_env_fingerprint(e) for e in base_envs]
    n_unique_base = len(set(base_prints))
    print(f"       generated {len(base_envs)} envs; "
          f"{n_unique_base} unique fingerprints "
          f"({100.0 * n_unique_base / len(base_envs):.2f}%)")
    if n_unique_base != len(base_envs):
        print("       WARNING: some base envs are duplicates; keeping all "
              "(the D4 assignment / seed choice below is deterministic).")

    # -------- Step 2: balanced D4 assignment (250 per sym) --------
    print(f"[2/6] Assigning D4 symmetries — balanced, "
          f"{per_sym} of each of {N_SYMMETRIES} symmetries ...")
    # Balanced by construction: symmetry i is applied to base envs
    # [i*per_sym : (i+1)*per_sym]. Shuffle happens next.
    sym_assignment = np.repeat(np.arange(N_SYMMETRIES, dtype=np.int32), per_sym)
    assert sym_assignment.shape == (args.n_base,)
    print(f"       sym counts: {dict(Counter(sym_assignment.tolist()))}")

    print(f"[3/6] Applying symmetries ...")
    transformed_envs: List[GridEnv] = []
    for base_env, sym_idx in zip(base_envs, sym_assignment):
        transformed_envs.append(apply_sym_to_env(base_env, int(sym_idx),
                                                  switching_k=args.switching_k))

    trans_prints = [_env_fingerprint(e) for e in transformed_envs]
    n_unique_trans = len(set(trans_prints))
    print(f"       {n_unique_trans} unique transformed envs "
          f"({100.0 * n_unique_trans / len(transformed_envs):.2f}%)")
    assert n_unique_trans == len(transformed_envs), (
        f"transformed corpus has duplicates ({n_unique_trans} unique / "
        f"{len(transformed_envs)}) — cannot guarantee disjoint splits"
    )

    # -------- Step 3: deterministic shuffle --------
    print(f"[4/6] Deterministic shuffle (shuffle_seed={args.shuffle_seed}) ...")
    rng = np.random.default_rng(args.shuffle_seed)
    order = rng.permutation(args.n_base)
    shuffled_envs = [transformed_envs[i] for i in order]
    shuffled_syms = [int(sym_assignment[i]) for i in order]

    # -------- Step 4: split 1600 / 200 / 200 --------
    print(f"[5/6] Split -> train={args.n_train} val={args.n_val} "
          f"test={args.n_test} ...")
    idx_train = range(0, args.n_train)
    idx_val   = range(args.n_train, args.n_train + args.n_val)
    idx_test  = range(args.n_train + args.n_val,
                       args.n_train + args.n_val + args.n_test)
    split_ranges = {"train": idx_train, "val": idx_val, "test": idx_test}
    print(f"       train sym counts: {dict(Counter(shuffled_syms[i] for i in idx_train))}")
    print(f"       val   sym counts: {dict(Counter(shuffled_syms[i] for i in idx_val))}")
    print(f"       test  sym counts: {dict(Counter(shuffled_syms[i] for i in idx_test))}")

    # Uniqueness + disjointness assertions (post-split).
    fps = [_env_fingerprint(e) for e in shuffled_envs]
    fps_train = set(fps[i] for i in idx_train)
    fps_val   = set(fps[i] for i in idx_val)
    fps_test  = set(fps[i] for i in idx_test)
    assert len(fps_train) == args.n_train, \
        f"train has {len(fps_train)} unique / {args.n_train}"
    assert len(fps_val) == args.n_val
    assert len(fps_test) == args.n_test
    assert not (fps_train & fps_val), \
        f"train ∩ val = {len(fps_train & fps_val)} — not disjoint"
    assert not (fps_train & fps_test)
    assert not (fps_val & fps_test)
    assert len(set(fps)) == args.n_base
    print("       uniqueness + disjointness assertions ✓")

    # -------- Step 5: save --------
    print(f"[6/6] Saving to {args.out_dir} ...")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, rng_ in split_ranges.items():
        layouts_dir = args.out_dir / "layouts" / split
        renders_dir = args.out_dir / "renders" / split
        layouts_dir.mkdir(parents=True, exist_ok=True)
        if not args.skip_render:
            renders_dir.mkdir(parents=True, exist_ok=True)
        width = max(4, len(str(len(rng_))))
        compiled = []
        for local_i, global_i in enumerate(rng_):
            env = shuffled_envs[global_i]
            sym_idx = shuffled_syms[global_i]
            layout_id = f"{split}_{local_i:0{width}d}"
            d = env_to_json_dict(env, layout_id)
            d.setdefault("metadata", {})
            d["metadata"]["sym_applied_idx"] = int(sym_idx)
            d["metadata"]["sym_applied_name"] = SYMMETRY_NAMES[sym_idx]
            with open(layouts_dir / f"{layout_id}.json", "w") as f:
                f.write(_dumps_compact_arrays(d))
            if not args.skip_render:
                render_env(env, renders_dir / f"{layout_id}.png")
            compiled.append(encode_env(env))
        arr = np.stack(compiled, axis=0)
        np.savez(args.out_dir / f"{split}_layouts.npz", layouts=arr)
        print(f"       [{split:5s}] wrote {len(rng_)} envs")

    # -------- Manifest --------
    manifest = {
        "n_base":            args.n_base,
        "n_train":           args.n_train,
        "n_val":             args.n_val,
        "n_test":            args.n_test,
        "master_seed":       args.master_seed,
        "shuffle_seed":      args.shuffle_seed,
        "grid_size":         args.grid_size,
        "wall_density":      list(args.wall_density),
        "min_agent_sep":     args.min_agent_sep,
        "min_goal_sep":      args.min_goal_sep,
        "max_assignment_gap": args.max_assignment_gap,
        "min_switching_cost": args.min_switching_cost,
        "geometric_pref_threshold": args.geometric_pref_threshold,
        "switching_k":       args.switching_k,
        "max_attempts":      args.max_attempts,
        "d4_symmetries":     list(SYMMETRY_NAMES),
        "per_sym":           per_sym,
        "note":              (
            "One D4 symmetry applied per base layout (balanced 250 each). "
            "Training env MUST use augment_symmetries=false — the "
            "transformations are already baked in."
        ),
    }
    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Corpus at {args.out_dir}")
    print(f"  train/val/test = {args.n_train} / {args.n_val} / {args.n_test}")


if __name__ == "__main__":
    main()
