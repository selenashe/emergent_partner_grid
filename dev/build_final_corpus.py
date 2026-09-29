"""Build the final experiment layout corpus with capability-sensitive selection.

Pipeline (see ``dev/capability_selection.py`` for the science):

    1.  Generate ``--n_candidates`` base 7x7 layouts using
        ``env_generator.generate_envs`` (basic validity + geometric checks).
    2.  Deduplicate.
    3.  For each candidate compute ``LayoutStats`` over
        ``TRAINING_CAPABILITY_PAIRS`` under a generous evaluation horizon.
    4.  Filter on partner observability (Filter C).
    5.  Filter on capability-dependent allocation balance (Filter A).
    6.  Filter on oracle feasibility (near-full coverage).
    7.  Compute ``delta_reward`` distribution; keep everything at or above
        ``--min_delta_reward`` (or ``--delta_reward_quantile``).
    8.  Horizon-outlier pass: reject layouts whose per-cap worst-case oracle
        completion is at or beyond ``mean + n_sigma * std``.
    9.  Derive ``recommended_max_steps`` from the retained pool.
    10. Stratify the survivors on a small set of geometric + collaborative-
        pressure descriptors, and sample ``--n_final`` layouts.
    11. Apply balanced D4 symmetries (one per layout, ``n_final / 8`` each).
    12. Recompute BFS + LayoutStats on the transformed layout and assert
        selection invariance under D4 (BFS distances are exactly invariant;
        anything else is flagged).
    13. Deterministic shuffle + train/val/test split.
    14. Save JSONs, NPZs, renders, per-candidate audit CSV, diagnostic
        plots, and manifest.

Output goes to ``dev/grids_capability_selected`` by default so the previous
``dev/grids_final`` corpus is preserved for comparison. Nothing here
touches training or the environment.

Example usage:
    python dev/build_final_corpus.py \\
        --n_candidates 20000 --n_final 1600 \\
        --n_train 1280 --n_val 160 --n_test 160 \\
        --master_seed 2026
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))              # dev/
sys.path.insert(0, str(_THIS_DIR.parent))       # repo root, for `baselines.*`
from env_generator import (
    GridEnv,
    _dumps_compact_arrays,
    compute_metrics,
    encode_env,
    env_to_json_dict,
    generate_envs,
    render_env,
)
from capability_selection import (
    LayoutStats,
    derive_max_steps,
    evaluate_layout,
    horizon_outlier_mask,
    layout_bfs_distances,
    passes_allocation_balance,
    passes_delta_reward,
    passes_feasibility,
    passes_observability,
    stratified_sample,
    summarize_distribution,
    summarize_stats,
    worst_case_oracle_steps,
)

from jaxmarl.environments.coordination_grid.coordination_grid import (
    N_SYMMETRIES,
    SYMMETRY_NAMES,
    TRAINING_CAPABILITY_PAIRS,
    _sym_position,
    _sym_wall,
)

from baselines.IPPO.sweep_scheduler import (
    build_schedule,
    sanity_check_schedule,
)


# --------------------------------------------------------------------------- #
# Symmetry                                                                    #
# --------------------------------------------------------------------------- #

def _rc_to_xy(rc):
    return np.array([int(rc[1]), int(rc[0])], dtype=np.int32)


def _xy_to_rc(xy) -> Tuple[int, int]:
    return (int(xy[1]), int(xy[0]))


def apply_sym_to_env(env: GridEnv, g: int, switching_k: int = 2) -> GridEnv:
    n = env.grid.shape[0]
    new_wall = _sym_wall(g, env.grid.astype(np.bool_)).astype(env.grid.dtype)
    new_ego     = _xy_to_rc(_sym_position(g, _rc_to_xy(env.ego_start),     n))
    new_partner = _xy_to_rc(_sym_position(g, _rc_to_xy(env.partner_start), n))
    new_red     = _xy_to_rc(_sym_position(g, _rc_to_xy(env.red_goal),      n))
    new_blue    = _xy_to_rc(_sym_position(g, _rc_to_xy(env.blue_goal),     n))
    out = GridEnv(grid=new_wall, ego_start=new_ego, partner_start=new_partner,
                  red_goal=new_red, blue_goal=new_blue,
                  seed=env.seed, wall_density=env.wall_density, metadata=None)
    out.metadata = compute_metrics(out, switching_k=switching_k)
    return out


def _env_fingerprint(env: GridEnv) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(env.grid).tobytes())
    h.update(np.array([env.ego_start, env.partner_start,
                       env.red_goal, env.blue_goal], dtype=np.int32).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Diagnostics                                                                 #
# --------------------------------------------------------------------------- #

def _save_hist(values: Sequence[float], title: str, out_path: Path,
               bins: int = 40) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"       [warn] matplotlib unavailable; skipping {title}: {e}")
        return
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    fig, ax = plt.subplots(figsize=(4, 3))
    if arr.size > 0:
        ax.hist(arr, bins=bins, color="#4b78b8", edgecolor="black")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _write_candidate_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ----- corpus sizes -----
    p.add_argument("--n_candidates", type=int, default=20000,
                   help="Number of raw candidate layouts to generate.")
    p.add_argument("--n_final", type=int, default=1600,
                   help="Number of layouts in the final corpus (train+val+test). "
                        "Must be divisible by 8 for balanced D4 assignment.")
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--n_val",   type=int, default=None)
    p.add_argument("--n_test",  type=int, default=None)
    p.add_argument("--train_val_test_ratio", type=float, nargs=3,
                   default=[0.80, 0.10, 0.10],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Only used when explicit --n_{train,val,test} are unset.")

    # ----- seeds -----
    p.add_argument("--master_seed",  type=int, default=2026)
    p.add_argument("--shuffle_seed", type=int, default=2026)
    p.add_argument("--stratify_seed", type=int, default=2026)

    # ----- env-generator passthrough (basic validity) -----
    p.add_argument("--grid_size",   type=int, default=7)
    p.add_argument("--wall_density", type=float, nargs=2, default=[0.15, 0.60])
    p.add_argument("--min_agent_sep",            type=int,   default=3)
    p.add_argument("--min_goal_sep",             type=int,   default=3)
    p.add_argument("--max_assignment_gap",       type=int,   default=20,
                   help="Loose upper bound; capability-sensitive filters do the real work.")
    p.add_argument("--min_switching_cost",       type=float, default=0.0,
                   help="Disabled by default: capability filters supersede.")
    p.add_argument("--geometric_pref_threshold", type=int,   default=99,
                   help="Disabled by default: capability filters supersede.")
    p.add_argument("--switching_k",              type=int,   default=2)
    p.add_argument("--max_attempts",             type=int,   default=500)

    # ----- capability-sensitive selection -----
    p.add_argument("--eval_max_steps", type=int, default=500,
                   help="Horizon used when evaluating layouts. Should be large "
                        "enough that a well-formed layout is always oracle-feasible.")
    p.add_argument("--step_penalty",   type=float, default=0.01,
                   help="Matches CoordinationGrid default.")
    p.add_argument("--success_reward", type=float, default=1.0,
                   help="Matches CoordinationGrid default.")
    p.add_argument("--min_partner_goal_distance", type=int, default=3,
                   help="Partner must be at least this many BFS steps from each goal.")
    p.add_argument("--min_optimal_alloc_fraction", type=float, default=0.25)
    p.add_argument("--max_optimal_alloc_fraction", type=float, default=0.75)
    p.add_argument("--max_tie_fraction", type=float, default=0.5,
                   help="Reject layouts where more than this fraction of cap "
                        "pairs are ties (allocation balance is ill-defined).")
    p.add_argument("--min_oracle_success", type=float, default=0.99,
                   help="Fraction of cap pairs the oracle must complete within eval_max_steps.")
    p.add_argument("--min_delta_reward", type=float, default=None,
                   help="Absolute cutoff on oracle-vs-blind reward gap.")
    p.add_argument("--delta_reward_quantile", type=float, default=0.25,
                   help="Alternative to --min_delta_reward: keep layouts whose "
                        "delta_reward is at or above this quantile of the "
                        "post-Filter-A/C/feasibility survivor distribution.")

    # ----- horizon -----
    p.add_argument("--horizon_n_sigma", type=float, default=3.0)
    p.add_argument("--horizon_safety_margin", type=int, default=0,
                   help="Optional extra steps added to recommended_max_steps.")

    # ----- stratification -----
    p.add_argument("--n_bins_per_feature", type=int, default=3,
                   help="Number of quantile bins per stratification feature.")

    # ----- optional centroid balance filter -----
    p.add_argument("--centroid_p_opt_red_target", type=float, default=None,
                   help="If set, after all other filters keep only the n_final "
                        "layouts whose p_opt_red is closest to this target. "
                        "Bypasses stratified sampling. Typical value: 0.5.")

    # ----- I/O -----
    p.add_argument("--out_dir", type=Path,
                   default=Path("/juice6/u/jshe/emergent_partner_grid/dev/grids_capability_selected"))
    p.add_argument("--skip_render", action="store_true")
    p.add_argument("--skip_plots",  action="store_true")
    args = p.parse_args()

    # ---------------- resolve split sizes ----------------
    if args.n_train is None and args.n_val is None and args.n_test is None:
        r_train, r_val, r_test = args.train_val_test_ratio
        args.n_train = int(round(args.n_final * r_train))
        args.n_val   = int(round(args.n_final * r_val))
        args.n_test  = args.n_final - args.n_train - args.n_val
    if args.n_train is None or args.n_val is None or args.n_test is None:
        raise SystemExit("Provide either all three of --n_train/--n_val/--n_test or none.")
    if args.n_train + args.n_val + args.n_test != args.n_final:
        raise SystemExit(
            f"split counts {args.n_train}+{args.n_val}+{args.n_test} "
            f"!= --n_final={args.n_final}"
        )
    if args.n_final % N_SYMMETRIES != 0:
        raise SystemExit(
            f"--n_final={args.n_final} must be divisible by {N_SYMMETRIES} "
            f"for balanced D4 assignment"
        )
    per_sym = args.n_final // N_SYMMETRIES

    cap_pairs = list(TRAINING_CAPABILITY_PAIRS)
    print(f"[cfg] capability pool: {len(cap_pairs)} training pairs; "
          f"eval_max_steps={args.eval_max_steps}, "
          f"step_penalty={args.step_penalty}, success_reward={args.success_reward}")

    # ================================================================
    # Step 1: generate raw candidates
    # ================================================================
    print(f"\n[1/13] Generating {args.n_candidates} candidate layouts "
          f"(master_seed={args.master_seed}) ...")
    candidates = generate_envs(
        n=args.n_candidates, master_seed=args.master_seed,
        grid_size=args.grid_size, wall_density=args.wall_density,
        min_agent_sep=args.min_agent_sep, min_goal_sep=args.min_goal_sep,
        max_assignment_gap=args.max_assignment_gap,
        min_switching_cost=args.min_switching_cost,
        geometric_pref_threshold=args.geometric_pref_threshold,
        switching_k=args.switching_k, max_attempts=args.max_attempts,
    )
    n_generated = len(candidates)

    # ================================================================
    # Step 2: dedupe on fingerprint (post-basic-validity)
    # ================================================================
    print(f"[2/13] Deduplicating on grid fingerprint ...")
    seen = set()
    unique_candidates: List[GridEnv] = []
    for env in candidates:
        fp = _env_fingerprint(env)
        if fp in seen:
            continue
        seen.add(fp)
        unique_candidates.append(env)
    n_after_basic = len(unique_candidates)
    print(f"       {n_after_basic}/{n_generated} unique after basic validity")

    # ================================================================
    # Step 3: evaluate each candidate over TRAINING capability pool
    # ================================================================
    print(f"[3/13] Evaluating {n_after_basic} candidates over "
          f"{len(cap_pairs)} training capabilities ...")
    all_stats: List[LayoutStats] = []
    for env in unique_candidates:
        m = env.metadata or compute_metrics(env, switching_k=args.switching_k)
        stats = evaluate_layout(
            ego_to_red=int(m["ego_to_red"]),
            ego_to_blue=int(m["ego_to_blue"]),
            partner_to_red=int(m["partner_to_red"]),
            partner_to_blue=int(m["partner_to_blue"]),
            cap_pairs=cap_pairs,
            max_steps=args.eval_max_steps,
            step_penalty=args.step_penalty,
            success_reward=args.success_reward,
        )
        all_stats.append(stats)

    # ================================================================
    # Steps 4-6: apply Filters C (observability), A (alloc balance),
    #            and feasibility.
    # ================================================================
    print(f"[4/13] Filter C: observability "
          f"(partner_to_{{red,blue}} >= {args.min_partner_goal_distance}) ...")
    pass_obs = np.array([
        passes_observability(s, args.min_partner_goal_distance) for s in all_stats
    ], dtype=bool)
    n_after_obs = int(pass_obs.sum())
    print(f"       {n_after_obs}/{n_after_basic} pass observability")

    print(f"[5/13] Filter A: allocation balance "
          f"[{args.min_optimal_alloc_fraction}, {args.max_optimal_alloc_fraction}] "
          f"(max_tie_fraction={args.max_tie_fraction}) ...")
    pass_alloc = np.array([
        passes_allocation_balance(
            s,
            args.min_optimal_alloc_fraction,
            args.max_optimal_alloc_fraction,
            args.max_tie_fraction,
        ) for s in all_stats
    ], dtype=bool)
    n_after_alloc = int((pass_obs & pass_alloc).sum())
    print(f"       {n_after_alloc}/{n_after_basic} pass observability AND balance")

    print(f"[6/13] Feasibility filter: oracle_success_fraction >= "
          f"{args.min_oracle_success} ...")
    pass_feas = np.array([
        passes_feasibility(s, args.min_oracle_success) for s in all_stats
    ], dtype=bool)
    survive_abc = pass_obs & pass_alloc & pass_feas
    n_after_feas = int(survive_abc.sum())
    print(f"       {n_after_feas}/{n_after_basic} pass A + C + feasibility")

    # ================================================================
    # Step 7: delta_reward distribution -> Filter B threshold
    # ================================================================
    print(f"[7/13] Delta-reward distribution over {n_after_feas} "
          f"post-A/C/feasibility survivors ...")
    delta_vals = np.array([s.delta_reward for s in all_stats], dtype=np.float64)
    survivor_deltas = delta_vals[survive_abc]
    delta_dist_pre = summarize_distribution(survivor_deltas)
    print(f"       delta_reward distribution:")
    for k, v in delta_dist_pre.items():
        print(f"         {k:>6s}: {v}")

    if args.min_delta_reward is not None:
        delta_threshold = float(args.min_delta_reward)
        threshold_source = "absolute"
    else:
        q = float(args.delta_reward_quantile)
        if survivor_deltas.size == 0:
            delta_threshold = -np.inf
        else:
            delta_threshold = float(np.quantile(survivor_deltas, q))
        threshold_source = f"quantile={q}"
    print(f"       delta_reward threshold = {delta_threshold:.6f} "
          f"({threshold_source})")

    pass_delta = np.array([
        passes_delta_reward(s, delta_threshold) for s in all_stats
    ], dtype=bool)
    survive_abcd = survive_abc & pass_delta
    n_after_delta = int(survive_abcd.sum())
    print(f"       {n_after_delta}/{n_after_basic} pass Filter B (delta_reward)")

    if n_after_delta < args.n_final:
        raise SystemExit(
            f"Only {n_after_delta} survivors after Filter B, need >= "
            f"{args.n_final}. Increase --n_candidates or relax filters."
        )

    # ================================================================
    # Step 8: horizon-outlier pass on the survivor pool
    # ================================================================
    print(f"[8/13] Horizon-outlier pass "
          f"(reject worst_case_steps >= mean + {args.horizon_n_sigma}*std) ...")
    worst_case = np.array([
        worst_case_oracle_steps(all_stats[i]) if survive_abcd[i] else np.nan
        for i in range(n_after_basic)
    ], dtype=np.float64)
    wc_survivors = worst_case[survive_abcd]
    keep_wc, horizon_diag = horizon_outlier_mask(wc_survivors,
                                                 n_sigma=args.horizon_n_sigma)
    # Fold keep_wc back into a full-length mask.
    survivor_idx = np.flatnonzero(survive_abcd)
    pass_horizon = np.zeros(n_after_basic, dtype=bool)
    pass_horizon[survivor_idx[keep_wc]] = True
    survive_all = survive_abcd & pass_horizon
    n_after_horizon = int(survive_all.sum())
    print(f"       horizon pool: mu={horizon_diag['mean']:.2f} "
          f"sd={horizon_diag['std']:.2f} upper={horizon_diag['upper_bound']:.2f} "
          f"-> kept {horizon_diag['n_kept']}/{horizon_diag['n_input']}")

    if n_after_horizon < args.n_final:
        raise SystemExit(
            f"Only {n_after_horizon} survivors after horizon prune, need "
            f">= {args.n_final}. Increase --n_candidates or relax filters."
        )

    # ================================================================
    # Step 9: derive recommended_max_steps
    # ================================================================
    retained_wc = worst_case[survive_all]
    recommended_max_steps = derive_max_steps(retained_wc, args.horizon_safety_margin)
    max_observed_retained = int(np.nanmax(retained_wc))
    print(f"[9/13] max observed retained oracle completion = "
          f"{max_observed_retained}; recommended_max_steps = "
          f"{recommended_max_steps}")

    # ================================================================
    # Step 10: select n_final layouts (centroid filter OR stratified sample)
    # ================================================================
    survivor_indices = np.flatnonzero(survive_all)
    centroid_diag: Optional[Dict[str, float]] = None
    if args.centroid_p_opt_red_target is not None:
        target = float(args.centroid_p_opt_red_target)
        print(f"[10/13] Centroid filter: keeping the {args.n_final} survivors "
              f"whose p_opt_red is closest to {target} ...")
        survivor_p = np.array([all_stats[i].p_opt_red for i in survivor_indices],
                              dtype=np.float64)
        # Ascending distance to target. Ties broken by (delta_reward desc, index asc)
        # for full determinism.
        survivor_delta = np.array([all_stats[i].delta_reward for i in survivor_indices],
                                  dtype=np.float64)
        order_local = np.lexsort((survivor_indices,
                                  -survivor_delta,
                                  np.abs(survivor_p - target)))
        picked_local = order_local[:args.n_final]
        picked_global = survivor_indices[picked_local]
        picked_p = survivor_p[picked_local]
        centroid_diag = {
            "target": target,
            "n_input": int(survivor_indices.size),
            "n_selected": int(args.n_final),
            "p_opt_red_min": float(picked_p.min()),
            "p_opt_red_max": float(picked_p.max()),
            "p_opt_red_mean": float(picked_p.mean()),
            "p_opt_red_std": float(picked_p.std()),
            "max_abs_deviation": float(np.abs(picked_p - target).max()),
        }
        print(f"       selected {len(picked_global)} layouts; "
              f"p_opt_red in [{centroid_diag['p_opt_red_min']:.4f}, "
              f"{centroid_diag['p_opt_red_max']:.4f}] "
              f"(mean={centroid_diag['p_opt_red_mean']:.4f}, "
              f"std={centroid_diag['p_opt_red_std']:.4f})")
    else:
        print(f"[10/13] Stratified sampling {args.n_final} from "
              f"{n_after_horizon} survivors ...")
        feats = []
        for i in survivor_indices:
            env = unique_candidates[i]
            m = env.metadata
            s = all_stats[i]
            mean_sp = float(np.mean([m["ego_to_red"], m["ego_to_blue"],
                                     m["partner_to_red"], m["partner_to_blue"]]))
            mean_overlap = float(np.mean([m["shortest_path_overlap_assignment_1"],
                                          m["shortest_path_overlap_assignment_2"]]))
            feats.append([
                float(m["realized_wall_density"]),
                mean_sp,
                float(m["num_junctions"]),
                mean_overlap,
                float(s.delta_reward),
                float(s.p_opt_red),
            ])
        feat_mat = np.asarray(feats, dtype=np.float64)
        picked_local = stratified_sample(feat_mat, args.n_final,
                                         args.n_bins_per_feature, args.stratify_seed)
        picked_global = survivor_indices[picked_local]
        print(f"       selected {len(picked_global)} layouts")
    selected_envs = [unique_candidates[i] for i in picked_global]
    selected_stats = [all_stats[i] for i in picked_global]

    # ================================================================
    # Step 11: apply balanced D4 symmetries
    # ================================================================
    print(f"[11/13] Applying balanced D4 symmetries "
          f"({per_sym} of each of {N_SYMMETRIES}) ...")
    sym_assignment = np.repeat(np.arange(N_SYMMETRIES, dtype=np.int32), per_sym)
    assert sym_assignment.shape == (args.n_final,)
    transformed_envs: List[GridEnv] = []
    for base_env, sym_idx in zip(selected_envs, sym_assignment):
        transformed_envs.append(
            apply_sym_to_env(base_env, int(sym_idx), switching_k=args.switching_k)
        )

    # Re-evaluate LayoutStats on transformed geometry; assert D4 invariance
    # of the selection statistics (BFS-based, so equality is exact modulo
    # partial reachability of degenerate edge cases).
    print(f"       verifying D4 invariance of selection statistics ...")
    invariance_flags = []
    transformed_stats: List[LayoutStats] = []
    for base_env, t_env, s_base in zip(selected_envs, transformed_envs, selected_stats):
        m2 = t_env.metadata
        s2 = evaluate_layout(
            ego_to_red=int(m2["ego_to_red"]),
            ego_to_blue=int(m2["ego_to_blue"]),
            partner_to_red=int(m2["partner_to_red"]),
            partner_to_blue=int(m2["partner_to_blue"]),
            cap_pairs=cap_pairs,
            max_steps=args.eval_max_steps,
            step_penalty=args.step_penalty,
            success_reward=args.success_reward,
        )
        transformed_stats.append(s2)
        # Under D4, the four BFS distances are preserved exactly (see
        # coordination_grid._sym_position + _sym_wall); therefore every
        # scalar in LayoutStats should be equal within floating tolerance.
        # We only flag genuine mismatches (>1e-9 in reward, or any integer
        # count change) so the user is warned if anything drifts.
        flagged = False
        if s_base.n_opt_red != s2.n_opt_red or s_base.n_opt_blue != s2.n_opt_blue \
           or s_base.n_opt_ties != s2.n_opt_ties:
            flagged = True
        if abs(s_base.delta_reward - s2.delta_reward) > 1e-9:
            flagged = True
        if s_base.partner_to_red != s2.partner_to_red \
           or s_base.partner_to_blue != s2.partner_to_blue:
            flagged = True
        invariance_flags.append(flagged)
    n_flagged = int(sum(invariance_flags))
    if n_flagged:
        print(f"       [warn] {n_flagged}/{len(invariance_flags)} layouts show "
              f"D4-nontrivial selection statistics — inspect tie-breaking")
    else:
        print(f"       ✓ D4 invariance holds for all {len(invariance_flags)} layouts")

    trans_prints = [_env_fingerprint(e) for e in transformed_envs]
    n_unique_trans = len(set(trans_prints))
    assert n_unique_trans == len(transformed_envs), (
        f"transformed corpus has duplicates ({n_unique_trans} unique / "
        f"{len(transformed_envs)}) — cannot guarantee disjoint splits"
    )

    # ================================================================
    # Step 12: deterministic shuffle + split
    # ================================================================
    print(f"[12/13] Shuffle (shuffle_seed={args.shuffle_seed}) and split ...")
    rng = np.random.default_rng(args.shuffle_seed)
    order = rng.permutation(args.n_final)
    shuffled_envs  = [transformed_envs[i] for i in order]
    shuffled_stats = [transformed_stats[i] for i in order]
    shuffled_syms  = [int(sym_assignment[i]) for i in order]

    idx_train = range(0, args.n_train)
    idx_val   = range(args.n_train, args.n_train + args.n_val)
    idx_test  = range(args.n_train + args.n_val, args.n_final)
    split_ranges = {name: rng_ for name, rng_ in
                    (("train", idx_train), ("val", idx_val), ("test", idx_test))
                    if len(rng_) > 0}
    for name, rng_ in split_ranges.items():
        print(f"       {name:5s} sym counts: "
              f"{dict(Counter(shuffled_syms[i] for i in rng_))}")
    fps = [_env_fingerprint(e) for e in shuffled_envs]
    fps_train = set(fps[i] for i in idx_train)
    fps_val   = set(fps[i] for i in idx_val)
    fps_test  = set(fps[i] for i in idx_test)
    assert len(fps_train) == args.n_train
    assert len(fps_val)   == args.n_val
    assert len(fps_test)  == args.n_test
    assert not (fps_train & fps_val)
    assert not (fps_train & fps_test)
    assert not (fps_val   & fps_test)
    print("       ✓ uniqueness + disjointness")

    # ================================================================
    # Step 13: save
    # ================================================================
    print(f"[13/13] Writing corpus + diagnostics to {args.out_dir} ...")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = args.out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    for split, indices in split_ranges.items():
        if len(indices) == 0:
            continue
        layouts_dir = args.out_dir / "layouts" / split
        renders_dir = args.out_dir / "renders" / split
        layouts_dir.mkdir(parents=True, exist_ok=True)
        if not args.skip_render:
            renders_dir.mkdir(parents=True, exist_ok=True)
        width = max(4, len(str(len(indices))))
        compiled = []
        for local_i, global_i in enumerate(indices):
            env  = shuffled_envs[global_i]
            stat = shuffled_stats[global_i]
            sym_idx = shuffled_syms[global_i]
            layout_id = f"{split}_{local_i:0{width}d}"
            d = env_to_json_dict(env, layout_id)
            d.setdefault("metadata", {})
            d["metadata"]["sym_applied_idx"]  = int(sym_idx)
            d["metadata"]["sym_applied_name"] = SYMMETRY_NAMES[sym_idx]
            # Attach the capability-sensitive selection metrics.
            d["metadata"]["selection"] = stat.to_metadata_dict()
            d["metadata"]["selection_pass_observability"]      = True
            d["metadata"]["selection_pass_allocation_balance"] = True
            d["metadata"]["selection_pass_feasibility"]        = True
            d["metadata"]["selection_pass_delta_reward"]       = True
            d["metadata"]["selection_pass_horizon"]            = True
            with open(layouts_dir / f"{layout_id}.json", "w") as f:
                f.write(_dumps_compact_arrays(d))
            if not args.skip_render:
                render_env(env, renders_dir / f"{layout_id}.png")
            compiled.append(encode_env(env))
        arr = np.stack(compiled, axis=0)
        np.savez(args.out_dir / f"{split}_layouts.npz", layouts=arr)
        print(f"       [{split:5s}] wrote {len(indices)} envs")

    # ---- per-candidate audit CSV ----
    audit_rows: List[Dict[str, object]] = []
    for i, env in enumerate(unique_candidates):
        s = all_stats[i]
        audit_rows.append({
            "index": i,
            "seed": int(env.seed),
            "fingerprint": _env_fingerprint(env),
            "p_opt_red": s.p_opt_red,
            "n_opt_red": s.n_opt_red,
            "n_opt_blue": s.n_opt_blue,
            "n_opt_ties": s.n_opt_ties,
            "fraction_ties": s.fraction_ties,
            "delta_reward": s.delta_reward,
            "oracle_expected_reward": s.oracle_expected_reward,
            "best_fixed_expected_reward": s.best_fixed_expected_reward,
            "oracle_max_completion": s.oracle_max_completion,
            "oracle_success_fraction": s.oracle_success_fraction,
            "partner_to_red": s.partner_to_red,
            "partner_to_blue": s.partner_to_blue,
            "pass_observability":      bool(pass_obs[i]),
            "pass_allocation_balance": bool(pass_alloc[i]),
            "pass_feasibility":        bool(pass_feas[i]),
            "pass_delta_reward":       bool(pass_delta[i]),
            "pass_horizon":            bool(pass_horizon[i]),
            "selected_final":          bool(i in set(picked_global.tolist())),
        })
    _write_candidate_csv(audit_rows, diag_dir / "candidates.csv")

    # ---- diagnostic plots ----
    if not args.skip_plots:
        # Selected-layout distributions.
        _save_hist([s.p_opt_red for s in selected_stats],
                   "p_opt_red (selected)", diag_dir / "hist_p_opt_red.png")
        _save_hist([s.delta_reward for s in selected_stats],
                   "delta_reward (selected)", diag_dir / "hist_delta_reward.png")
        _save_hist([s.oracle_mean_completion for s in selected_stats],
                   "oracle_mean_completion (selected)",
                   diag_dir / "hist_oracle_mean_completion.png")
        _save_hist([s.oracle_max_completion for s in selected_stats],
                   "oracle_max_completion (selected)",
                   diag_dir / "hist_oracle_max_completion.png")
        _save_hist([s.partner_to_red for s in selected_stats],
                   "partner_to_red (selected)",
                   diag_dir / "hist_partner_to_red.png")
        _save_hist([s.partner_to_blue for s in selected_stats],
                   "partner_to_blue (selected)",
                   diag_dir / "hist_partner_to_blue.png")
        _save_hist([e.metadata["realized_wall_density"] for e in selected_envs],
                   "wall density (selected)",
                   diag_dir / "hist_wall_density.png")
        _save_hist([e.metadata["num_junctions"] for e in selected_envs],
                   "num_junctions (selected)",
                   diag_dir / "hist_num_junctions.png")
        _save_hist([e.metadata["assignment_cost_difference"] for e in selected_envs],
                   "assignment_cost_difference (selected)",
                   diag_dir / "hist_assignment_cost_diff.png")

    # ---- rejection table ----
    reject_lines = [
        f"generated                          {n_generated}",
        f"unique (basic validity)            {n_after_basic}",
        f"pass observability                 {int(pass_obs.sum())}",
        f"pass allocation balance            {int(pass_alloc.sum())}",
        f"pass feasibility                   {int(pass_feas.sum())}",
        f"survive A+C+feasibility            {n_after_feas}",
        f"pass delta_reward filter           {int(pass_delta.sum())}",
        f"survive A+C+feas+delta             {n_after_delta}",
        f">= mean+{args.horizon_n_sigma}SD horizon outliers   "
        f"{n_after_delta - n_after_horizon}",
        f"eligible after all filters         {n_after_horizon}",
        f"selected final                     {args.n_final}",
    ]
    print("\n=== rejection table ===")
    for line in reject_lines:
        print("  " + line)
    with open(diag_dir / "rejection_table.txt", "w") as f:
        f.write("\n".join(reject_lines) + "\n")

    # ---- manifest ----
    manifest = {
        "n_candidates_generated":      n_generated,
        "n_after_basic_validity":      n_after_basic,
        "n_after_observability":       int(pass_obs.sum()),
        "n_after_allocation_balance":  n_after_alloc,
        "n_after_feasibility":         n_after_feas,
        "n_after_delta_filter":        n_after_delta,
        "n_after_horizon_filter":      n_after_horizon,
        "n_final":                     args.n_final,
        "n_train":                     args.n_train,
        "n_val":                       args.n_val,
        "n_test":                      args.n_test,
        "seeds": {
            "master_seed":   args.master_seed,
            "shuffle_seed":  args.shuffle_seed,
            "stratify_seed": args.stratify_seed,
        },
        "thresholds": {
            "min_partner_goal_distance":    args.min_partner_goal_distance,
            "min_optimal_alloc_fraction":   args.min_optimal_alloc_fraction,
            "max_optimal_alloc_fraction":   args.max_optimal_alloc_fraction,
            "max_tie_fraction":             args.max_tie_fraction,
            "min_oracle_success":           args.min_oracle_success,
            "delta_reward_threshold":       delta_threshold,
            "delta_reward_threshold_source": threshold_source,
            "horizon_n_sigma":              args.horizon_n_sigma,
            "horizon_safety_margin":        args.horizon_safety_margin,
        },
        "eval_reward_config": {
            "eval_max_steps":  args.eval_max_steps,
            "step_penalty":    args.step_penalty,
            "success_reward":  args.success_reward,
        },
        "centroid_p_opt_red":       centroid_diag,
        "horizon_pre_prune":        horizon_diag,
        "delta_reward_distribution_survivors_pre_delta_filter": delta_dist_pre,
        "post_horizon_completion":  summarize_distribution(retained_wc.tolist()),
        "recommended_max_steps":    int(recommended_max_steps),
        "max_observed_retained_oracle_completion": int(max_observed_retained),
        "selected_layout_stats":    summarize_stats(selected_stats),
        "d4_symmetries":            list(SYMMETRY_NAMES),
        "per_sym":                  per_sym,
        "d4_invariance_flagged":    int(sum(invariance_flags)),
        "capability_pool":          [list(map(int, c)) for c in cap_pairs],
        "env_generator": {
            "grid_size":                 args.grid_size,
            "wall_density":              list(args.wall_density),
            "min_agent_sep":             args.min_agent_sep,
            "min_goal_sep":              args.min_goal_sep,
            "max_assignment_gap":        args.max_assignment_gap,
            "min_switching_cost":        args.min_switching_cost,
            "geometric_pref_threshold":  args.geometric_pref_threshold,
            "switching_k":               args.switching_k,
            "max_attempts":              args.max_attempts,
        },
        "reproduce_command": _reproduce_command(args),
    }
    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- sweep-schedule sanity check on the new corpus ----
    if args.n_train > 0:
        print("\n=== sweep-schedule sanity check ===")
        # Pick the largest divisor of n_train that is <= 20 so we get a
        # reasonable rounds_per_episode without failing on odd sizes.
        rpe = next((k for k in (20, 10, 8, 5, 4, 2, 1) if args.n_train % k == 0), 1)
        sched = build_schedule(
            partner_capability_pairs=cap_pairs,
            n_layouts_train=args.n_train,
            rounds_per_episode=rpe,
            n_sweeps=1,
            seed=args.master_seed,
        )
        sanity_check_schedule(sched)

    print(f"\nDone. Corpus at {args.out_dir}")
    print(f"  train/val/test = {args.n_train} / {args.n_val} / {args.n_test}")
    print(f"  recommended_max_steps = {recommended_max_steps}  "
          f"(max observed retained = {max_observed_retained})")
    print(f"  reproduce: {manifest['reproduce_command']}")


def _reproduce_command(args) -> str:
    parts = ["python dev/build_final_corpus.py"]
    for k, v in vars(args).items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                parts.append(f"--{k}")
        elif isinstance(v, list):
            parts.append(f"--{k} " + " ".join(str(x) for x in v))
        else:
            parts.append(f"--{k} {v}")
    return " \\\n    ".join(parts)


if __name__ == "__main__":
    main()
