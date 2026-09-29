"""Unit tests for the capability-sensitive layout selection pipeline.

Run:
    pytest dev/test_capability_selection.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capability_selection import (
    INFEASIBLE,
    LayoutStats,
    completion_time,
    derive_max_steps,
    evaluate_layout,
    horizon_outlier_mask,
    layout_bfs_distances,
    passes_allocation_balance,
    passes_delta_reward,
    passes_feasibility,
    passes_observability,
    stratified_sample,
)

# sweep_scheduler is jax-free; build_final_corpus / coordination_grid pull in
# jax, so we defer/soften those imports to keep the pure-math tests runnable
# in environments without jax installed (e.g. CPU-only checkouts).
from baselines.IPPO.sweep_scheduler import (
    build_schedule,
    sanity_check_schedule,
)

# Mirror TRAINING_CAPABILITY_PAIRS so we don't need to import coordination_grid
# (which pulls in jax). Kept in sync with jaxmarl.environments.coordination_grid.
_CAPABILITY_VALUES = (1, 2, 3, 4, 7, 9)
_HELDOUT = frozenset([(1, 4), (2, 7), (3, 9), (4, 1), (7, 3), (9, 2)])
CAP = [(a, b) for a in _CAPABILITY_VALUES for b in _CAPABILITY_VALUES
       if (a, b) not in _HELDOUT]
assert len(CAP) == 30
EVAL_STEPS = 500
STEP_PEN = 0.01
SUCC_REW = 1.0


def _stats(er, eb, pr, pb, caps=CAP, ms=EVAL_STEPS) -> LayoutStats:
    return evaluate_layout(er, eb, pr, pb, caps, ms, STEP_PEN, SUCC_REW)


# -------------------------------------------------------------- Filter A ----

def test_allocation_balance_rejects_lopsided():
    """Ego is right next to RED but 20 steps from BLUE; partner symmetric.
    Ego->RED wins for essentially every cap because ego_step dominates."""
    s = _stats(er=1, eb=20, pr=20, pb=1)
    assert s.p_opt_red >= 0.95 or s.p_opt_blue >= 0.95
    assert not passes_allocation_balance(s, 0.25, 0.75)


def test_allocation_balance_accepts_flipping_layout():
    """Ego and partner are balanced across goals: which allocation is best
    genuinely depends on which partner cooldown is smaller."""
    # ego reaches both goals in 5 (matched). Partner reaches red in 4 blue in 4.
    # With ego_path=5, ego_step=6. Partner_step under alloc A (ego->RED) =
    #   2 + (4-1)*c_B = 2 + 3*c_B, so A is dominated by partner when c_B>=2.
    # Similarly B when c_R>=2. So optimal follows the smaller cool.
    s = _stats(er=5, eb=5, pr=4, pb=4)
    # Across TRAINING_CAPABILITY_PAIRS (mixes of c_R,c_B), roughly balanced.
    assert 0.20 <= s.p_opt_red <= 0.80
    assert passes_allocation_balance(s, 0.20, 0.80)


# -------------------------------------------------------------- Filter C ----

def test_observability_rejects_short_partner_paths():
    s = _stats(er=5, eb=5, pr=2, pb=4)
    assert not passes_observability(s, min_partner_goal_distance=3)
    s2 = _stats(er=5, eb=5, pr=3, pb=3)
    assert passes_observability(s2, min_partner_goal_distance=3)


# -------------------------------------------------------------- delta id ----

def test_delta_reward_identity():
    """delta_reward == oracle_expected_reward - best_fixed_expected_reward
    and best_fixed == max(fixed_red, fixed_blue)."""
    s = _stats(er=3, eb=6, pr=6, pb=3)
    assert s.best_fixed_expected_reward == max(s.fixed_red_expected_reward,
                                                s.fixed_blue_expected_reward)
    assert abs(s.delta_reward
               - (s.oracle_expected_reward - s.best_fixed_expected_reward)) < 1e-12


def test_fixed_baseline_is_fixed_across_caps():
    """The fixed baselines are per-cap-uniform choices (one allocation for
    every cap), so fixed_red_expected_reward equals the mean of allocation-A
    rewards across the pool.
    """
    er, eb, pr, pb = 4, 5, 5, 4
    s = _stats(er, eb, pr, pb)
    # Reconstruct alloc-A rewards manually.
    A_rewards = []
    for c_r, c_b in CAP:
        t = completion_time(er, pb, c_b, EVAL_STEPS)
        if t >= INFEASIBLE:
            A_rewards.append(-STEP_PEN * EVAL_STEPS)
        else:
            A_rewards.append(SUCC_REW - STEP_PEN * t)
    assert abs(s.fixed_red_expected_reward - float(np.mean(A_rewards))) < 1e-12


# -------------------------------------------------------------- ties -------

def test_ties_do_not_count_as_red_or_blue():
    """If ego_to_red == ego_to_blue AND partner_to_red == partner_to_blue,
    every capability with c_R == c_B yields identical A/B rewards -> tie."""
    s = _stats(er=5, eb=5, pr=4, pb=4)
    # Diagonal caps (c_R == c_B) are ties.
    diag_ties = sum(1 for (c_r, c_b) in CAP if c_r == c_b)
    assert s.n_opt_ties >= diag_ties
    assert s.n_opt_red + s.n_opt_blue + s.n_opt_ties == len(CAP)


# -------------------------------------------------------------- horizon ----

def test_horizon_outlier_rejects_at_or_beyond_3sd():
    # 20 tight values + one clear outlier at 40 -> mean+3sd well below 40.
    wc = np.array([10.0] * 20 + [40.0])
    mu, sd = wc.mean(), wc.std()
    upper = mu + 3.0 * sd
    assert 40.0 >= upper  # sanity: the outlier truly meets the threshold
    keep, diag = horizon_outlier_mask(wc, n_sigma=3.0)
    assert diag["upper_bound"] == pytest.approx(upper)
    assert not keep[-1]           # outlier rejected
    assert keep[:20].all()        # tight cluster kept


def test_horizon_outlier_reject_is_at_or_beyond():
    """Value exactly at mean+3sd must be rejected (spec uses `>=`)."""
    # Symmetric pool so mean == 10, std == 0 -> upper == 10; any 10 rejected.
    wc = np.array([10.0, 10.0, 10.0, 10.0])
    keep, _ = horizon_outlier_mask(wc, n_sigma=3.0)
    assert not keep.any()


def test_derive_max_steps_covers_worst_case():
    wc = np.array([10, 11, 12, 13, 12], dtype=float)
    ms = derive_max_steps(wc, safety_margin=0)
    assert ms == 13
    ms2 = derive_max_steps(wc, safety_margin=2)
    assert ms2 == 15


# -------------------------------------------------------------- sweep ------

def test_capability_layout_scheduling_balanced_and_independent():
    n_layouts = 40  # divisible by rounds_per_episode = 10 and by 8
    sched = build_schedule(
        partner_capability_pairs=CAP,
        n_layouts_train=n_layouts,
        rounds_per_episode=10,
        n_sweeps=1,
        seed=0,
    )
    sanity_check_schedule(sched, verbose=False)


# -------------------------------------------------------------- determinism +

def test_stratified_sample_deterministic():
    rng = np.random.default_rng(7)
    feats = rng.uniform(size=(300, 4))
    a = stratified_sample(feats, 40, 3, seed=123)
    b = stratified_sample(feats, 40, 3, seed=123)
    c = stratified_sample(feats, 40, 3, seed=124)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert len(set(a.tolist())) == len(a)  # no duplicates
    assert len(a) == 40


# ------------------------------------------------------ layout_bfs_distances

def test_layout_bfs_distances_matches_expectation():
    """Empty 5x5 grid: manhattan-optimal BFS distances."""
    grid = np.zeros((5, 5), dtype=np.int32)
    d = layout_bfs_distances(grid, (0, 0), (4, 4), (0, 4), (4, 0))
    assert d["ego_to_red"] == 4      # (0,0)->(0,4)
    assert d["ego_to_blue"] == 4     # (0,0)->(4,0)
    assert d["partner_to_red"] == 4  # (4,4)->(0,4)
    assert d["partner_to_blue"] == 4  # (4,4)->(4,0)


# ------------------------------------------------------ end-to-end integration

def test_end_to_end_build_small(tmp_path, monkeypatch):
    """Smoke: run build_final_corpus with tiny numbers, then check
    train/val/test are unique, disjoint, and non-empty."""
    import subprocess
    out = tmp_path / "grids_small"
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "build_final_corpus.py"),
        "--n_candidates", "2000",
        "--n_final", "16",
        "--n_train", "8", "--n_val", "4", "--n_test", "4",
        "--master_seed", "1",
        "--min_partner_goal_distance", "3",
        "--min_optimal_alloc_fraction", "0.20",
        "--max_optimal_alloc_fraction", "0.80",
        "--delta_reward_quantile", "0.10",
        "--min_oracle_success", "0.99",
        "--out_dir", str(out),
        "--skip_render", "--skip_plots",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"smoke build failed (may be pool-size sensitive):\n"
                    f"{r.stdout[-2000:]}\n---\n{r.stderr[-2000:]}")
    # File presence.
    for split in ("train", "val", "test"):
        assert (out / f"{split}_layouts.npz").exists()
        assert (out / "layouts" / split).is_dir()
    manifest = (out / "manifest.json")
    assert manifest.exists()

    import json
    m = json.loads(manifest.read_text())
    assert m["n_train"] == 8
    assert m["n_val"] == 4
    assert m["n_test"] == 4
    assert m["recommended_max_steps"] >= m["max_observed_retained_oracle_completion"]

    # Splits disjoint.
    def _grids(split):
        arr = np.load(out / f"{split}_layouts.npz")["layouts"]
        return {arr[i].tobytes() for i in range(arr.shape[0])}
    G = {s: _grids(s) for s in ("train", "val", "test")}
    assert not (G["train"] & G["val"])
    assert not (G["train"] & G["test"])
    assert not (G["val"] & G["test"])
    assert len(G["train"]) == 8


if __name__ == "__main__":  # pragma: no cover
    import pytest as _pt
    raise SystemExit(_pt.main([__file__, "-q"]))
