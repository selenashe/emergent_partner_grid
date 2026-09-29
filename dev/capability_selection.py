"""Capability-sensitive layout selection.

Given a pool of candidate 7x7 grid layouts, analytically evaluate every
layout against a fixed pool of capability pairs (typically
``TRAINING_CAPABILITY_PAIRS``) and produce a filtered, stratified,
horizon-controlled final corpus.

The scientific selection criteria are:

1. **Observability** — the partner is far enough from both goals that
   inter-move timing is observable to the ego.
2. **Allocation balance** — the *optimal* ego allocation genuinely
   depends on partner capability. Non-tied optimum for RED must live in
   ``[min_frac, max_frac]`` of the training capability pool.
3. **Feasibility** — the analytical oracle can complete essentially
   every training capability pair under a generous evaluation horizon.
4. **Delta reward** — expected reward under an oracle-of-capability
   policy meaningfully exceeds the best fixed partner-blind allocation.
5. **Horizon** — layouts whose worst-case oracle completion time is at
   or beyond ``mean + 3*std`` across the current survivor pool are
   dropped, and a global ``recommended_max_steps`` is derived from what
   remains.

All completion-time and reward math routes through :func:`completion_time`
and :func:`evaluate_layout` — the analytical completion formula lives in
exactly one place. ``dev/capability_validation.py`` re-uses these
helpers.

Public API used by ``dev/build_final_corpus.py``:

    * :class:`LayoutStats`
    * :func:`layout_bfs_distances`
    * :func:`evaluate_layout`
    * :func:`layout_stats`
    * :func:`passes_observability`
    * :func:`passes_allocation_balance`
    * :func:`passes_feasibility`
    * :func:`passes_delta_reward`
    * :func:`derive_max_steps`
    * :func:`stratified_sample`
    * :func:`summarize_stats`
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# NOTE: we DELIBERATELY do not import from env_generator here — this
# module must be usable both during generation (where each candidate
# already has BFS metadata) and post-hoc from a saved layout JSON.


# --------------------------------------------------------------------------- #
# Completion-time / reward primitives                                          #
# --------------------------------------------------------------------------- #

INFEASIBLE = 10 ** 9  # sentinel value returned by completion_time on timeout


def completion_time(ego_path: int, partner_path: int, partner_cool: int,
                    max_steps: int) -> int:
    """Analytical joint-completion time in env steps.

    Assumes shortest-path navigation with no agent collisions. Ego moves
    once every step starting at t=1 (total = 1 + ego_path). Partner
    moves at t=2, 2+c, 2+2c, ..., so the k-th move happens at step
    ``2 + (k-1)*c`` and reaching a goal requires ``partner_path`` moves.
    Both must occupy their assigned goal on the SAME env step, so the
    joint completion time is the max of the two.

    Returns :data:`INFEASIBLE` if either path is unreachable (``< 0``)
    or if the joint completion exceeds ``max_steps``.
    """
    if ego_path < 0 or partner_path < 0:
        return INFEASIBLE
    ego_step = 1 + ego_path
    partner_step = 2 + (partner_path - 1) * partner_cool
    completed = max(ego_step, partner_step)
    if completed > max_steps:
        return INFEASIBLE
    return completed


def _reward_from_time(completion: int,
                      max_steps: int,
                      step_penalty: float,
                      success_reward: float) -> float:
    """Env-consistent expected reward from a scalar completion time.

    Matches ``CoordinationGrid``: success grants ``success_reward`` minus
    ``step_penalty`` per env step; a timed-out episode grants no success
    reward and pays ``step_penalty * max_steps``.
    """
    if completion >= INFEASIBLE:
        return -step_penalty * float(max_steps)
    return float(success_reward) - float(step_penalty) * float(completion)


def layout_bfs_distances(grid: np.ndarray,
                         ego_rc: Tuple[int, int],
                         partner_rc: Tuple[int, int],
                         red_rc: Tuple[int, int],
                         blue_rc: Tuple[int, int]) -> Dict[str, int]:
    """BFS distances (`ego_to_red`, ...) for a layout with walls==1.

    Returns the same four keys used by :class:`LayoutStats` inputs. This
    is used when loading a JSON layout that doesn't already carry
    metadata (typical for held-out corpora); ``env_generator.compute_metrics``
    covers the in-pipeline case.
    """
    from collections import deque

    H, W = grid.shape

    def bfs(src: Tuple[int, int]) -> np.ndarray:
        dist = -np.ones((H, W), dtype=np.int32)
        if grid[src] != 0:
            return dist
        dist[src] = 0
        q = deque([src])
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0 and dist[nr, nc] == -1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
        return dist

    d_ego = bfs(tuple(ego_rc))
    d_part = bfs(tuple(partner_rc))
    return {
        "ego_to_red":     int(d_ego[tuple(red_rc)]),
        "ego_to_blue":    int(d_ego[tuple(blue_rc)]),
        "partner_to_red": int(d_part[tuple(red_rc)]),
        "partner_to_blue": int(d_part[tuple(blue_rc)]),
    }


# --------------------------------------------------------------------------- #
# Per-layout evaluation over a capability pool                                #
# --------------------------------------------------------------------------- #

# Tolerance on reward-equality for "tie" classification of the optimal
# allocation. A step_penalty of 0.01 and an integer completion difference
# of 1 gives a reward gap of 0.01, so 1e-9 is safely below any real
# operational tick — ties only fire when both allocations complete in
# the same number of steps under the same partner cooldown.
_REWARD_TIE_TOL = 1e-9


@dataclass
class LayoutStats:
    """Analytical selection statistics for one layout across a cap pool."""
    # Optimal-allocation distribution.
    n_opt_red: int
    n_opt_blue: int
    n_opt_ties: int
    fraction_ties: float
    p_opt_red: float
    p_opt_blue: float
    n_non_tied: int
    # Reward.
    oracle_expected_reward: float
    fixed_red_expected_reward: float
    fixed_blue_expected_reward: float
    best_fixed_expected_reward: float
    delta_reward: float
    # Completion diagnostics.
    oracle_mean_completion: float
    oracle_std_completion: float
    oracle_max_completion: float
    oracle_p95_completion: float
    blind_mean_completion: float
    delta_completion: float
    # Feasibility.
    oracle_success_fraction: float
    fixed_red_success_fraction: float
    fixed_blue_success_fraction: float
    only_one_feasible_fraction: float
    # Observability.
    partner_to_red: int
    partner_to_blue: int
    # Raw per-cap arrays (kept for downstream diagnostics; not saved to JSON).
    per_cap_times_A: np.ndarray = field(repr=False)
    per_cap_times_B: np.ndarray = field(repr=False)
    per_cap_oracle_completion: np.ndarray = field(repr=False)

    def to_metadata_dict(self) -> Dict[str, float]:
        """JSON-safe subset written into per-layout metadata."""
        d = asdict(self)
        for k in ("per_cap_times_A", "per_cap_times_B", "per_cap_oracle_completion"):
            d.pop(k, None)
        # Cast numpy scalars.
        for k, v in list(d.items()):
            if isinstance(v, (np.floating, np.integer)):
                d[k] = float(v) if isinstance(v, np.floating) else int(v)
        return d


def evaluate_layout(ego_to_red: int, ego_to_blue: int,
                    partner_to_red: int, partner_to_blue: int,
                    cap_pairs: Sequence[Tuple[int, int]],
                    max_steps: int,
                    step_penalty: float,
                    success_reward: float) -> LayoutStats:
    """Evaluate a single layout over ``cap_pairs`` and return :class:`LayoutStats`.

    Two allocations per capability profile:
        * ``A``: ego -> RED,  partner -> BLUE  (partner cooldown = c_B)
        * ``B``: ego -> BLUE, partner -> RED   (partner cooldown = c_R)

    A cap profile is a *tie* iff the two allocations produce equal
    expected reward within :data:`_REWARD_TIE_TOL`. Ties are counted
    separately from RED-optimal and BLUE-optimal so they cannot bias the
    allocation-balance filter.
    """
    n_C = len(cap_pairs)
    if n_C == 0:
        raise ValueError("cap_pairs must be non-empty")

    times_A = np.empty(n_C, dtype=np.int64)
    times_B = np.empty(n_C, dtype=np.int64)
    for j, (c_r, c_b) in enumerate(cap_pairs):
        # Allocation A: ego->RED, partner->BLUE. Partner uses c_B.
        times_A[j] = completion_time(ego_to_red, partner_to_blue, int(c_b), max_steps)
        # Allocation B: ego->BLUE, partner->RED. Partner uses c_R.
        times_B[j] = completion_time(ego_to_blue, partner_to_red, int(c_r), max_steps)

    # Reward per (cap, alloc). We pass max_steps into _reward_from_time so
    # infeasible allocations pay -step_penalty * max_steps (matches env).
    rewards_A = np.array([_reward_from_time(int(t), max_steps, step_penalty, success_reward)
                          for t in times_A], dtype=np.float64)
    rewards_B = np.array([_reward_from_time(int(t), max_steps, step_penalty, success_reward)
                          for t in times_B], dtype=np.float64)

    # Success flags (any completion within horizon).
    succ_A = times_A < INFEASIBLE
    succ_B = times_B < INFEASIBLE

    # Optimal allocation labelling with an explicit tie band.
    diff = rewards_A - rewards_B
    is_tie = np.abs(diff) <= _REWARD_TIE_TOL
    opt_red  = (diff >  _REWARD_TIE_TOL)          # A > B: ego->RED optimal
    opt_blue = (diff < -_REWARD_TIE_TOL)          # B > A: ego->BLUE optimal
    n_opt_red = int(opt_red.sum())
    n_opt_blue = int(opt_blue.sum())
    n_opt_ties = int(is_tie.sum())
    n_non_tied = n_opt_red + n_opt_blue
    p_opt_red  = float(n_opt_red / n_non_tied) if n_non_tied > 0 else 0.0
    p_opt_blue = float(n_opt_blue / n_non_tied) if n_non_tied > 0 else 0.0
    fraction_ties = float(n_opt_ties / n_C)

    # Oracle per-cap: pick the reward-max allocation (ties go to A by
    # np.maximum, which is fine — tied reward is by construction the same).
    oracle_reward_per_cap = np.maximum(rewards_A, rewards_B)
    # Oracle completion per cap: on ties, take the min completion time.
    oracle_completion = np.where(diff > 0, times_A,
                        np.where(diff < 0, times_B,
                                 np.minimum(times_A, times_B))).astype(np.int64)
    oracle_success = oracle_completion < INFEASIBLE

    # Fixed-allocation baselines (partner-blind).
    fixed_red_reward  = float(rewards_A.mean())
    fixed_blue_reward = float(rewards_B.mean())
    best_fixed_reward = max(fixed_red_reward, fixed_blue_reward)
    oracle_reward = float(oracle_reward_per_cap.mean())
    delta_reward = oracle_reward - best_fixed_reward

    # Completion diagnostics (only over successful oracle cases).
    if oracle_success.any():
        succ_completions = oracle_completion[oracle_success].astype(np.float64)
        oracle_mean = float(succ_completions.mean())
        oracle_std  = float(succ_completions.std())
        oracle_max  = float(succ_completions.max())
        oracle_p95  = float(np.percentile(succ_completions, 95))
    else:
        oracle_mean = oracle_std = oracle_max = oracle_p95 = float("nan")

    # Blind completion: pick whichever fixed allocation had higher mean reward.
    blind_choice_A = fixed_red_reward >= fixed_blue_reward
    blind_times = times_A if blind_choice_A else times_B
    blind_success = blind_times < INFEASIBLE
    blind_mean = (float(blind_times[blind_success].mean())
                  if blind_success.any() else float("nan"))
    delta_completion = (blind_mean - oracle_mean
                        if (not np.isnan(oracle_mean) and not np.isnan(blind_mean))
                        else float("nan"))

    # Feasibility fractions.
    oracle_success_frac    = float(oracle_success.mean())
    fixed_red_success_frac  = float(succ_A.mean())
    fixed_blue_success_frac = float(succ_B.mean())
    only_one_feasible_frac = float(np.logical_xor(succ_A, succ_B).mean())

    return LayoutStats(
        n_opt_red=n_opt_red,
        n_opt_blue=n_opt_blue,
        n_opt_ties=n_opt_ties,
        fraction_ties=fraction_ties,
        p_opt_red=p_opt_red,
        p_opt_blue=p_opt_blue,
        n_non_tied=n_non_tied,
        oracle_expected_reward=oracle_reward,
        fixed_red_expected_reward=fixed_red_reward,
        fixed_blue_expected_reward=fixed_blue_reward,
        best_fixed_expected_reward=best_fixed_reward,
        delta_reward=delta_reward,
        oracle_mean_completion=oracle_mean,
        oracle_std_completion=oracle_std,
        oracle_max_completion=oracle_max,
        oracle_p95_completion=oracle_p95,
        blind_mean_completion=blind_mean,
        delta_completion=delta_completion,
        oracle_success_fraction=oracle_success_frac,
        fixed_red_success_fraction=fixed_red_success_frac,
        fixed_blue_success_fraction=fixed_blue_success_frac,
        only_one_feasible_fraction=only_one_feasible_frac,
        partner_to_red=int(partner_to_red),
        partner_to_blue=int(partner_to_blue),
        per_cap_times_A=times_A,
        per_cap_times_B=times_B,
        per_cap_oracle_completion=oracle_completion,
    )


def layout_stats(bfs_metadata: Dict[str, int],
                 cap_pairs: Sequence[Tuple[int, int]],
                 max_steps: int,
                 step_penalty: float,
                 success_reward: float) -> LayoutStats:
    """Convenience wrapper reading BFS distances from a metadata dict."""
    return evaluate_layout(
        ego_to_red=int(bfs_metadata["ego_to_red"]),
        ego_to_blue=int(bfs_metadata["ego_to_blue"]),
        partner_to_red=int(bfs_metadata["partner_to_red"]),
        partner_to_blue=int(bfs_metadata["partner_to_blue"]),
        cap_pairs=cap_pairs,
        max_steps=max_steps,
        step_penalty=step_penalty,
        success_reward=success_reward,
    )


# --------------------------------------------------------------------------- #
# Filter predicates                                                            #
# --------------------------------------------------------------------------- #

def passes_observability(stats_or_meta,
                         min_partner_goal_distance: int) -> bool:
    """Partner must be at least this many BFS steps from BOTH goals."""
    if isinstance(stats_or_meta, LayoutStats):
        pr, pb = stats_or_meta.partner_to_red, stats_or_meta.partner_to_blue
    else:
        pr = int(stats_or_meta["partner_to_red"])
        pb = int(stats_or_meta["partner_to_blue"])
    if pr < 0 or pb < 0:
        return False
    return pr >= min_partner_goal_distance and pb >= min_partner_goal_distance


def passes_allocation_balance(stats: LayoutStats,
                              min_frac: float,
                              max_frac: float,
                              max_tie_fraction: float = 0.5) -> bool:
    """Optimal ego allocation must genuinely depend on partner capability.

    Requires that ``p_opt_red`` lies in ``[min_frac, max_frac]`` computed
    over the *non-tied* subset of the cap pool, AND that ties do not
    dominate (else the balance criterion is not meaningful).
    """
    if stats.fraction_ties > max_tie_fraction:
        return False
    if stats.n_non_tied == 0:
        return False
    return min_frac <= stats.p_opt_red <= max_frac


def passes_feasibility(stats: LayoutStats,
                       min_oracle_success: float) -> bool:
    """Oracle must complete on essentially every cap pair."""
    return stats.oracle_success_fraction >= min_oracle_success


def passes_delta_reward(stats: LayoutStats,
                        threshold: float) -> bool:
    """Simple absolute threshold on the oracle-vs-blind reward gap."""
    return stats.delta_reward >= threshold


# --------------------------------------------------------------------------- #
# Horizon                                                                      #
# --------------------------------------------------------------------------- #

def worst_case_oracle_steps(stats: LayoutStats) -> float:
    """Max oracle completion across the cap pool. NaN if none feasible."""
    return float(stats.oracle_max_completion)


def horizon_outlier_mask(worst_case: np.ndarray,
                         n_sigma: float = 3.0) -> Tuple[np.ndarray, Dict[str, float]]:
    """Boolean mask (True = KEEP) rejecting entries ``>= mean + n_sigma*std``.

    Returns ``(keep_mask, diag)`` where ``diag`` records the cutoff and
    pool statistics. ``worst_case`` should be finite for every entry (we
    treat non-finite values as always rejected).
    """
    wc = np.asarray(worst_case, dtype=np.float64)
    finite = np.isfinite(wc)
    if not finite.any():
        raise ValueError("all worst-case values are non-finite — pool empty?")
    mu = float(wc[finite].mean())
    sd = float(wc[finite].std())
    upper = mu + n_sigma * sd
    keep = finite & (wc < upper)  # spec: reject at or beyond upper bound
    diag = {
        "mean": mu, "std": sd, "n_sigma": float(n_sigma), "upper_bound": upper,
        "n_kept": int(keep.sum()), "n_rejected": int((~keep).sum()),
        "n_input": int(wc.size),
    }
    return keep, diag


def derive_max_steps(retained_worst_case: np.ndarray,
                     safety_margin: int = 0) -> int:
    """Smallest integer horizon covering every retained worst-case step count."""
    wc = np.asarray(retained_worst_case, dtype=np.float64)
    finite = wc[np.isfinite(wc)]
    if finite.size == 0:
        raise ValueError("no finite worst-case values")
    return int(np.ceil(finite.max())) + int(safety_margin)


# --------------------------------------------------------------------------- #
# Stratified sampling                                                          #
# --------------------------------------------------------------------------- #

def _quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Map each entry to a bin id in ``[0, n_bins)`` using quantile edges.

    Robust to duplicate quantile edges (common on discrete metrics like
    ``num_junctions``): degenerate bins simply merge.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return np.zeros(0, dtype=np.int64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(v, qs) if qs.size > 0 else np.array([])
    return np.searchsorted(edges, v, side="right").astype(np.int64)


def stratified_sample(feature_matrix: np.ndarray,
                      n_target: int,
                      n_bins_per_feature: int,
                      seed: int) -> np.ndarray:
    """Deterministic stratified sample of ``n_target`` row indices.

    ``feature_matrix`` is ``(N, F)`` float array. Each column is binned
    into ``n_bins_per_feature`` quantile buckets; the composite
    ``F``-tuple bucket id is used as a stratum. We then round-robin
    across occupied strata, drawing one deterministic-shuffled member
    per stratum per cycle until ``n_target`` layouts are selected.

    Falls back to a plain shuffle if ``N <= n_target``.
    """
    N = int(feature_matrix.shape[0])
    if n_target >= N:
        rng = np.random.default_rng(seed)
        order = np.arange(N)
        rng.shuffle(order)
        return order

    rng = np.random.default_rng(seed)
    if feature_matrix.shape[1] == 0:
        order = np.arange(N)
        rng.shuffle(order)
        return order[:n_target]

    bins = np.stack([_quantile_bin(feature_matrix[:, f], n_bins_per_feature)
                     for f in range(feature_matrix.shape[1])], axis=1)
    # Composite id via structured hashing (row-major with per-feature stride).
    strides = np.array(
        [n_bins_per_feature ** f for f in range(feature_matrix.shape[1])],
        dtype=np.int64,
    )
    stratum = (bins * strides[None, :]).sum(axis=1)

    strata_ids, inv = np.unique(stratum, return_inverse=True)
    # Sort each stratum's members by a per-row hash for deterministic ordering.
    per_stratum: List[List[int]] = [[] for _ in strata_ids]
    for i, s in enumerate(inv):
        per_stratum[int(s)].append(int(i))
    for lst in per_stratum:
        rng.shuffle(lst)

    picked: List[int] = []
    stratum_order = np.arange(len(strata_ids))
    rng.shuffle(stratum_order)  # deterministic-random stratum visit order
    cursors = [0] * len(strata_ids)
    active = list(stratum_order)
    while len(picked) < n_target and active:
        next_active = []
        for s in active:
            if len(picked) >= n_target:
                break
            if cursors[s] < len(per_stratum[s]):
                picked.append(per_stratum[s][cursors[s]])
                cursors[s] += 1
                if cursors[s] < len(per_stratum[s]):
                    next_active.append(s)
        active = next_active if next_active else []
    return np.array(picked, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Summaries                                                                    #
# --------------------------------------------------------------------------- #

_QUANTILES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)


def summarize_distribution(values: Sequence[float]) -> Dict[str, float]:
    """Return min/quantiles/max/mean/std of ``values`` (NaNs dropped)."""
    a = np.asarray(list(values), dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "n": 0,
            "min": float("nan"), "max": float("nan"),
            "mean": float("nan"), "std": float("nan"),
            **{f"p{int(q*100):02d}": float("nan") for q in _QUANTILES if 0 < q < 1},
        }
    out = {
        "n": int(a.size),
        "min":  float(a.min()),
        "max":  float(a.max()),
        "mean": float(a.mean()),
        "std":  float(a.std()),
    }
    for q in _QUANTILES:
        if q in (0.0, 1.0):
            continue
        out[f"p{int(q*100):02d}"] = float(np.quantile(a, q))
    return out


def summarize_stats(stats_list: Sequence[LayoutStats]) -> Dict[str, Dict[str, float]]:
    """Distributions over a list of :class:`LayoutStats` for the manifest."""
    def col(attr):
        return [getattr(s, attr) for s in stats_list]
    return {
        "p_opt_red":                summarize_distribution(col("p_opt_red")),
        "fraction_ties":            summarize_distribution(col("fraction_ties")),
        "delta_reward":             summarize_distribution(col("delta_reward")),
        "oracle_expected_reward":   summarize_distribution(col("oracle_expected_reward")),
        "best_fixed_expected_reward": summarize_distribution(col("best_fixed_expected_reward")),
        "oracle_mean_completion":   summarize_distribution(col("oracle_mean_completion")),
        "oracle_max_completion":    summarize_distribution(col("oracle_max_completion")),
        "oracle_p95_completion":    summarize_distribution(col("oracle_p95_completion")),
        "oracle_success_fraction":  summarize_distribution(col("oracle_success_fraction")),
        "partner_to_red":           summarize_distribution(col("partner_to_red")),
        "partner_to_blue":          summarize_distribution(col("partner_to_blue")),
    }
