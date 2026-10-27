"""Exploratory mean-trend and adjacent-decrease tests for sliding returns.

Training-run mean return is the observation; rollout episodes are not counted
as independent training replicates. Independent-run inference and a conservative
matched-seed sensitivity analysis are reported separately.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

CONDITIONS = ("mlp_diverse_influence", "rnn_diverse_influence",
              "rnn_single_influence", "rnn_diverse_noinfluence")


def holm(pvalues):
    values = np.asarray(pvalues, float)
    order = np.argsort(values)
    adjusted = np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)).clip(0, 1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def welch_contrast(groups, coefficients):
    groups = [np.asarray(group, float) for group in groups]
    coefficients = np.asarray(coefficients, float)
    estimate = float(coefficients @ np.array([group.mean() for group in groups]))
    components = coefficients ** 2 * np.array([group.var(ddof=1) / len(group) for group in groups])
    variance = components.sum()
    degrees = float(variance ** 2 / sum(component ** 2 / (len(group) - 1)
                                        for component, group in zip(components, groups)))
    se = float(np.sqrt(variance))
    statistic = estimate / se
    quantile = stats.t.ppf(.975, degrees)
    return dict(estimate=estimate, se=se, df=degrees, statistic=statistic,
                p_less=float(stats.t.cdf(statistic, degrees)),
                p_greater=float(stats.t.sf(statistic, degrees)),
                ci95_low=float(estimate - quantile * se), ci95_high=float(estimate + quantile * se))


def check_calculations():
    # A two-group contrast must reproduce SciPy's unequal-variance t test.
    before, after = np.array([3., 4., 4., 6., 8.]), np.array([1., 1.5, 2., 3., 4.])
    ours = welch_contrast([before, after], [-1, 1])
    reference = stats.ttest_ind(after, before, equal_var=False, alternative="less")
    assert np.isclose(ours["statistic"], reference.statistic)
    assert np.isclose(ours["p_less"], reference.pvalue)
    assert np.allclose(holm([.01, .04, .03]), [.03, .06, .06])


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(directories, out):
    check_calculations()
    records, source_hashes = {}, {}
    for directory in directories:
        path = directory / "behavior_per_seed.csv"
        source_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in csv.DictReader(path.open()):
            key = (row["population"], row["reversibility"], row["condition"], int(row["t"]), int(row["seed"]))
            value = float(row["episode_return"])
            if key in records:
                if key[3] != 0 or records[key] != value:
                    raise ValueError("Non-baseline duplicate or inconsistent baseline")
            records[key] = value
    times = sorted({key[3] for key in records})
    if times != list(range(11)):
        raise ValueError("Require all observation times t=0..10")
    times = np.asarray(times, float)
    coefficients = (times - times.mean()) / np.sum((times - times.mean()) ** 2)
    summaries, adjacent = [], []
    for population in ("test", "train"):
        for mode in ("irreversible", "reversible"):
            for condition in CONDITIONS:
                values = np.array([[records[population, mode, condition, int(t), seed]
                                    for seed in range(1, 6)] for t in times])
                assert values.shape == (11, 5) and np.isfinite(values).all()
                trend = welch_contrast(values, coefficients)
                # Coefficients equal the slope fitted to balanced individual-run data.
                reference = stats.linregress(np.repeat(times, 5), values.ravel())
                assert np.isclose(trend["estimate"], reference.slope)
                seed_slopes = coefficients @ values
                blocked = stats.ttest_1samp(seed_slopes, 0, alternative="less")
                blocked_se = float(seed_slopes.std(ddof=1) / np.sqrt(5))
                pairs = []
                for index in range(10):
                    difference = welch_contrast([values[index], values[index + 1]], [-1, 1])
                    blocked_difference = stats.ttest_rel(values[index + 1], values[index], alternative="less")
                    pairs.append(dict(population=population, allocation=mode, condition=condition,
                                      t_before=index, t_after=index + 1, mean_change=difference["estimate"],
                                      se=difference["se"], df=difference["df"],
                                      change_ci95_low=difference["ci95_low"], change_ci95_high=difference["ci95_high"],
                                      p_decrease=difference["p_less"], p_increase=difference["p_greater"],
                                      seed_block_p_decrease=float(blocked_difference.pvalue)))
                adjacent.extend(pairs)
                summaries.append(dict(population=population, allocation=mode, condition=condition,
                                      n_times=11, n_runs_per_time=5, slope=trend["estimate"],
                                      slope_se=trend["se"], slope_df=trend["df"],
                                      slope_ci95_low=trend["ci95_low"], slope_ci95_high=trend["ci95_high"],
                                      trend_p=trend["p_less"], seed_block_slope_se=blocked_se,
                                      seed_block_trend_p=float(blocked.pvalue),
                                      strict_decrease_iut_p=max(pair["p_decrease"] for pair in pairs),
                                      seed_block_strict_decrease_iut_p=max(pair["seed_block_p_decrease"] for pair in pairs),
                                      observed_upward_steps=int(np.sum(np.diff(values.mean(axis=1)) > 0)),
                                      observed_mean_curve_nonincreasing=bool(np.all(np.diff(values.mean(axis=1)) <= 0)),
                                      seed_slopes=";".join(map(str, seed_slopes))))
    for population in ("test", "train"):
        selected = [row for row in summaries if row["population"] == population]
        # Eight curves per population: four conditions x two allocation modes.
        for column in ("trend_p", "seed_block_trend_p", "strict_decrease_iut_p", "seed_block_strict_decrease_iut_p"):
            for row, adjusted in zip(selected, holm([row[column] for row in selected])):
                row[column + "_holm8"] = float(adjusted)
        selected_pairs = [row for row in adjacent if row["population"] == population]
        # Finding any violation of nonincrease is a different, 80-comparison family.
        for column in ("p_increase", "p_decrease"):
            for row, adjusted in zip(selected_pairs, holm([row[column] for row in selected_pairs])):
                row[column + "_holm80"] = float(adjusted)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "trend_and_monotonicity_tests.csv", summaries)
    write_csv(out / "adjacent_t_tests.csv", adjacent)
    metadata = dict(primary_population="test (novel partner profiles)", secondary_population="train (familiar profiles); separate correction family",
                    t_range="0..10; baselines included once", experimental_unit="one trained policy's mean episode return",
                    n_runs_per_curve=55, n_seeds_per_time=5,
                    primary_trend="One-sided Welch-Satterthwaite linear contrast of the eleven mean returns; H0: slope >= 0, H1: slope < 0",
                    slope_coefficients=coefficients.tolist(),
                    strict_decrease="Intersection-union test: all five adjacent true-mean differences must be < 0; combined p=max(component one-sided p)",
                    nonincrease="A nonincreasing mean curve permits equality. Failure of the stronger strict-decrease test neither establishes nor disproves nonincrease.",
                    multiplicity="Holm across eight curves, separately for each inferential question and population. Any adjacent increase/decrease discovery uses Holm across 80 comparisons per population.",
                    seed_block_sensitivity="Treat each nominal seed's whole t-curve as a block: five per-seed slopes, one-sided t test with df=4. Matching identifiers does not establish identical experiences.",
                    confidence_intervals="Pointwise two-sided 95%; not simultaneous/multiplicity-adjusted",
                    limitations=["Independent-run analysis assumes independence across t; repeated random seed identifiers may induce dependence.",
                                 "Both t approximations rely on approximately normal seed-level means/contrasts; only five learner seeds per t, with occasional failed-learning outliers.",
                                 "These tests were selected after viewing the curves and are exploratory.",
                                 "A negative overall slope allows local increases and does not prove monotonicity.",
                                 "Evaluation episodes share trained weights and are not independent training replicates.",
                                 "t=0 reuses historical policies; reversibility branches retain original sliding irreversible/v3 trainer differences."],
                    sources={"Welch": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html",
                             "Holm": "https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html",
                             "intersection_union": "https://doi.org/10.1214/ss/1032280304"},
                    scipy_version=scipy.__version__, numpy_version=np.__version__, input_sha256=source_hashes,
                    summaries=summaries)
    (out / "methods_and_results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for row in summaries:
        if row["population"] == "test":
            print(row["allocation"], row["condition"], "slope", round(row["slope"], 4),
                  "trend_Holm_p", round(row["trend_p_holm8"], 6),
                  "seed_block_Holm_p", round(row["seed_block_trend_p_holm8"], 6),
                  "strict_Holm_p", round(row["strict_decrease_iut_p_holm8"], 6))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.results, args.out_dir)
