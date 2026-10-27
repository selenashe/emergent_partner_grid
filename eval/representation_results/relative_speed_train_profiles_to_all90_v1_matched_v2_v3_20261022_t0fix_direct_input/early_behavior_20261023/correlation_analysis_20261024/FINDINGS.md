# Behavioral correlates of instantaneous decoding

All seven distinct behavioral metrics plotted across the two figures are compared on identical replay-validated held-out episodes, steps0–30, multi-partner RNNs only, three experiments × green/yellow/pink, five policy seeds. The dashed fixed-step20 readout is another decoding measurement, not a behavioral metric, and is not ranked. Round completion appears in both figures and is counted once.

Compute episode event indicators exactly as in the original figures; average episodes within each profile, profiles equally within each seed, then average five seeds. The same episodes validated through step30 are retained at every timestep for every metric. Actual assignment at step t follows h_t; all other plotted events precede h_t.

For each panel separately, subtract its time mean from predictor and decoding. Pool the centered samples across nine equally represented panels and compute Pearson r. This removes between-panel offsets. Constant behavioral curves contribute zero covariance without dropping their panels. Individual-panel undefined correlations are marked flat.

Levels correlate x(t) with accuracy(t), 31 points/panel. One-step changes correlate x(t+1)-x(t) with accuracy(t+1)-accuracy(t),30points/panel. Five-step changes use t+5 instead,26points/panel. No lag is selected for the main comparisons. These describe contemporaneous alignment of changes, not prediction of unseen data. Five-step changes overlap and are temporally dependent.

Uncertainty: displayed approximate95% intervals use delete-one-policy-seed jackknife standard errors on Fisher-z-transformed correlations, stratified by experiment, with Student t12 (three experiments times four within-experiment seed degrees of freedom). Each deletion keeps all categories together. Correlation contrasts use the same jackknife on their raw differences. These intervals describe policy-seed uncertainty, not resampling of capabilities or evaluation episodes. No timestep-level significance tests.

Sensitivity also independently resamples five policy seeds within each experiment, keeping all categories and metrics together; recompute seed-mean curves and correlations,10,000draws,RNG30000. Rank frequencies and paired differences are saved. Raw bootstrap percentile ranges are retained in CSV but are not the displayed confidence intervals: with only five seeds, duplicate resampling increases noise in the mean curves and biases correlation estimates downward; several percentile ranges fail to contain the observed estimate. The approximate jackknife interval avoids using that biased percentile distribution as a confidence interval.

Positive-lag sensitivity is also saved: correlate behavior changes at t with decoding changes at t+lag, lags0..5. These comparisons are descriptive and were not used to optimize the primary ranking. Timing resolution, readout changes, mixtures of capabilities, and cumulative event definitions can affect correlations. None identifies a causal mechanism.

| Metric | Level r | 1-step change r | 5-step change r (approx95% seed jackknife CI) |
|---|---:|---:|---:|
| Four off-goal waits after a move | 0.905 | 0.267 | 0.715 [0.578, 0.813] |
| At least two partner moves | 0.917 | 0.134 | 0.626 [0.425, 0.769] |
| Partner assigned faster color | 0.314 | 0.008 | 0.209 [-0.060, 0.450] |
| Within-round goal switch | 0.632 | 0.026 | 0.087 [-0.051, 0.223] |
| Both goal colors pursued | 0.637 | 0.019 | 0.077 [-0.080, 0.231] |
| First partner move | 0.482 | -0.195 | 0.006 [-0.131, 0.142] |
| First round finished | 0.832 | -0.017 | -0.215 [-0.434, 0.026] |

The waiting metric and second-move metric are correlated behavioral descriptions and are not independent evidence. Raw levels are especially vulnerable to a shared increasing time trend; compare the change columns when asking what changes alongside decoding.

Files: correlation_summary.csv; panel_correlations.csv; policy_seed_correlations.csv; paired_metric_differences.csv; lag_sensitivity.csv; matched_curves_per_seed.csv/NPZ; correlation_bootstrap.npz; comparison and per-panel PNG/PDF figures; analysis_audit.json; frozen source/.

