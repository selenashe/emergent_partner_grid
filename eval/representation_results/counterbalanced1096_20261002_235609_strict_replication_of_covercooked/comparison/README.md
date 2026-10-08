# Counterbalanced v1/v2 representation comparison

All three RNN conditions and all five trained seeds are included for each version. The pinned upstream JAX/Flax routine fits AdamW probes using fresh scalar-label-stratified splits, best-test-checkpoint selection, and warm starts. Corresponding fits have identical split masks across v1/v2, but splits change across targets, networks and cutoffs. The dashed line reproduces the released random-feature/random-label baseline. Only linear readouts are trained; policy weights and evaluation rollouts are unchanged.

The score is `1 − mean(abs(predicted_delay − actual_delay))/9`, rather than exact-class accuracy. Shading resamples the five policy seeds, using 10,000 percentile bootstrap draws. Version differences pair nominal seed numbers; the episode trajectories are not paired. These are descriptive comparisons.

Probe training includes all 46 capability profiles. Probe testing holds out episodes within each profile; the familiar/novel breakdown refers to whether a profile appeared in policy training. It does not test a probe's transfer to capability classes absent from probe fitting.

| Condition | Target | v1, round 20 | v2, round 20 | v2 − v1 [95% CI] |
| --- | --- | --- | --- | --- |
| Multi-partner RNN | d_B | 0.925 | 0.934 | +0.010 [-0.038, +0.054] |
| Multi-partner RNN | d_R | 0.940 | 0.931 | -0.009 [-0.050, +0.031] |
| No-influence RNN | d_B | 0.796 | 0.778 | -0.018 [-0.079, +0.049] |
| No-influence RNN | d_R | 0.780 | 0.755 | -0.025 [-0.056, +0.005] |
| Single-partner RNN | d_B | 0.863 | 0.874 | +0.011 [-0.009, +0.039] |
| Single-partner RNN | d_R | 0.994 | 0.939 | -0.055 [-0.088, -0.021] |

`v1_v2_time_all_seeds` and `v1_v2_round_all_seeds` compare version means. The four `*_every_seed` figures expose all individual policy curves. PNG and PDF versions are saved. Fitted probes are in the adjacent version directories. UMAPs use unchanged features and remain in the preceding analysis. Decoding establishes recoverable information, not causal use of that information by the actor.

Reproduce after completing both version analyses:

```bash
python eval/compare_representation_versions.py --results-root eval/representation_results/counterbalanced1096_20261002_235609_strict_replication_of_covercooked
```

The released routine selects on test scores and resamples splits while carrying previously fitted weights. Some later test episodes were training examples at earlier cutoffs. These are source-replication scores, not fully independent hold-out estimates. The per-version reports and checkpoint traces document this.
