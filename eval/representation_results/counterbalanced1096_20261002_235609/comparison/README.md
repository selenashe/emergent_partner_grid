# Counterbalanced v1/v2 representation comparison

All three RNN conditions and all five trained seeds are included for each version. Probes, preprocessing and the shared 16/4-per-profile split match the earlier analysis. Only linear readouts are trained; policy weights and evaluation rollouts are unchanged.

The score is `1 − mean(abs(predicted_delay − actual_delay))/9`, rather than exact-class accuracy. The dashed baseline uses five independent random Normal representations. Shading resamples the five policy seeds, using 10,000 percentile bootstrap draws. Version differences pair nominal seed numbers; the episode trajectories are not paired. These are descriptive comparisons.

Probe training includes all 46 capability profiles. Probe testing holds out episodes within each profile; the familiar/novel breakdown refers to whether a profile appeared in policy training. It does not test a probe's transfer to capability classes absent from probe fitting.

The single-partner training profile is `(d_R, d_B) = (1, 4)`, a red specialist.
An additional goal-exposure check of all 30 saved evaluations found that the
single-partner policies assign the partner to red at round end more often:
97.96% versus 58.75% for multi-partner v1, and 79.39% versus 44.11% for v2
(means across five seeds). This supports asymmetric task exposure as a
possible explanation of the red-decoding advantage; it is not a causal test.
Goal occupancy includes waiting and completed-task states, rather than counting
movements. `task_exposure_per_seed.csv` and `task_exposure_summary.csv` record
both valid-step occupancy and round-end assignments, joined to decoding scores.
Reproduce with `python eval/summarize_decoding_task_exposure.py`.

| Condition | Target | v1, round 20 | v2, round 20 | v2 − v1 [95% CI] |
| --- | --- | --- | --- | --- |
| Multi-partner RNN | d_B | 0.893 | 0.899 | +0.006 [-0.045, +0.052] |
| Multi-partner RNN | d_R | 0.904 | 0.884 | -0.020 [-0.081, +0.041] |
| No-influence RNN | d_B | 0.738 | 0.683 | -0.054 [-0.083, -0.011] |
| No-influence RNN | d_R | 0.712 | 0.696 | -0.016 [-0.056, +0.020] |
| Single-partner RNN | d_B | 0.829 | 0.822 | -0.007 [-0.048, +0.037] |
| Single-partner RNN | d_R | 0.973 | 0.898 | -0.075 [-0.124, -0.026] |

`v1_v2_time_all_seeds` and `v1_v2_round_all_seeds` compare version means. The four `*_every_seed` figures expose all individual policy curves. PNG and PDF versions are saved. Every network's UMAP figures and fitted probes are in the adjacent version directories. Decoding establishes recoverable information, not causal use of that information by the actor.

Reproduce after completing both version analyses:

```bash
python eval/compare_representation_versions.py --results-root eval/representation_results/counterbalanced1096_20261002_235609
```
