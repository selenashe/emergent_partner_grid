# Early behavior and decoding

Representation: Frozen value estimate. Windows are recomputed from each version/category mean curve as the maximum five-step accuracy increase within0..50. The same held-out episode IDs and five-seed aggregation are used as in the baseline.

| Version | Category | Window | Accuracy before | Accuracy after |
|---|---|---|---:|---:|
| v1_matched | training_profile | 5–10 | 0.5437 | 0.7438 |
| v1_matched | one_delay_novel | 2–7 | 0.4917 | 0.6119 |
| v1_matched | both_delays_novel | 3–8 | 0.5500 | 0.8000 |
| v2 | training_profile | 5–10 | 0.5417 | 0.6729 |
| v2 | one_delay_novel | 3–8 | 0.4917 | 0.5952 |
| v2 | both_delays_novel | 3–8 | 0.5500 | 0.7500 |
| v3 | training_profile | 3–8 | 0.4896 | 0.6729 |
| v3 | one_delay_novel | 3–8 | 0.4905 | 0.6214 |
| v3 | both_delays_novel | 2–7 | 0.5750 | 0.7750 |

Paired gains use identical selected episodes at the window endpoints, equal profile weights within seed, and all five policy seeds. A category/cohort lacking any seed is omitted from its paired dot and identified in paired_cohort_coverage.csv. Motion diagnostics retain the original fixed cohort validated through step30; window endpoints outside this motion range are not summarized.
