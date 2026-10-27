# Early behavior and decoding

Representation: Masked action probabilities. Windows are recomputed from each version/category mean curve as the maximum five-step accuracy increase within0..50. The same held-out episode IDs and five-seed aggregation are used as in the baseline.

| Version | Category | Window | Accuracy before | Accuracy after |
|---|---|---|---:|---:|
| v1_matched | training_profile | 3–8 | 0.4375 | 0.5417 |
| v1_matched | one_delay_novel | 19–24 | 0.5167 | 0.5798 |
| v1_matched | both_delays_novel | 5–10 | 0.4083 | 0.5667 |
| v2 | training_profile | 21–26 | 0.5083 | 0.5854 |
| v2 | one_delay_novel | 18–23 | 0.5024 | 0.5714 |
| v2 | both_delays_novel | 5–10 | 0.4417 | 0.5750 |
| v3 | training_profile | 7–12 | 0.4583 | 0.5479 |
| v3 | one_delay_novel | 5–10 | 0.4857 | 0.5393 |
| v3 | both_delays_novel | 2–7 | 0.4083 | 0.5750 |

Paired gains use identical selected episodes at the window endpoints, equal profile weights within seed, and all five policy seeds. A category/cohort lacking any seed is omitted from its paired dot and identified in paired_cohort_coverage.csv. Motion diagnostics retain the original fixed cohort validated through step30; window endpoints outside this motion range are not summarized.
