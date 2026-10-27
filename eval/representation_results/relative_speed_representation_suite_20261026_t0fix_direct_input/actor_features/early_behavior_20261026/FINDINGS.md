# Early behavior and decoding

Representation: Frozen actor features. Windows are recomputed from each version/category mean curve as the maximum five-step accuracy increase within0..50. The same held-out episode IDs and five-seed aggregation are used as in the baseline.

| Version | Category | Window | Accuracy before | Accuracy after |
|---|---|---|---:|---:|
| v1_matched | training_profile | 10–15 | 0.5771 | 0.6750 |
| v1_matched | one_delay_novel | 4–9 | 0.4714 | 0.5619 |
| v1_matched | both_delays_novel | 4–9 | 0.4917 | 0.6917 |
| v2 | training_profile | 1–6 | 0.4792 | 0.5563 |
| v2 | one_delay_novel | 5–10 | 0.5036 | 0.5857 |
| v2 | both_delays_novel | 7–12 | 0.4583 | 0.6833 |
| v3 | training_profile | 1–6 | 0.4625 | 0.5938 |
| v3 | one_delay_novel | 3–8 | 0.4714 | 0.5833 |
| v3 | both_delays_novel | 5–10 | 0.4333 | 0.6750 |

Paired gains use identical selected episodes at the window endpoints, equal profile weights within seed, and all five policy seeds. A category/cohort lacking any seed is omitted from its paired dot and identified in paired_cohort_coverage.csv. Motion diagnostics retain the original fixed cohort validated through step30; window endpoints outside this motion range are not summarized.
