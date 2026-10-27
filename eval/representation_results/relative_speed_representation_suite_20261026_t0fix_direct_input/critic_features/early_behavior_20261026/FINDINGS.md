# Early behavior and decoding

Representation: Frozen critic features. Windows are recomputed from each version/category mean curve as the maximum five-step accuracy increase within0..50. The same held-out episode IDs and five-seed aggregation are used as in the baseline.

| Version | Category | Window | Accuracy before | Accuracy after |
|---|---|---|---:|---:|
| v1_matched | training_profile | 5–10 | 0.5063 | 0.6583 |
| v1_matched | one_delay_novel | 3–8 | 0.4345 | 0.5952 |
| v1_matched | both_delays_novel | 5–10 | 0.4917 | 0.7417 |
| v2 | training_profile | 4–9 | 0.4708 | 0.5771 |
| v2 | one_delay_novel | 4–9 | 0.4774 | 0.5905 |
| v2 | both_delays_novel | 9–14 | 0.5500 | 0.7333 |
| v3 | training_profile | 4–9 | 0.4688 | 0.6583 |
| v3 | one_delay_novel | 3–8 | 0.4595 | 0.6071 |
| v3 | both_delays_novel | 4–9 | 0.4667 | 0.7167 |

Paired gains use identical selected episodes at the window endpoints, equal profile weights within seed, and all five policy seeds. A category/cohort lacking any seed is omitted from its paired dot and identified in paired_cohort_coverage.csv. Motion diagnostics retain the original fixed cohort validated through step30; window endpoints outside this motion range are not summarized.
