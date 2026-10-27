# Early behavior and decoding

Representation: Previous memory h_(t-1). Windows are recomputed from each version/category mean curve as the maximum five-step accuracy increase within0..50. The same held-out episode IDs and five-seed aggregation are used as in the baseline.

| Version | Category | Window | Accuracy before | Accuracy after |
|---|---|---|---:|---:|
| v1_matched | training_profile | 5–10 | 0.4812 | 0.6438 |
| v1_matched | one_delay_novel | 3–8 | 0.4631 | 0.5833 |
| v1_matched | both_delays_novel | 4–9 | 0.4667 | 0.7000 |
| v2 | training_profile | 13–18 | 0.5667 | 0.6833 |
| v2 | one_delay_novel | 4–9 | 0.4667 | 0.5607 |
| v2 | both_delays_novel | 12–17 | 0.5833 | 0.7417 |
| v3 | training_profile | 5–10 | 0.4604 | 0.6354 |
| v3 | one_delay_novel | 4–9 | 0.4548 | 0.5786 |
| v3 | both_delays_novel | 6–11 | 0.4750 | 0.6750 |

Paired gains use identical selected episodes at the window endpoints, equal profile weights within seed, and all five policy seeds. A category/cohort lacking any seed is omitted from its paired dot and identified in paired_cohort_coverage.csv. Motion diagnostics retain the original fixed cohort validated through step30; window endpoints outside this motion range are not summarized.
