# Early behavior and decoding

Representation: Current observation x_t. Windows are recomputed from each version/category mean curve as the maximum five-step accuracy increase within0..50. The same held-out episode IDs and five-seed aggregation are used as in the baseline.

| Version | Category | Window | Accuracy before | Accuracy after |
|---|---|---|---:|---:|
| v1_matched | training_profile | 3–8 | 0.4375 | 0.5104 |
| v1_matched | one_delay_novel | 2–7 | 0.4440 | 0.5155 |
| v1_matched | both_delays_novel | 44–49 | 0.3083 | 0.5583 |
| v2 | training_profile | 24–29 | 0.5000 | 0.5750 |
| v2 | one_delay_novel | 37–42 | 0.5179 | 0.5810 |
| v2 | both_delays_novel | 8–13 | 0.4833 | 0.6167 |
| v3 | training_profile | 9–14 | 0.4729 | 0.5375 |
| v3 | one_delay_novel | 20–25 | 0.5214 | 0.5714 |
| v3 | both_delays_novel | 6–11 | 0.3833 | 0.5500 |

Paired gains use identical selected episodes at the window endpoints, equal profile weights within seed, and all five policy seeds. A category/cohort lacking any seed is omitted from its paired dot and identified in paired_cohort_coverage.csv. Motion diagnostics retain the original fixed cohort validated through step30; window endpoints outside this motion range are not summarized.
