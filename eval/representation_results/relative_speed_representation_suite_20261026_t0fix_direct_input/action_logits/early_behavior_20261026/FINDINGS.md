# Early behavior and decoding

Representation: Unmasked action logits. Windows are recomputed from each version/category mean curve as the maximum five-step accuracy increase within0..50. The same held-out episode IDs and five-seed aggregation are used as in the baseline.

| Version | Category | Window | Accuracy before | Accuracy after |
|---|---|---|---:|---:|
| v1_matched | training_profile | 6–11 | 0.5021 | 0.6125 |
| v1_matched | one_delay_novel | 6–11 | 0.5012 | 0.5917 |
| v1_matched | both_delays_novel | 5–10 | 0.4500 | 0.7167 |
| v2 | training_profile | 2–7 | 0.4708 | 0.5500 |
| v2 | one_delay_novel | 1–6 | 0.4798 | 0.5298 |
| v2 | both_delays_novel | 7–12 | 0.5250 | 0.7083 |
| v3 | training_profile | 5–10 | 0.5000 | 0.6354 |
| v3 | one_delay_novel | 4–9 | 0.4940 | 0.5571 |
| v3 | both_delays_novel | 7–12 | 0.5333 | 0.7417 |

Paired gains use identical selected episodes at the window endpoints, equal profile weights within seed, and all five policy seeds. A category/cohort lacking any seed is omitted from its paired dot and identified in paired_cohort_coverage.csv. Motion diagnostics retain the original fixed cohort validated through step30; window endpoints outside this motion range are not summarized.
