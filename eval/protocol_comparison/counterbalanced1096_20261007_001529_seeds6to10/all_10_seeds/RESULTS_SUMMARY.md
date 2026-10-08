# Counterbalanced v1/v2: all ten learner seeds

All 40 additional policies (seeds 6–10) completed successfully, along with both
standalone evaluation jobs and the combined averaging job. The results below
combine original seeds 1–5 with extension seeds 6–10 using equal seed weights.
Every mean/SD uses ten independent trained policies in that condition/protocol;
SD is the sample standard deviation across learner seeds (ddof=1).

Each new policy completed 915 PPO updates and 59,965,440 environment transitions
(60M nominal budget). Saved configs confirm actual learner seeds 6–10,
NUM_SEEDS=1 and the original categorical action-sampling method. The exact
original frozen trainers, environment, layouts and counterbalanced schedules
passed their recorded hash checks. Both versions use the same 1096 layouts;
there is no layout holdout.

## Novel partners

22 profiles × 20 episodes/profile × 20 rounds/episode for each policy.

Success is the percentage of completed rounds that succeed. Return is per
20-round episode, including failure penalties. All entries are mean ± sample SD.

| Condition | v1 success (%) | v2 success (%) | v1 return | v2 return | v1 episode steps | v2 episode steps |
|---|---:|---:|---:|---:|---:|---:|
| Diverse RNN + influence | 97.77 ± 1.56 | 95.96 ± 4.56 | 15.07 ± 1.14 | 13.07 ± 2.48 | 467.7 ± 100.2 | 631.1 ± 163.6 |
| Diverse MLP + influence | 97.16 ± 4.71 | 95.42 ± 4.94 | 13.08 ± 1.93 | 12.14 ± 2.02 | 654.9 ± 97.8 | 713.4 ± 102.4 |
| Single-partner RNN + influence | 98.80 ± 0.53 | 89.35 ± 14.92 | 13.86 ± 0.22 | 10.12 ± 5.37 | 609.6 ± 11.7 | 792.8 ± 236.5 |
| Diverse RNN, no influence | 90.73 ± 10.02 | 90.94 ± 7.52 | 10.21 ± 4.12 | 10.19 ± 3.05 | 811.3 ± 210.1 | 818.0 ± 152.9 |

[Full metrics](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/novel/comparison.json), [per-seed results](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/novel/per_seed.csv), [performance figure](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/novel/performance_comparison.png).

## Familiar partners

24 profiles × 20 episodes/profile × 20 rounds/episode for each policy.

Success is the percentage of completed rounds that succeed. Return is per
20-round episode, including failure penalties. All entries are mean ± sample SD.

| Condition | v1 success (%) | v2 success (%) | v1 return | v2 return | v1 episode steps | v2 episode steps |
|---|---:|---:|---:|---:|---:|---:|
| Diverse RNN + influence | 98.32 ± 1.24 | 96.60 ± 4.40 | 14.43 ± 1.06 | 12.42 ± 2.36 | 543.6 ± 95.1 | 709.6 ± 155.0 |
| Diverse MLP + influence | 96.94 ± 4.92 | 95.91 ± 4.53 | 12.25 ± 1.93 | 11.56 ± 1.78 | 732.9 ± 93.2 | 781.8 ± 86.7 |
| Single-partner RNN + influence | 99.09 ± 0.22 | 89.68 ± 15.17 | 13.24 ± 0.09 | 9.32 ± 5.43 | 677.2 ± 4.8 | 879.3 ± 238.3 |
| Diverse RNN, no influence | 90.06 ± 10.45 | 90.25 ± 7.84 | 9.34 ± 4.11 | 9.31 ± 3.05 | 885.1 ± 200.5 | 891.8 ± 147.4 |

[Full metrics](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/comparison.json), [per-seed results](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/per_seed.csv), [performance figure](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/performance_comparison.png).

## Observations

On novel partners, diverse RNNs have higher mean returns and fewer episode steps
than diverse MLPs in both versions. V1's diverse RNN has the highest mean return
(15.07 versus 13.07 for v2). V1's single-partner RNN has higher round success
(98.80%) but takes more steps than its diverse RNN and earns lower return.

V2's single-partner RNN has substantial between-seed variability: novel success
ranges from 50.05% (seed 8) to 99.26% (seed 5). Seed 8 was retained in the ten-seed
mean, as were every other original and additional learner seed. This helps
explain the 89.35% mean and 14.92 percentage-point SD; the mean was not selected
using a best seed. This observation describes the results and does not establish
why that policy performed poorly.

Completion checked: 2026-10-07T19:23:21.550724+00:00. Training jobs 17747295–17747334; evaluators
17747335/17747336; aggregation 17747337. All 43 exited successfully. Earlier failed
launch/check attempts are preserved in the manifest and snapshot records.

## Per-profile subplot figures

Every partner profile now has its own subplot. Paired v1/v2 figures report
success, episode return and episode steps, with ten-seed means, sample SD and
individual policy dots. Separate v1/v2 grids show success through rounds 1–20.
All four conditions and all learner seeds 1–10 are included.

- [Held-out profiles: all figures](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/novel/per_profile/README.md)
- [Familiar population: all figures](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/per_profile/README.md)

PNG and vector PDF versions, per-profile CSVs and JSON metrics are saved in each
population's `per_profile/` folder. Per-profile means reconstruct the completed
pooled results.
