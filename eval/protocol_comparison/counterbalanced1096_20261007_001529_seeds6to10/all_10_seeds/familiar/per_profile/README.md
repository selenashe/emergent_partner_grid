# Every familiar partner profile: ten-seed performance

Each figure has **24 subplots**, one for every partner profile in this
population. Panels are sorted by `(RED delay, BLUE delay)`. A delay is the number
of wait steps between partner moves; movement occurs every `delay + 1` steps.

Each policy/profile was evaluated for 20 episodes of 20 rounds. All four
conditions and both original allocation protocols are included. Means weight
learner seeds 1–10 equally. Error bars/bands show sample SD across those ten
policies. This is the same seed set used by the pooled comparison figures.

| Figure | PNG | PDF |
|---|---|---|
| Round success by profile | [View](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/per_profile/profile_success.png) | [Download](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/per_profile/profile_success.pdf) |
| Episode return by profile | [View](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/per_profile/profile_episode_return.png) | [Download](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/per_profile/profile_episode_return.pdf) |
| Episode steps by profile | [View](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/per_profile/profile_episode_steps.png) | [Download](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/per_profile/profile_episode_steps.pdf) |
| Success across rounds, v1 | [View](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/per_profile/profile_round_success_v1.png) | [Download](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/per_profile/profile_round_success_v1.pdf) |
| Success across rounds, v2 | [View](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/per_profile/profile_round_success_v2.png) | [Download](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/per_profile/profile_round_success_v2.pdf) |

The first three figures compare v1 (blue) and v2 (orange) within each condition;
black dots are individual policy seeds. Return includes failure penalties.
Episode steps count every valid transition through final episode termination,
including timeouts and the terminal transition. Scan padding is excluded.

The two round figures show rounds 1–20 with condition colors shared between v1
and v2. Bands are clipped to 0–100% for display; exported SD values are intact.
These are performance across rounds inside an evaluation episode, rather than
curves over PPO training updates.

Familiar refers to the shared training population. The single-partner control
was trained only with profile `(1, 4)`, rather than all 24 profiles.

[Aggregated metrics and provenance](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/per_profile/profile_comparison.json),
[per-profile, per-seed scalar metrics](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/per_profile/profile_per_seed.csv),
[per-seed metrics and round curves](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/per_profile/profile_per_seed_rounds.json).

Regenerate both populations with:

```bash
python eval/plot_profile_performance.py \
  --extension-manifest train/manifests/sbatch_counterbalanced1096_20261007_001529_seeds6to10.json
```

Use `--population test` for held-out profiles, `--population train` for the shared
training population, or `--population both` (the default). The per-profile means
reconstruct the original pooled summaries for every policy. Ten regression tests
cover grouping by explicit profile labels, padding exclusion, terminal rewards,
round ordering, malformed records, missing seeds, and sample SD.
