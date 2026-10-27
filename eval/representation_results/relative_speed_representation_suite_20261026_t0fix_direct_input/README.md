# Matched representation probe suite

Eight representations; identical 24-profile x16-episode fitting set and all90 x4 held-out testing set. Separate affine classifiers at each step0..400 for thirty frozen policies. Recovered inputs share the original episode IDs and RNG schedules.

- [Current observation x_t](current_observation/README.md): 249 input features.
- [Previous memory h_(t-1)](previous_memory/README.md): 128 input features.
- [Updated memory h_t](updated_memory/README.md): 128 input features.
- [Frozen actor features](actor_features/README.md): 128 input features.
- [Frozen critic features](critic_features/README.md): 128 input features.
- [Unmasked action logits](action_logits/README.md): 15 input features.
- [Masked action probabilities](action_probabilities/README.md): 15 input features.
- [Frozen value estimate](value_estimate/README.md): 1 input features.

Each directory contains per-policy readouts/partitions, ninety individual profile plots, a multipage PDF, combined all-profile figures, category averages, and recomputed early-behavior/motion/correlation diagnostics.

Comparison CSVs and figures are in this directory. completion_audit.json records the final verification.

Validation: all eight completion audits passed; the final inventory and original-baseline preservation checks are in [final_inventory_check.json](final_inventory_check.json). Exact category counts are in [category_denominators.csv](category_denominators.csv). Shared action/reward figures link to the original frozen-policy analysis.

The updated-memory rerun differs from the archived baseline on 18 of 4,330,800 held-out episode–timestep decisions (0.000416%); its maximum absolute class-probability difference is 0.012020. The original training states and partitions are exact; the original outputs remain unchanged. Per-policy comparisons are in [updated_memory/baseline_regression_summary.json](updated_memory/baseline_regression_summary.json).

Comparison figures:

- [Influence, first 60 steps](representation_comparison_rnn_diverse_influence_first60.png) · [all 400 steps](representation_comparison_rnn_diverse_influence_all400.png)
- [No influence, first 60 steps](representation_comparison_rnn_diverse_noinfluence_first60.png) · [all 400 steps](representation_comparison_rnn_diverse_noinfluence_all400.png)
- [Full comparison table](representation_comparison.csv) · [Selected steps](representation_comparison_selected_steps.csv)
