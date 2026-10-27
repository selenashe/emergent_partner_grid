# v1_matched_rnn_diverse_influence_seed4

Direct input + t0 fix; step penalty 0.01. Final-update training return 13.172.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed4_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed4_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v1_matched_influence_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 6.9825 | 13.9789 | 0.0157 | -0.0344 | 0.0650 | approximately flat | 104 | 0.0000 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v1_matched_rnn_diverse_influence_seed4_worst_return.png)

Layouts: train_0392;train_0107;train_0916;train_1011;train_1783;train_0375;train_1884;train_0538;train_0484;train_0524;train_1067;train_0153;train_0390;train_0435;train_0375;train_0695;train_0817;train_1054;train_1160;train_0525

### near_cutoff

![Recorded round ledger](../cases/v1_matched_rnn_diverse_influence_seed4_near_cutoff.png)

Layouts: train_1765;train_1665;train_1430;train_1307;train_0206;train_0167;train_0661;train_0482;train_0041;train_0201;train_0790;train_0582;train_1938;train_0807;train_1841;train_1505;train_0888;train_0731;train_1379;train_0151

### most_switches

![Recorded round ledger](../cases/v1_matched_rnn_diverse_influence_seed4_most_switches.png)

Layouts: train_0158;train_1923;train_0392;train_1703;train_0594;train_0311;train_0148;train_0843;train_1521;train_0091;train_0732;train_0220;train_1683;train_0932;train_1361;train_1592;train_0960;train_0754;train_0365;train_0705

### largest_late_gain

![Recorded round ledger](../cases/v1_matched_rnn_diverse_influence_seed4_largest_late_gain.png)

Layouts: train_0375;train_0861;train_0562;train_1358;train_0296;train_0097;train_1925;train_0714;train_1421;train_0091;train_1746;train_0774;train_1182;train_0443;train_0064;train_0310;train_0855;train_1150;train_1805;train_0708

