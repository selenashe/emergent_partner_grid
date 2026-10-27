# v2_rnn_diverse_influence_seed2

Direct input + t0 fix; step penalty 0.01. Final-update training return 14.238.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed2_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed2_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v2_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 11.5477 | 14.9413 | -0.0217 | -0.0745 | 0.0307 | approximately flat | 104 | 0.7875 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed2_worst_return.png)

Layouts: train_1212;train_0991;train_0803;train_1712;train_1869;train_0304;train_1252;train_1613;train_0523;train_0739;train_1864;train_0656;train_0980;train_0015;train_0875;train_0055;train_0314;train_0809;train_0303;train_0809

### near_cutoff

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed2_near_cutoff.png)

Layouts: train_1114;train_1862;train_1218;train_0796;train_0958;train_1896;train_1853;train_0567;train_0228;train_0168;train_0819;train_1008;train_0402;train_1734;train_1103;train_1778;train_1468;train_1688;train_1689;train_1453

### most_switches

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed2_most_switches.png)

Layouts: train_0556;train_0961;train_0436;train_1171;train_1715;train_1627;train_1043;train_0961;train_1316;train_1341;train_1509;train_0296;train_1732;train_1950;train_0341;train_1270;train_1035;train_1671;train_1031;train_1003

### largest_late_gain

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed2_largest_late_gain.png)

Layouts: train_0485;train_1266;train_0989;train_0786;train_0968;train_1362;train_0152;train_1953;train_1052;train_0953;train_1978;train_0714;train_0685;train_1663;train_0446;train_1266;train_0460;train_1041;train_0081;train_0495

