# v3_rnn_single_influence_seed1

Direct input + t0 fix; step penalty 0.01. Final-update training return 17.431.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed1_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed1_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v3_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4.8663 | 13.9285 | -0.2911 | -0.3563 | -0.2254 | decreasing | 228 | 2.0658 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v3_rnn_single_influence_seed1_worst_return.png)

Layouts: train_0392;train_0107;train_0916;train_1011;train_1783;train_0375;train_1884;train_0538;train_0484;train_0524;train_1067;train_0153;train_0390;train_0435;train_0375;train_0695;train_0817;train_1054;train_1160;train_0525

### near_cutoff

![Recorded round ledger](../cases/v3_rnn_single_influence_seed1_near_cutoff.png)

Layouts: train_0988;train_0443;train_1453;train_1595;train_1111;train_0583;train_0579;train_0922;train_1094;train_1114;train_0785;train_0120;train_1896;train_1884;train_0525;train_1612;train_0358;train_0446;train_1202;train_1554

### most_switches

![Recorded round ledger](../cases/v3_rnn_single_influence_seed1_most_switches.png)

Layouts: train_1589;train_0596;train_1964;train_0651;train_0651;train_0078;train_0992;train_1453;train_0879;train_1002;train_1701;train_0338;train_1505;train_0900;train_1004;train_1120;train_0922;train_0730;train_0019;train_0912

### largest_late_gain

![Recorded round ledger](../cases/v3_rnn_single_influence_seed1_largest_late_gain.png)

Layouts: train_0375;train_0861;train_0562;train_1358;train_0296;train_0097;train_1925;train_0714;train_1421;train_0091;train_1746;train_0774;train_1182;train_0443;train_0064;train_0310;train_0855;train_1150;train_1805;train_0708

