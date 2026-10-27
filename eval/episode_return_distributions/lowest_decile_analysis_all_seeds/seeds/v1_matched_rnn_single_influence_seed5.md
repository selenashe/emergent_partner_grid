# v1_matched_rnn_single_influence_seed5

Direct input + t0 fix; step penalty 0.01. Final-update training return 17.333.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed5_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed5_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v1_matched_influence_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 6.6446 | 14.1280 | -0.1143 | -0.1725 | -0.0556 | decreasing | 127 | 0.0000 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v1_matched_rnn_single_influence_seed5_worst_return.png)

Layouts: train_0789;train_1462;train_1001;train_1504;train_1601;train_0572;train_0778;train_0237;train_0935;train_1496;train_0030;train_0147;train_0514;train_0082;train_1062;train_0138;train_0139;train_0222;train_0048;train_1140

### near_cutoff

![Recorded round ledger](../cases/v1_matched_rnn_single_influence_seed5_near_cutoff.png)

Layouts: train_1701;train_0137;train_0218;train_1212;train_0403;train_0088;train_1670;train_1574;train_0897;train_1454;train_1952;train_0685;train_1637;train_1295;train_1859;train_1212;train_0226;train_1690;train_0815;train_1156

### most_switches

![Recorded round ledger](../cases/v1_matched_rnn_single_influence_seed5_most_switches.png)

Layouts: train_0388;train_0074;train_1445;train_1650;train_0343;train_0879;train_1184;train_0809;train_1313;train_1535;train_0214;train_0384;train_0522;train_0999;train_1655;train_1448;train_0873;train_0537;train_1665;train_1587

### largest_late_gain

![Recorded round ledger](../cases/v1_matched_rnn_single_influence_seed5_largest_late_gain.png)

Layouts: train_0375;train_0861;train_0562;train_1358;train_0296;train_0097;train_1925;train_0714;train_1421;train_0091;train_1746;train_0774;train_1182;train_0443;train_0064;train_0310;train_0855;train_1150;train_1805;train_0708

