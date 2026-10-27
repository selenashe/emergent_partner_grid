# v2_rnn_single_influence_seed4

Direct input + t0 fix; step penalty 0.01. Final-update training return 17.314.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed4_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed4_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v2_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 4.3511 | 13.7139 | -0.2615 | -0.3276 | -0.1984 | decreasing | 222 | 1.7549 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v2_rnn_single_influence_seed4_worst_return.png)

Layouts: train_0375;train_0861;train_0562;train_1358;train_0296;train_0097;train_1925;train_0714;train_1421;train_0091;train_1746;train_0774;train_1182;train_0443;train_0064;train_0310;train_0855;train_1150;train_1805;train_0708

### near_cutoff

![Recorded round ledger](../cases/v2_rnn_single_influence_seed4_near_cutoff.png)

Layouts: train_0849;train_0165;train_0603;train_1239;train_1356;train_0665;train_1115;train_0428;train_0427;train_0176;train_1081;train_1960;train_1243;train_0167;train_1894;train_1151;train_0537;train_0532;train_0955;train_0631

### most_switches

![Recorded round ledger](../cases/v2_rnn_single_influence_seed4_most_switches.png)

Layouts: train_1212;train_0377;train_1802;train_0055;train_1656;train_1416;train_0695;train_0988;train_1190;train_0352;train_0343;train_0366;train_0006;train_1359;train_0174;train_0585;train_0953;train_0444;train_0078;train_1986

### largest_late_gain

![Recorded round ledger](../cases/v2_rnn_single_influence_seed4_largest_late_gain.png)

Layouts: train_1312;train_1328;train_0805;train_0172;train_1226;train_1410;train_1316;train_0489;train_1030;train_1212;train_0433;train_1832;train_0919;train_0925;train_1656;train_0989;train_0300;train_0119;train_0220;train_0656

