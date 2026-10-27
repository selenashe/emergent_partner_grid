# v1_matched_rnn_diverse_influence_seed5

Direct input + t0 fix; step penalty 0.01. Final-update training return 14.014.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed5_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed5_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v1_matched_influence_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 11.2135 | 15.0884 | 0.0103 | -0.0597 | 0.0785 | approximately flat | 154 | 0.0000 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v1_matched_rnn_diverse_influence_seed5_worst_return.png)

Layouts: train_0082;train_0202;train_0665;train_1805;train_0858;train_1900;train_1936;train_0352;train_1143;train_1865;train_1730;train_0263;train_0461;train_0506;train_1439;train_1392;train_1159;train_1535;train_1523;train_0157

### near_cutoff

![Recorded round ledger](../cases/v1_matched_rnn_diverse_influence_seed5_near_cutoff.png)

Layouts: train_0178;train_0871;train_1238;train_1252;train_1112;train_1525;train_1982;train_0378;train_0230;train_0964;train_1351;train_1182;train_1977;train_1231;train_0587;train_1348;train_1987;train_1458;train_0621;train_0218

### most_switches

![Recorded round ledger](../cases/v1_matched_rnn_diverse_influence_seed5_most_switches.png)

Layouts: train_1281;train_1152;train_1324;train_1041;train_1150;train_0122;train_0900;train_0410;train_0815;train_1307;train_1122;train_1921;train_1327;train_0530;train_0619;train_1031;train_0377;train_0269;train_1511;train_0190

### largest_late_gain

![Recorded round ledger](../cases/v1_matched_rnn_diverse_influence_seed5_largest_late_gain.png)

Layouts: train_1514;train_0109;train_0019;train_1392;train_0790;train_0264;train_1359;train_1005;train_1075;train_1577;train_0104;train_0037;train_0180;train_1532;train_0002;train_0151;train_1338;train_0739;train_0637;train_1530

