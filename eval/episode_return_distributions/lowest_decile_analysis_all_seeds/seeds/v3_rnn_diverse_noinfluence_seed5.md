# v3_rnn_diverse_noinfluence_seed5

Direct input + t0 fix; step penalty 0.01. Final-update training return 11.774.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_noinfluence_seed5_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_noinfluence_seed5_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v3_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 7.7091 | 11.9624 | -0.0217 | -0.1054 | 0.0654 | approximately flat | 247 | 0.0000 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v3_rnn_diverse_noinfluence_seed5_worst_return.png)

Layouts: train_0787;train_0610;train_0089;train_1452;train_1923;train_0643;train_0688;train_1858;train_1611;train_0685;train_0803;train_1421;train_1812;train_1158;train_1652;train_1483;train_0115;train_1530;train_1897;train_1464

### near_cutoff

![Recorded round ledger](../cases/v3_rnn_diverse_noinfluence_seed5_near_cutoff.png)

Layouts: train_0191;train_0335;train_1882;train_0988;train_1312;train_0384;train_1396;train_1783;train_0342;train_0763;train_0134;train_0955;train_1723;train_1377;train_0660;train_0923;train_1521;train_1499;train_1071;train_0692

### most_switches

![Recorded round ledger](../cases/v3_rnn_diverse_noinfluence_seed5_most_switches.png)

Layouts: train_1307;train_1862;train_1712;train_1021;train_1514;train_1379;train_1011;train_0252;train_1771;train_0530;train_0694;train_1840;train_0457;train_0537;train_0932;train_1328;train_0900;train_1235;train_1114;train_1803

### largest_late_gain

![Recorded round ledger](../cases/v3_rnn_diverse_noinfluence_seed5_largest_late_gain.png)

Layouts: train_1864;train_1840;train_1528;train_1573;train_1338;train_1658;train_1930;train_1016;train_1118;train_1469;train_0163;train_1964;train_0343;train_0230;train_1111;train_1796;train_1577;train_1554;train_0300;train_1029

