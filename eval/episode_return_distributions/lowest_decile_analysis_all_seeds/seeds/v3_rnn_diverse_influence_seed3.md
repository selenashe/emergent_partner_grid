# v3_rnn_diverse_influence_seed3

Direct input + t0 fix; step penalty 0.01. Final-update training return 14.811.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed3_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed3_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v3_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 11.9923 | 15.4506 | -0.0477 | -0.0985 | 0.0014 | approximately flat | 79 | 0.2777 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v3_rnn_diverse_influence_seed3_worst_return.png)

Layouts: train_0079;train_0175;train_0559;train_1521;train_0049;train_0155;train_1581;train_0632;train_0473;train_0556;train_1762;train_0259;train_1824;train_1358;train_0810;train_0430;train_1645;train_0532;train_0517;train_1405

### near_cutoff

![Recorded round ledger](../cases/v3_rnn_diverse_influence_seed3_near_cutoff.png)

Layouts: train_1446;train_0952;train_1131;train_1770;train_1130;train_0343;train_0460;train_0567;train_0785;train_0310;train_0169;train_1386;train_0134;train_0556;train_1924;train_0158;train_1551;train_0031;train_0153;train_1613

### most_switches

![Recorded round ledger](../cases/v3_rnn_diverse_influence_seed3_most_switches.png)

Layouts: train_0058;train_1853;train_0658;train_0590;train_0175;train_1143;train_0220;train_1112;train_1778;train_0282;train_0770;train_0506;train_1931;train_0697;train_1615;train_0990;train_0151;train_0083;train_0065;train_0221

### largest_late_gain

![Recorded round ledger](../cases/v3_rnn_diverse_influence_seed3_largest_late_gain.png)

Layouts: train_0778;train_1372;train_1926;train_0298;train_1117;train_1732;train_1469;train_1562;train_1762;train_0396;train_0902;train_1772;train_1179;train_1444;train_1700;train_0796;train_0180;train_1684;train_1676;train_1307

