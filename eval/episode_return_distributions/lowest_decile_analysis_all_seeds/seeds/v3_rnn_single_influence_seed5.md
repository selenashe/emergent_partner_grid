# v3_rnn_single_influence_seed5

Direct input + t0 fix; step penalty 0.01. Final-update training return 17.436.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed5_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed5_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v3_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 5.6642 | 14.1752 | -0.3143 | -0.3849 | -0.2430 | decreasing | 185 | 1.4348 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v3_rnn_single_influence_seed5_worst_return.png)

Layouts: train_0921;train_1030;train_0240;train_1194;train_0772;train_1658;train_1009;train_1996;train_0783;train_0393;train_0658;train_1605;train_0097;train_0439;train_1035;train_1202;train_0260;train_1684;train_0551;train_1613

### near_cutoff

![Recorded round ledger](../cases/v3_rnn_single_influence_seed5_near_cutoff.png)

Layouts: train_1218;train_0233;train_0890;train_0897;train_1595;train_1052;train_0097;train_1996;train_0459;train_0863;train_1455;train_0796;train_1451;train_0288;train_1689;train_0632;train_0750;train_1081;train_0593;train_0534

### most_switches

![Recorded round ledger](../cases/v3_rnn_single_influence_seed5_most_switches.png)

Layouts: train_0451;train_1472;train_0172;train_0745;train_0583;train_1306;train_0513;train_1865;train_1794;train_1676;train_1247;train_0111;train_0169;train_1492;train_0089;train_0190;train_0877;train_1341;train_1299;train_0176

### largest_late_gain

![Recorded round ledger](../cases/v3_rnn_single_influence_seed5_largest_late_gain.png)

Layouts: train_1701;train_0137;train_0218;train_1212;train_0403;train_0088;train_1670;train_1574;train_0897;train_1454;train_1952;train_0685;train_1637;train_1295;train_1859;train_1212;train_0226;train_1690;train_0815;train_1156

