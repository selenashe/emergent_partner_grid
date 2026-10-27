# v2_rnn_diverse_influence_seed4

Direct input + t0 fix; step penalty 0.01. Final-update training return 13.726.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed4_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed4_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v2_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 8.6296 | 14.2825 | 0.0009 | -0.0598 | 0.0630 | approximately flat | 152 | 1.0614 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed4_worst_return.png)

Layouts: train_0651;train_1462;train_1115;train_0304;train_1643;train_0211;train_1469;train_1601;train_0048;train_1600;train_0955;train_1655;train_1348;train_0148;train_0532;train_1848;train_0363;train_1884;train_1622;train_0577

### near_cutoff

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed4_near_cutoff.png)

Layouts: train_1278;train_1371;train_1455;train_1679;train_1612;train_1114;train_1931;train_1665;train_0709;train_1741;train_0887;train_1917;train_0947;train_0783;train_0583;train_0661;train_1001;train_0609;train_0787;train_1645

### most_switches

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed4_most_switches.png)

Layouts: train_0298;train_1038;train_0856;train_1242;train_1031;train_0155;train_0237;train_0212;train_1581;train_1612;train_1643;train_1095;train_1921;train_1171;train_1913;train_0634;train_0346;train_1068;train_1479;train_0563

### largest_late_gain

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed4_largest_late_gain.png)

Layouts: train_1278;train_0152;train_1428;train_1003;train_1354;train_1838;train_0639;train_0951;train_1573;train_0995;train_0943;train_0769;train_0671;train_1469;train_0919;train_1242;train_1894;train_1910;train_0358;train_1080

