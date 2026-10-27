# v1_matched_rnn_diverse_noinfluence_seed3

Direct input + t0 fix; step penalty 0.01. Final-update training return 11.627.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_noinfluence_seed3_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_noinfluence_seed3_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v1_matched_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 8.0095 | 12.3417 | 0.0522 | -0.0136 | 0.1193 | no clear trend | 231 | 0.0000 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v1_matched_rnn_diverse_noinfluence_seed3_worst_return.png)

Layouts: train_1409;train_0138;train_0843;train_1108;train_0787;train_1939;train_1518;train_1497;train_1974;train_0281;train_1587;train_1509;train_1355;train_0585;train_0006;train_1590;train_1737;train_1353;train_0437;train_0301

### near_cutoff

![Recorded round ledger](../cases/v1_matched_rnn_diverse_noinfluence_seed3_near_cutoff.png)

Layouts: train_0330;train_1481;train_1511;train_0824;train_0119;train_1345;train_1396;train_0455;train_1192;train_0912;train_1162;train_0940;train_0629;train_1108;train_1179;train_0089;train_1791;train_1160;train_1111;train_1952

### most_switches

![Recorded round ledger](../cases/v1_matched_rnn_diverse_noinfluence_seed3_most_switches.png)

Layouts: train_0577;train_0726;train_1509;train_1869;train_1780;train_0609;train_0887;train_1427;train_0634;train_0462;train_1472;train_0245;train_1271;train_0907;train_1750;train_0878;train_1225;train_1967;train_1570;train_0770

### largest_late_gain

![Recorded round ledger](../cases/v1_matched_rnn_diverse_noinfluence_seed3_largest_late_gain.png)

Layouts: train_1463;train_1637;train_1409;train_0343;train_1620;train_1349;train_1115;train_1746;train_0209;train_0981;train_0271;train_1860;train_0651;train_0153;train_1225;train_1872;train_0544;train_1671;train_0794;train_0002

