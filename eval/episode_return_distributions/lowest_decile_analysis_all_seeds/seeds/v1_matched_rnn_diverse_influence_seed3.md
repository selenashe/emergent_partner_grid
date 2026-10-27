# v1_matched_rnn_diverse_influence_seed3

Direct input + t0 fix; step penalty 0.01. Final-update training return 12.831.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed3_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed3_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v1_matched_influence_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 6.8496 | 14.0423 | -0.0130 | -0.0672 | 0.0420 | approximately flat | 120 | 0.0000 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v1_matched_rnn_diverse_influence_seed3_worst_return.png)

Layouts: train_0259;train_0558;train_0953;train_0770;train_1873;train_0421;train_0218;train_1579;train_0745;train_0319;train_1718;train_0058;train_1146;train_1149;train_0292;train_1665;train_0638;train_1523;train_0172;train_0319

### near_cutoff

![Recorded round ledger](../cases/v1_matched_rnn_diverse_influence_seed3_near_cutoff.png)

Layouts: train_1747;train_0415;train_0856;train_1532;train_1453;train_1616;train_1878;train_0006;train_1700;train_0412;train_0629;train_1617;train_0922;train_1448;train_0196;train_1924;train_0643;train_1495;train_1840;train_1003

### most_switches

![Recorded round ledger](../cases/v1_matched_rnn_diverse_influence_seed3_most_switches.png)

Layouts: train_0506;train_0055;train_0492;train_0730;train_1643;train_1938;train_0999;train_1010;train_0287;train_0923;train_0774;train_1310;train_0530;train_0448;train_1587;train_0951;train_1814;train_0711;train_1463;train_0671

### largest_late_gain

![Recorded round ledger](../cases/v1_matched_rnn_diverse_influence_seed3_largest_late_gain.png)

Layouts: train_1145;train_0864;train_1665;train_1182;train_1700;train_1498;train_1005;train_0282;train_1244;train_0528;train_0030;train_1041;train_1520;train_0714;train_0970;train_1001;train_1380;train_1873;train_1316;train_1410

