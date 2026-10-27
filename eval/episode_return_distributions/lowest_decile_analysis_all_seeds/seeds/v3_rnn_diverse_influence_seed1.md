# v3_rnn_diverse_influence_seed1

Direct input + t0 fix; step penalty 0.01. Final-update training return 12.946.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed1_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed1_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v3_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 7.2087 | 13.9471 | 0.0625 | 0.0060 | 0.1192 | increasing | 110 | 0.3245 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v3_rnn_diverse_influence_seed1_worst_return.png)

Layouts: train_1145;train_0864;train_1665;train_1182;train_1700;train_1498;train_1005;train_0282;train_1244;train_0528;train_0030;train_1041;train_1520;train_0714;train_0970;train_1001;train_1380;train_1873;train_1316;train_1410

### near_cutoff

![Recorded round ledger](../cases/v3_rnn_diverse_influence_seed1_near_cutoff.png)

Layouts: train_0528;train_1571;train_0661;train_1005;train_0153;train_0739;train_0695;train_0165;train_0936;train_0708;train_1254;train_1876;train_1213;train_1663;train_1593;train_0562;train_1263;train_1130;train_1355;train_1429

### most_switches

![Recorded round ledger](../cases/v3_rnn_diverse_influence_seed1_most_switches.png)

Layouts: train_1577;train_1463;train_1043;train_0005;train_0527;train_0435;train_0638;train_0052;train_1743;train_0648;train_0166;train_1310;train_0108;train_0571;train_0990;train_0025;train_0709;train_1770;train_0473;train_1483

### largest_late_gain

![Recorded round ledger](../cases/v3_rnn_diverse_influence_seed1_largest_late_gain.png)

Layouts: train_1936;train_1487;train_0790;train_1126;train_1406;train_0551;train_0968;train_0907;train_0935;train_0964;train_0485;train_1289;train_0637;train_0609;train_1401;train_0685;train_0685;train_1499;train_0888;train_0074

