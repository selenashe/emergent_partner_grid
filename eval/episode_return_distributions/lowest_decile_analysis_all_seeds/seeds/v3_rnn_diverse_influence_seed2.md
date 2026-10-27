# v3_rnn_diverse_influence_seed2

Direct input + t0 fix; step penalty 0.01. Final-update training return 13.241.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed2_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed2_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v3_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 8.3405 | 14.3195 | -0.0014 | -0.0752 | 0.0791 | approximately flat | 167 | 1.0304 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v3_rnn_diverse_influence_seed2_worst_return.png)

Layouts: train_0430;train_1888;train_0115;train_0227;train_1416;train_1744;train_1268;train_0583;train_0064;train_1451;train_1281;train_0433;train_1281;train_0701;train_0189;train_0884;train_0688;train_1237;train_0843;train_0587

### near_cutoff

![Recorded round ledger](../cases/v3_rnn_diverse_influence_seed2_near_cutoff.png)

Layouts: train_0327;train_1926;train_1024;train_0292;train_0661;train_1748;train_1224;train_0218;train_0316;train_1183;train_1120;train_0186;train_0296;train_1928;train_0451;train_1512;train_1263;train_0108;train_1392;train_1062

### most_switches

![Recorded round ledger](../cases/v3_rnn_diverse_influence_seed2_most_switches.png)

Layouts: train_0896;train_1726;train_1481;train_0610;train_1737;train_0310;train_0089;train_0419;train_1574;train_1074;train_0516;train_1487;train_0270;train_0526;train_1307;train_0603;train_1770;train_1791;train_0855;train_1149

### largest_late_gain

![Recorded round ledger](../cases/v3_rnn_diverse_influence_seed2_largest_late_gain.png)

Layouts: train_0259;train_0558;train_0953;train_0770;train_1873;train_0421;train_0218;train_1579;train_0745;train_0319;train_1718;train_0058;train_1146;train_1149;train_0292;train_1665;train_0638;train_1523;train_0172;train_0319

