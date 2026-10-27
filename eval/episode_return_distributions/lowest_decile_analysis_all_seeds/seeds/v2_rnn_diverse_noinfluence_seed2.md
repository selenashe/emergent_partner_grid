# v2_rnn_diverse_noinfluence_seed2

Direct input + t0 fix; step penalty 0.01. Final-update training return 11.586.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_noinfluence_seed2_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_noinfluence_seed2_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v2_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 7.5160 | 12.0741 | -0.0083 | -0.0831 | 0.0653 | approximately flat | 255 | 0.0000 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v2_rnn_diverse_noinfluence_seed2_worst_return.png)

Layouts: train_0375;train_0861;train_0562;train_1358;train_0296;train_0097;train_1925;train_0714;train_1421;train_0091;train_1746;train_0774;train_1182;train_0443;train_0064;train_0310;train_0855;train_1150;train_1805;train_0708

### near_cutoff

![Recorded round ledger](../cases/v2_rnn_diverse_noinfluence_seed2_near_cutoff.png)

Layouts: train_0925;train_1622;train_0426;train_1479;train_0575;train_0995;train_0672;train_0139;train_1009;train_0639;train_0435;train_0589;train_0045;train_0803;train_1573;train_1095;train_0961;train_1589;train_1671;train_0513

### most_switches

![Recorded round ledger](../cases/v2_rnn_diverse_noinfluence_seed2_most_switches.png)

Layouts: train_1770;train_0314;train_0828;train_0809;train_0227;train_1448;train_1349;train_0878;train_0858;train_0005;train_1571;train_0787;train_0339;train_1233;train_0147;train_1013;train_1615;train_1458;train_0611;train_1392

### largest_late_gain

![Recorded round ledger](../cases/v2_rnn_diverse_noinfluence_seed2_largest_late_gain.png)

Layouts: train_1689;train_1022;train_0970;train_1937;train_1545;train_1432;train_0298;train_1940;train_0019;train_1265;train_1699;train_1458;train_1366;train_1159;train_1118;train_1040;train_0648;train_1789;train_1853;train_1926

