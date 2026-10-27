# v3_rnn_single_influence_seed4

Direct input + t0 fix; step penalty 0.01. Final-update training return 17.350.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed4_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed4_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v3_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 4.9672 | 13.9207 | -0.3967 | -0.4493 | -0.3435 | decreasing | 221 | 1.9625 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v3_rnn_single_influence_seed4_worst_return.png)

Layouts: train_1388;train_1358;train_1380;train_0378;train_1718;train_0222;train_1408;train_0769;train_1664;train_1554;train_0125;train_1320;train_1670;train_0810;train_1800;train_1934;train_0079;train_0068;train_1137;train_0058

### near_cutoff

![Recorded round ledger](../cases/v3_rnn_single_influence_seed4_near_cutoff.png)

Layouts: train_1944;train_1936;train_1978;train_0972;train_0470;train_0851;train_1404;train_1520;train_1103;train_1709;train_0791;train_1070;train_0621;train_0643;train_1952;train_0946;train_1376;train_0544;train_0675;train_1896

### most_switches

![Recorded round ledger](../cases/v3_rnn_single_influence_seed4_most_switches.png)

Layouts: train_0593;train_1531;train_0006;train_1238;train_0528;train_0628;train_0261;train_0962;train_0325;train_1927;train_1455;train_0571;train_1887;train_0817;train_1366;train_1703;train_1168;train_0704;train_1939;train_0379

### largest_late_gain

![Recorded round ledger](../cases/v3_rnn_single_influence_seed4_largest_late_gain.png)

Layouts: train_1386;train_0972;train_1016;train_1427;train_1535;train_0153;train_1383;train_1649;train_1897;train_1502;train_0711;train_1363;train_0538;train_1416;train_1067;train_1715;train_0945;train_0047;train_0339;train_0271

