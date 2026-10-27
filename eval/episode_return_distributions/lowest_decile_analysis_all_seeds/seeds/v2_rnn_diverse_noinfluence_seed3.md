# v2_rnn_diverse_noinfluence_seed3

Direct input + t0 fix; step penalty 0.01. Final-update training return 11.211.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_noinfluence_seed3_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_noinfluence_seed3_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v2_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 7.3553 | 12.0898 | 0.0051 | -0.0683 | 0.0803 | approximately flat | 247 | 0.0000 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v2_rnn_diverse_noinfluence_seed3_worst_return.png)

Layouts: train_0524;train_0546;train_1254;train_1630;train_1525;train_1130;train_0524;train_1580;train_0310;train_1509;train_0968;train_1359;train_0233;train_1730;train_0807;train_1688;train_0109;train_0752;train_1896;train_0640

### near_cutoff

![Recorded round ledger](../cases/v2_rnn_diverse_noinfluence_seed3_near_cutoff.png)

Layouts: train_1110;train_0692;train_0486;train_1924;train_1940;train_0405;train_1723;train_0939;train_1535;train_1372;train_1453;train_1762;train_1960;train_1372;train_0167;train_1165;train_0881;train_1054;train_0331;train_1349

### most_switches

![Recorded round ledger](../cases/v2_rnn_diverse_noinfluence_seed3_most_switches.png)

Layouts: train_0599;train_1604;train_1554;train_0108;train_1447;train_1242;train_1247;train_0019;train_1487;train_1263;train_0687;train_1605;train_0194;train_0098;train_1227;train_0733;train_0582;train_1766;train_0628;train_0212

### largest_late_gain

![Recorded round ledger](../cases/v2_rnn_diverse_noinfluence_seed3_largest_late_gain.png)

Layouts: train_1952;train_1905;train_1611;train_0629;train_1354;train_0065;train_1592;train_1062;train_0640;train_0218;train_1421;train_1383;train_0804;train_1791;train_0528;train_1213;train_1191;train_1888;train_1828;train_1864

