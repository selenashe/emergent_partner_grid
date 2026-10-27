# v2_mlp_diverse_influence_seed1

Direct input + t0 fix; step penalty 0.01. Final-update training return 12.609.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed1_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed1_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v2_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5.6757 | 13.5942 | -0.0132 | -0.0708 | 0.0456 | approximately flat | 147 | 0.5310 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v2_mlp_diverse_influence_seed1_worst_return.png)

Layouts: train_0375;train_0861;train_0562;train_1358;train_0296;train_0097;train_1925;train_0714;train_1421;train_0091;train_1746;train_0774;train_1182;train_0443;train_0064;train_0310;train_0855;train_1150;train_1805;train_0708

### near_cutoff

![Recorded round ledger](../cases/v2_mlp_diverse_influence_seed1_near_cutoff.png)

Layouts: train_1030;train_1913;train_1944;train_1940;train_0457;train_0074;train_1495;train_1679;train_1052;train_1331;train_0661;train_1890;train_1508;train_0271;train_0247;train_1629;train_0651;train_1496;train_0083;train_1120

### most_switches

![Recorded round ledger](../cases/v2_mlp_diverse_influence_seed1_most_switches.png)

Layouts: train_1180;train_1281;train_1320;train_1894;train_1652;train_0879;train_1975;train_0912;train_1566;train_1650;train_0753;train_0902;train_1295;train_0921;train_1876;train_0339;train_0204;train_1624;train_1502;train_0546

### largest_late_gain

![Recorded round ledger](../cases/v2_mlp_diverse_influence_seed1_largest_late_gain.png)

Layouts: train_1655;train_1371;train_1140;train_0505;train_1340;train_1244;train_1040;train_0226;train_0995;train_1332;train_1765;train_1431;train_0490;train_1353;train_1615;train_0583;train_1165;train_0065;train_1658;train_0537

