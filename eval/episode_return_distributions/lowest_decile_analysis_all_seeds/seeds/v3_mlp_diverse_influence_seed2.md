# v3_mlp_diverse_influence_seed2

Direct input + t0 fix; step penalty 0.01. Final-update training return 13.120.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed2_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed2_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v3_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 7.5888 | 14.1137 | -0.0093 | -0.0510 | 0.0327 | approximately flat | 81 | 0.0478 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v3_mlp_diverse_influence_seed2_worst_return.png)

Layouts: train_0921;train_1030;train_0240;train_1194;train_0772;train_1658;train_1009;train_1996;train_0783;train_0393;train_0658;train_1605;train_0097;train_0439;train_1035;train_1202;train_0260;train_1684;train_0551;train_1613

### near_cutoff

![Recorded round ledger](../cases/v3_mlp_diverse_influence_seed2_near_cutoff.png)

Layouts: train_0301;train_1340;train_0129;train_1411;train_1990;train_1395;train_1035;train_0490;train_0783;train_0660;train_1907;train_1998;train_0621;train_0744;train_0982;train_1897;train_1689;train_0222;train_0433;train_0394

### most_switches

![Recorded round ledger](../cases/v3_mlp_diverse_influence_seed2_most_switches.png)

Layouts: train_1349;train_0303;train_0176;train_1054;train_1268;train_0263;train_1372;train_1464;train_1518;train_0102;train_1432;train_1348;train_1509;train_0558;train_1534;train_1959;train_0611;train_0587;train_0889;train_0506

### largest_late_gain

![Recorded round ledger](../cases/v3_mlp_diverse_influence_seed2_largest_late_gain.png)

Layouts: train_1655;train_1371;train_1140;train_0505;train_1340;train_1244;train_1040;train_0226;train_0995;train_1332;train_1765;train_1431;train_0490;train_1353;train_1615;train_0583;train_1165;train_0065;train_1658;train_0537

