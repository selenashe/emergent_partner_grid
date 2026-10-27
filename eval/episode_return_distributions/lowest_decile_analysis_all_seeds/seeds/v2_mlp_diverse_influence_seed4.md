# v2_mlp_diverse_influence_seed4

Direct input + t0 fix; step penalty 0.01. Final-update training return 13.287.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed4_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed4_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v2_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 9.1499 | 13.7955 | 0.0344 | -0.0085 | 0.0790 | approximately flat | 67 | 0.5326 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v2_mlp_diverse_influence_seed4_worst_return.png)

Layouts: train_1676;train_1005;train_1996;train_1851;train_1717;train_1297;train_1504;train_1923;train_0153;train_0927;train_0982;train_1078;train_1043;train_1665;train_0347;train_0094;train_1492;train_0410;train_0583;train_1437

### near_cutoff

![Recorded round ledger](../cases/v2_mlp_diverse_influence_seed4_near_cutoff.png)

Layouts: train_1748;train_1154;train_1854;train_0988;train_1372;train_0881;train_1735;train_0873;train_0582;train_0809;train_1052;train_0462;train_1967;train_0912;train_0130;train_0275;train_0992;train_1882;train_0656;train_0861

### most_switches

![Recorded round ledger](../cases/v2_mlp_diverse_influence_seed4_most_switches.png)

Layouts: train_0148;train_1243;train_0857;train_1998;train_1817;train_1194;train_1213;train_1614;train_0730;train_0890;train_0530;train_1247;train_0947;train_1897;train_1384;train_1810;train_0315;train_0490;train_0563;train_1998

### largest_late_gain

![Recorded round ledger](../cases/v2_mlp_diverse_influence_seed4_largest_late_gain.png)

Layouts: train_1013;train_1224;train_1528;train_1665;train_0383;train_1315;train_0109;train_1150;train_0981;train_1964;train_1174;train_0726;train_1585;train_0992;train_0687;train_1416;train_0089;train_0964;train_0473;train_1617

