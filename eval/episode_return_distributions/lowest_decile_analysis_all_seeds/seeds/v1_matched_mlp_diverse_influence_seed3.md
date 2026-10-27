# v1_matched_mlp_diverse_influence_seed3

Direct input + t0 fix; step penalty 0.01. Final-update training return 12.974.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed3_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed3_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v1_matched_influence_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 10.2237 | 13.7555 | -0.0365 | -0.0886 | 0.0157 | approximately flat | 93 | 0.0000 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v1_matched_mlp_diverse_influence_seed3_worst_return.png)

Layouts: train_1859;train_0079;train_1703;train_1806;train_1964;train_0981;train_0675;train_1015;train_0041;train_0430;train_0569;train_0168;train_1558;train_1220;train_0419;train_1791;train_0292;train_1152;train_1911;train_1780

### near_cutoff

![Recorded round ledger](../cases/v1_matched_mlp_diverse_influence_seed3_near_cutoff.png)

Layouts: train_1262;train_0957;train_1004;train_0217;train_0562;train_1471;train_1315;train_0551;train_1567;train_0888;train_1194;train_0432;train_0709;train_1580;train_0651;train_0024;train_0209;train_0572;train_1650;train_0480

### most_switches

![Recorded round ledger](../cases/v1_matched_mlp_diverse_influence_seed3_most_switches.png)

Layouts: train_0577;train_0726;train_1509;train_1869;train_1780;train_0609;train_0887;train_1427;train_0634;train_0462;train_1472;train_0245;train_1271;train_0907;train_1750;train_0878;train_1225;train_1967;train_1570;train_0770

### largest_late_gain

![Recorded round ledger](../cases/v1_matched_mlp_diverse_influence_seed3_largest_late_gain.png)

Layouts: train_1347;train_1414;train_0362;train_1235;train_1843;train_1944;train_0792;train_1377;train_1973;train_0287;train_0278;train_1625;train_1954;train_1676;train_0139;train_0202;train_0867;train_1506;train_0455;train_1744

