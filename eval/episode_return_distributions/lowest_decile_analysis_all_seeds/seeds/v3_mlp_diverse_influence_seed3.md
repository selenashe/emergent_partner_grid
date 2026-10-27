# v3_mlp_diverse_influence_seed3

Direct input + t0 fix; step penalty 0.01. Final-update training return 13.358.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed3_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed3_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v3_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 7.1793 | 13.9961 | -0.0210 | -0.0643 | 0.0242 | approximately flat | 98 | 0.0201 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v3_mlp_diverse_influence_seed3_worst_return.png)

Layouts: train_1859;train_0079;train_1703;train_1806;train_1964;train_0981;train_0675;train_1015;train_0041;train_0430;train_0569;train_0168;train_1558;train_1220;train_0419;train_1791;train_0292;train_1152;train_1911;train_1780

### near_cutoff

![Recorded round ledger](../cases/v3_mlp_diverse_influence_seed3_near_cutoff.png)

Layouts: train_1212;train_0991;train_0803;train_1712;train_1869;train_0304;train_1252;train_1613;train_0523;train_0739;train_1864;train_0656;train_0980;train_0015;train_0875;train_0055;train_0314;train_0809;train_0303;train_0809

### most_switches

![Recorded round ledger](../cases/v3_mlp_diverse_influence_seed3_most_switches.png)

Layouts: train_0281;train_0328;train_1732;train_0784;train_1597;train_1323;train_0733;train_0979;train_0528;train_1312;train_0567;train_0940;train_0332;train_0247;train_1118;train_1439;train_0634;train_1613;train_1613;train_0952

### largest_late_gain

![Recorded round ledger](../cases/v3_mlp_diverse_influence_seed3_largest_late_gain.png)

Layouts: train_0375;train_0861;train_0562;train_1358;train_0296;train_0097;train_1925;train_0714;train_1421;train_0091;train_1746;train_0774;train_1182;train_0443;train_0064;train_0310;train_0855;train_1150;train_1805;train_0708

