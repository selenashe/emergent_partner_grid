# v3_mlp_diverse_influence_seed1

Direct input + t0 fix; step penalty 0.01. Final-update training return 12.906.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed1_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed1_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v3_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 7.0771 | 14.0109 | -0.0014 | -0.0484 | 0.0494 | approximately flat | 98 | 0.0103 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v3_mlp_diverse_influence_seed1_worst_return.png)

Layouts: train_0375;train_0861;train_0562;train_1358;train_0296;train_0097;train_1925;train_0714;train_1421;train_0091;train_1746;train_0774;train_1182;train_0443;train_0064;train_0310;train_0855;train_1150;train_1805;train_0708

### near_cutoff

![Recorded round ledger](../cases/v3_mlp_diverse_influence_seed1_near_cutoff.png)

Layouts: train_1998;train_1429;train_0045;train_0661;train_0175;train_1115;train_1103;train_0331;train_1435;train_0000;train_1070;train_0577;train_0312;train_0526;train_0572;train_0658;train_0176;train_0067;train_0777;train_0005

### most_switches

![Recorded round ledger](../cases/v3_mlp_diverse_influence_seed1_most_switches.png)

Layouts: train_0516;train_1504;train_0713;train_0932;train_0974;train_0381;train_1458;train_1497;train_0521;train_0378;train_1769;train_1358;train_0938;train_1295;train_0352;train_1839;train_1517;train_1341;train_1574;train_1054

### largest_late_gain

![Recorded round ledger](../cases/v3_mlp_diverse_influence_seed1_largest_late_gain.png)

Layouts: train_1701;train_0137;train_0218;train_1212;train_0403;train_0088;train_1670;train_1574;train_0897;train_1454;train_1952;train_0685;train_1637;train_1295;train_1859;train_1212;train_0226;train_1690;train_0815;train_1156

