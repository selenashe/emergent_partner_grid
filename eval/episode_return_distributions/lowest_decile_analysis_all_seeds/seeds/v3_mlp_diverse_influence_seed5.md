# v3_mlp_diverse_influence_seed5

Direct input + t0 fix; step penalty 0.01. Final-update training return 12.715.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed5_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed5_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v3_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 7.2060 | 13.9665 | -0.0145 | -0.0574 | 0.0280 | approximately flat | 99 | 0.0473 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v3_mlp_diverse_influence_seed5_worst_return.png)

Layouts: train_0392;train_0107;train_0916;train_1011;train_1783;train_0375;train_1884;train_0538;train_0484;train_0524;train_1067;train_0153;train_0390;train_0435;train_0375;train_0695;train_0817;train_1054;train_1160;train_0525

### near_cutoff

![Recorded round ledger](../cases/v3_mlp_diverse_influence_seed5_near_cutoff.png)

Layouts: train_1734;train_1806;train_1941;train_1053;train_1001;train_1965;train_1671;train_1954;train_0349;train_1878;train_0516;train_0707;train_0600;train_1281;train_0612;train_0282;train_0209;train_0523;train_0186;train_1354

### most_switches

![Recorded round ledger](../cases/v3_mlp_diverse_influence_seed5_most_switches.png)

Layouts: train_0745;train_1406;train_0616;train_1987;train_1327;train_1117;train_1865;train_0149;train_0921;train_0441;train_1468;train_1455;train_0075;train_1514;train_1246;train_0139;train_0974;train_0970;train_1670;train_0362

### largest_late_gain

![Recorded round ledger](../cases/v3_mlp_diverse_influence_seed5_largest_late_gain.png)

Layouts: train_1701;train_0137;train_0218;train_1212;train_0403;train_0088;train_1670;train_1574;train_0897;train_1454;train_1952;train_0685;train_1637;train_1295;train_1859;train_1212;train_0226;train_1690;train_0815;train_1156

