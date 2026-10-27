# v1_matched_mlp_diverse_influence_seed4

Direct input + t0 fix; step penalty 0.01. Final-update training return 12.942.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed4_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed4_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v1_matched_influence_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 6.9478 | 13.9805 | 0.0003 | -0.0488 | 0.0525 | approximately flat | 114 | 0.0000 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v1_matched_mlp_diverse_influence_seed4_worst_return.png)

Layouts: train_0259;train_0558;train_0953;train_0770;train_1873;train_0421;train_0218;train_1579;train_0745;train_0319;train_1718;train_0058;train_1146;train_1149;train_0292;train_1665;train_0638;train_1523;train_0172;train_0319

### near_cutoff

![Recorded round ledger](../cases/v1_matched_mlp_diverse_influence_seed4_near_cutoff.png)

Layouts: train_1016;train_1901;train_1108;train_0594;train_0365;train_0227;train_0062;train_1897;train_1939;train_0211;train_1228;train_0573;train_1362;train_1940;train_1120;train_0153;train_0872;train_1625;train_1359;train_1601

### most_switches

![Recorded round ledger](../cases/v1_matched_mlp_diverse_influence_seed4_most_switches.png)

Layouts: train_0713;train_1947;train_1114;train_0532;train_0365;train_1110;train_0062;train_1879;train_1464;train_0226;train_0964;train_1517;train_0739;train_1934;train_1556;train_1358;train_0612;train_0819;train_0439;train_1506

### largest_late_gain

![Recorded round ledger](../cases/v1_matched_mlp_diverse_influence_seed4_largest_late_gain.png)

Layouts: train_1145;train_0864;train_1665;train_1182;train_1700;train_1498;train_1005;train_0282;train_1244;train_0528;train_0030;train_1041;train_1520;train_0714;train_0970;train_1001;train_1380;train_1873;train_1316;train_1410

