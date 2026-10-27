# v2_mlp_diverse_influence_seed5

Direct input + t0 fix; step penalty 0.01. Final-update training return 12.123.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed5_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed5_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v2_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 6.9229 | 13.8753 | -0.0271 | -0.0769 | 0.0236 | approximately flat | 104 | 0.5527 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v2_mlp_diverse_influence_seed5_worst_return.png)

Layouts: train_0064;train_1315;train_0366;train_0952;train_1403;train_0216;train_0260;train_1662;train_1952;train_1478;train_1624;train_1047;train_1416;train_0164;train_1720;train_1512;train_0341;train_0341;train_1083;train_1781

### near_cutoff

![Recorded round ledger](../cases/v2_mlp_diverse_influence_seed5_near_cutoff.png)

Layouts: train_0938;train_0354;train_0703;train_1977;train_1417;train_1605;train_0789;train_1853;train_0597;train_1726;train_0638;train_1621;train_1451;train_0699;train_0126;train_1394;train_0392;train_1262;train_0104;train_0075

### most_switches

![Recorded round ledger](../cases/v2_mlp_diverse_influence_seed5_most_switches.png)

Layouts: train_1079;train_0421;train_1741;train_0289;train_1900;train_1748;train_1800;train_1554;train_1083;train_1427;train_0927;train_0075;train_1491;train_0466;train_0954;train_1535;train_1720;train_0546;train_1363;train_1771

### largest_late_gain

![Recorded round ledger](../cases/v2_mlp_diverse_influence_seed5_largest_late_gain.png)

Layouts: train_1145;train_0864;train_1665;train_1182;train_1700;train_1498;train_1005;train_0282;train_1244;train_0528;train_0030;train_1041;train_1520;train_0714;train_0970;train_1001;train_1380;train_1873;train_1316;train_1410

