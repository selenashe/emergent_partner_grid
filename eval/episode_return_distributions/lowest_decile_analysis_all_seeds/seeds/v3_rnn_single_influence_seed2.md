# v3_rnn_single_influence_seed2

Direct input + t0 fix; step penalty 0.01. Final-update training return 17.420.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed2_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed2_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v3_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 6.3438 | 14.1578 | -0.1773 | -0.2348 | -0.1180 | decreasing | 146 | 0.8168 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v3_rnn_single_influence_seed2_worst_return.png)

Layouts: train_0375;train_0861;train_0562;train_1358;train_0296;train_0097;train_1925;train_0714;train_1421;train_0091;train_1746;train_0774;train_1182;train_0443;train_0064;train_0310;train_0855;train_1150;train_1805;train_0708

### near_cutoff

![Recorded round ledger](../cases/v3_rnn_single_influence_seed2_near_cutoff.png)

Layouts: train_0745;train_1406;train_0616;train_1987;train_1327;train_1117;train_1865;train_0149;train_0921;train_0441;train_1468;train_1455;train_0075;train_1514;train_1246;train_0139;train_0974;train_0970;train_1670;train_0362

### most_switches

![Recorded round ledger](../cases/v3_rnn_single_influence_seed2_most_switches.png)

Layouts: train_1347;train_1414;train_0362;train_1235;train_1843;train_1944;train_0792;train_1377;train_1973;train_0287;train_0278;train_1625;train_1954;train_1676;train_0139;train_0202;train_0867;train_1506;train_0455;train_1744

### largest_late_gain

![Recorded round ledger](../cases/v3_rnn_single_influence_seed2_largest_late_gain.png)

Layouts: train_1386;train_0972;train_1016;train_1427;train_1535;train_0153;train_1383;train_1649;train_1897;train_1502;train_0711;train_1363;train_0538;train_1416;train_1067;train_1715;train_0945;train_0047;train_0339;train_0271

