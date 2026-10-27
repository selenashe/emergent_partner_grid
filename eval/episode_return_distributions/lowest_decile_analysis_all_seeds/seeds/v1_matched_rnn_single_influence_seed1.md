# v1_matched_rnn_single_influence_seed1

Direct input + t0 fix; step penalty 0.01. Final-update training return 17.228.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed1_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed1_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v1_matched_influence_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6.7514 | 14.0294 | 0.0087 | -0.0472 | 0.0659 | approximately flat | 122 | 0.0000 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v1_matched_rnn_single_influence_seed1_worst_return.png)

Layouts: train_1386;train_0972;train_1016;train_1427;train_1535;train_0153;train_1383;train_1649;train_1897;train_1502;train_0711;train_1363;train_0538;train_1416;train_1067;train_1715;train_0945;train_0047;train_0339;train_0271

### near_cutoff

![Recorded round ledger](../cases/v1_matched_rnn_single_influence_seed1_near_cutoff.png)

Layouts: train_1743;train_1595;train_0689;train_0643;train_0064;train_0107;train_0347;train_1151;train_0991;train_1088;train_0397;train_1002;train_1247;train_0270;train_1154;train_0785;train_1975;train_1023;train_1849;train_1512

### most_switches

![Recorded round ledger](../cases/v1_matched_rnn_single_influence_seed1_most_switches.png)

Layouts: train_0339;train_0857;train_0556;train_0206;train_1789;train_0120;train_1768;train_0790;train_1627;train_0890;train_0292;train_1349;train_0587;train_0403;train_0327;train_0399;train_0331;train_1534;train_1890;train_0097

### largest_late_gain

![Recorded round ledger](../cases/v1_matched_rnn_single_influence_seed1_largest_late_gain.png)

Layouts: train_0375;train_0861;train_0562;train_1358;train_0296;train_0097;train_1925;train_0714;train_1421;train_0091;train_1746;train_0774;train_1182;train_0443;train_0064;train_0310;train_0855;train_1150;train_1805;train_0708

