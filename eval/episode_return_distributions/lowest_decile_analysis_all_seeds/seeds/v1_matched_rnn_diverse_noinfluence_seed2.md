# v1_matched_rnn_diverse_noinfluence_seed2

Direct input + t0 fix; step penalty 0.01. Final-update training return 11.617.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_noinfluence_seed2_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_noinfluence_seed2_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v1_matched_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 8.2643 | 12.4259 | 0.0302 | -0.0411 | 0.1012 | approximately flat | 208 | 0.0000 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v1_matched_rnn_diverse_noinfluence_seed2_worst_return.png)

Layouts: train_1032;train_0393;train_0801;train_0325;train_0331;train_0558;train_1320;train_0530;train_0569;train_1271;train_0624;train_1437;train_1840;train_1810;train_1670;train_0616;train_1186;train_1265;train_0109;train_1673

### near_cutoff

![Recorded round ledger](../cases/v1_matched_rnn_diverse_noinfluence_seed2_near_cutoff.png)

Layouts: train_1079;train_0421;train_1741;train_0289;train_1900;train_1748;train_1800;train_1554;train_1083;train_1427;train_0927;train_0075;train_1491;train_0466;train_0954;train_1535;train_1720;train_0546;train_1363;train_1771

### most_switches

![Recorded round ledger](../cases/v1_matched_rnn_diverse_noinfluence_seed2_most_switches.png)

Layouts: train_0446;train_1960;train_0352;train_1663;train_0645;train_1112;train_1304;train_1505;train_1791;train_0884;train_0024;train_0314;train_0677;train_1427;train_0022;train_1836;train_1170;train_1781;train_1422;train_0120

### largest_late_gain

![Recorded round ledger](../cases/v1_matched_rnn_diverse_noinfluence_seed2_largest_late_gain.png)

Layouts: train_0153;train_1737;train_0637;train_1015;train_0621;train_1031;train_1231;train_0938;train_0315;train_1988;train_1445;train_0244;train_0559;train_0263;train_1897;train_1508;train_0730;train_0414;train_1447;train_1476

