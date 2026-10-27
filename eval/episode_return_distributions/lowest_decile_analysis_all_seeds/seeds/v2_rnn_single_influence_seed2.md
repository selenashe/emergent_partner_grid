# v2_rnn_single_influence_seed2

Direct input + t0 fix; step penalty 0.01. Final-update training return 17.284.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed2_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed2_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v2_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1.2971 | 13.1739 | -0.5618 | -0.6357 | -0.4880 | decreasing | 425 | 4.7120 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v2_rnn_single_influence_seed2_worst_return.png)

Layouts: train_0567;train_1356;train_1315;train_0495;train_1615;train_0437;train_1172;train_1428;train_1457;train_1168;train_1490;train_0860;train_1304;train_1244;train_1900;train_0624;train_1791;train_0619;train_1625;train_0989

### near_cutoff

![Recorded round ledger](../cases/v2_rnn_single_influence_seed2_near_cutoff.png)

Layouts: train_0187;train_1211;train_0403;train_0311;train_1351;train_1132;train_1180;train_0465;train_0106;train_1013;train_0276;train_0430;train_1211;train_1743;train_0430;train_0402;train_0147;train_1191;train_1165;train_1916

### most_switches

![Recorded round ledger](../cases/v2_rnn_single_influence_seed2_most_switches.png)

Layouts: train_0540;train_1023;train_1551;train_1358;train_1146;train_1170;train_1821;train_1671;train_1406;train_0244;train_0217;train_0587;train_1310;train_1598;train_0364;train_0849;train_0282;train_1172;train_1931;train_1730

### largest_late_gain

![Recorded round ledger](../cases/v2_rnn_single_influence_seed2_largest_late_gain.png)

Layouts: train_1655;train_1371;train_1140;train_0505;train_1340;train_1244;train_1040;train_0226;train_0995;train_1332;train_1765;train_1431;train_0490;train_1353;train_1615;train_0583;train_1165;train_0065;train_1658;train_0537

