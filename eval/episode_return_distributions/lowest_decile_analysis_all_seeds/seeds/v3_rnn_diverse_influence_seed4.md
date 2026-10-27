# v3_rnn_diverse_influence_seed4

Direct input + t0 fix; step penalty 0.01. Final-update training return 13.691.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed4_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed4_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v3_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 7.2009 | 14.0339 | 0.0091 | -0.0407 | 0.0573 | approximately flat | 98 | 0.2255 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v3_rnn_diverse_influence_seed4_worst_return.png)

Layouts: train_1396;train_0158;train_1227;train_1514;train_1227;train_0068;train_0487;train_0457;train_0852;train_1478;train_0594;train_1278;train_0045;train_1242;train_1247;train_1921;train_1431;train_0890;train_1227;train_1580

### near_cutoff

![Recorded round ledger](../cases/v3_rnn_diverse_influence_seed4_near_cutoff.png)

Layouts: train_1476;train_0540;train_1512;train_1479;train_1010;train_0349;train_0677;train_1013;train_0325;train_0068;train_0532;train_1387;train_0099;train_0986;train_0375;train_0506;train_0855;train_1682;train_0794;train_0343

### most_switches

![Recorded round ledger](../cases/v3_rnn_diverse_influence_seed4_most_switches.png)

Layouts: train_1952;train_1905;train_1611;train_0629;train_1354;train_0065;train_1592;train_1062;train_0640;train_0218;train_1421;train_1383;train_0804;train_1791;train_0528;train_1213;train_1191;train_1888;train_1828;train_1864

### largest_late_gain

![Recorded round ledger](../cases/v3_rnn_diverse_influence_seed4_largest_late_gain.png)

Layouts: train_0375;train_0861;train_0562;train_1358;train_0296;train_0097;train_1925;train_0714;train_1421;train_0091;train_1746;train_0774;train_1182;train_0443;train_0064;train_0310;train_0855;train_1150;train_1805;train_0708

