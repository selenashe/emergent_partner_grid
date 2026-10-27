# v1_matched_rnn_diverse_noinfluence_seed5

Direct input + t0 fix; step penalty 0.01. Final-update training return 11.045.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_noinfluence_seed5_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_noinfluence_seed5_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v1_matched_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 7.8761 | 12.3368 | -0.0349 | -0.1045 | 0.0404 | approximately flat | 225 | 0.0000 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v1_matched_rnn_diverse_noinfluence_seed5_worst_return.png)

Layouts: train_1145;train_0864;train_1665;train_1182;train_1700;train_1498;train_1005;train_0282;train_1244;train_0528;train_0030;train_1041;train_1520;train_0714;train_0970;train_1001;train_1380;train_1873;train_1316;train_1410

### near_cutoff

![Recorded round ledger](../cases/v1_matched_rnn_diverse_noinfluence_seed5_near_cutoff.png)

Layouts: train_1620;train_1700;train_0574;train_0079;train_0473;train_1573;train_0789;train_1894;train_1959;train_1794;train_1747;train_1853;train_0958;train_0953;train_0229;train_1081;train_1558;train_1095;train_1785;train_1476

### most_switches

![Recorded round ledger](../cases/v1_matched_rnn_diverse_noinfluence_seed5_most_switches.png)

Layouts: train_0551;train_1218;train_1803;train_0665;train_1033;train_1002;train_1931;train_0099;train_1168;train_1152;train_1907;train_1304;train_0187;train_0324;train_1844;train_1907;train_0176;train_0843;train_1734;train_0122

### largest_late_gain

![Recorded round ledger](../cases/v1_matched_rnn_diverse_noinfluence_seed5_largest_late_gain.png)

Layouts: train_0786;train_0366;train_0801;train_1445;train_0927;train_1380;train_1478;train_1605;train_0086;train_0408;train_1709;train_1353;train_0864;train_1504;train_0111;train_0082;train_1768;train_0987;train_0217;train_0377

