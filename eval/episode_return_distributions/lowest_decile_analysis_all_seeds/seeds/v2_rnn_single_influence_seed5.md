# v2_rnn_single_influence_seed5

Direct input + t0 fix; step penalty 0.01. Final-update training return 17.290.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed5_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed5_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v2_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 3.8757 | 13.4849 | -0.2188 | -0.2797 | -0.1561 | decreasing | 268 | 2.6473 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v2_rnn_single_influence_seed5_worst_return.png)

Layouts: train_0640;train_1954;train_0889;train_0524;train_1812;train_0849;train_0989;train_0218;train_1160;train_1316;train_1814;train_0988;train_0626;train_0575;train_1005;train_1221;train_1166;train_1772;train_1228;train_0815

### near_cutoff

![Recorded round ledger](../cases/v2_rnn_single_influence_seed5_near_cutoff.png)

Layouts: train_0513;train_1986;train_0469;train_1556;train_0711;train_1780;train_1859;train_1404;train_0218;train_1479;train_1409;train_1860;train_0230;train_1095;train_0964;train_1817;train_1094;train_0911;train_0187;train_0852

### most_switches

![Recorded round ledger](../cases/v2_rnn_single_influence_seed5_most_switches.png)

Layouts: train_1435;train_0540;train_0815;train_0062;train_1462;train_1204;train_1765;train_0916;train_0201;train_0271;train_0000;train_1759;train_0288;train_1806;train_0315;train_0841;train_0216;train_1302;train_0397;train_1307

### largest_late_gain

![Recorded round ledger](../cases/v2_rnn_single_influence_seed5_largest_late_gain.png)

Layouts: train_0375;train_0861;train_0562;train_1358;train_0296;train_0097;train_1925;train_0714;train_1421;train_0091;train_1746;train_0774;train_1182;train_0443;train_0064;train_0310;train_0855;train_1150;train_1805;train_0708

