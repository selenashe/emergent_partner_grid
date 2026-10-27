# v2_rnn_diverse_influence_seed5

Direct input + t0 fix; step penalty 0.01. Final-update training return 13.072.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed5_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed5_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v2_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 8.2492 | 14.2711 | -0.0587 | -0.1335 | 0.0146 | no clear trend | 173 | 1.1446 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed5_worst_return.png)

Layouts: train_1094;train_1029;train_0282;train_0000;train_1428;train_0244;train_1741;train_1525;train_0873;train_1191;train_1380;train_1340;train_0030;train_1122;train_1279;train_1764;train_1081;train_0691;train_1103;train_0660

### near_cutoff

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed5_near_cutoff.png)

Layouts: train_0521;train_0138;train_0330;train_1246;train_1498;train_1306;train_1285;train_0879;train_0278;train_0637;train_0461;train_1766;train_1354;train_0645;train_0287;train_0119;train_0352;train_1018;train_0537;train_1289

### most_switches

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed5_most_switches.png)

Layouts: train_0466;train_0340;train_0463;train_1283;train_1808;train_1406;train_0485;train_1682;train_1483;train_0854;train_1018;train_1316;train_1166;train_1392;train_0108;train_0867;train_0890;train_0609;train_1445;train_1757

### largest_late_gain

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed5_largest_late_gain.png)

Layouts: train_1913;train_1781;train_1611;train_0626;train_0453;train_1491;train_0867;train_0751;train_0545;train_1058;train_0328;train_1944;train_0932;train_0638;train_0695;train_0269;train_1332;train_0426;train_0715;train_1864

