# v2_rnn_diverse_influence_seed1

Direct input + t0 fix; step penalty 0.01. Final-update training return 13.625.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed1_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed1_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v2_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 9.4299 | 14.2918 | 0.0134 | -0.0525 | 0.0812 | approximately flat | 136 | 0.8668 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed1_worst_return.png)

Layouts: train_1808;train_1122;train_0137;train_0692;train_1068;train_1237;train_1307;train_1663;train_1285;train_1878;train_0872;train_0397;train_0864;train_0183;train_0849;train_0640;train_1353;train_1307;train_0544;train_1997

### near_cutoff

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed1_near_cutoff.png)

Layouts: train_1746;train_1067;train_0310;train_0412;train_1518;train_0332;train_1717;train_0687;train_0031;train_0164;train_1884;train_1409;train_0970;train_0930;train_0951;train_0384;train_1445;train_0377;train_1003;train_1862

### most_switches

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed1_most_switches.png)

Layouts: train_0946;train_0165;train_0109;train_1967;train_1416;train_0024;train_0138;train_0824;train_1457;train_1851;train_1086;train_0427;train_1397;train_1641;train_0587;train_0925;train_1923;train_1412;train_1514;train_1324

### largest_late_gain

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed1_largest_late_gain.png)

Layouts: train_1001;train_0344;train_1062;train_1974;train_1041;train_1233;train_0220;train_0109;train_0214;train_0138;train_1262;train_1406;train_0960;train_1859;train_0289;train_1872;train_1253;train_0375;train_0701;train_0390

