# v1_matched_rnn_diverse_noinfluence_seed4

Direct input + t0 fix; step penalty 0.01. Final-update training return 11.306.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_noinfluence_seed4_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_noinfluence_seed4_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v1_matched_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 7.9795 | 12.3784 | -0.0461 | -0.1166 | 0.0232 | approximately flat | 229 | 0.0000 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v1_matched_rnn_diverse_noinfluence_seed4_worst_return.png)

Layouts: train_1936;train_1487;train_0790;train_1126;train_1406;train_0551;train_0968;train_0907;train_0935;train_0964;train_0485;train_1289;train_0637;train_0609;train_1401;train_0685;train_0685;train_1499;train_0888;train_0074

### near_cutoff

![Recorded round ledger](../cases/v1_matched_rnn_diverse_noinfluence_seed4_near_cutoff.png)

Layouts: train_1615;train_0191;train_0332;train_1525;train_1168;train_0731;train_0856;train_0022;train_0751;train_1332;train_1627;train_0138;train_1095;train_0612;train_1359;train_1332;train_0656;train_0376;train_0651;train_1685

### most_switches

![Recorded round ledger](../cases/v1_matched_rnn_diverse_noinfluence_seed4_most_switches.png)

Layouts: train_1307;train_1862;train_1712;train_1021;train_1514;train_1379;train_1011;train_0252;train_1771;train_0530;train_0694;train_1840;train_0457;train_0537;train_0932;train_1328;train_0900;train_1235;train_1114;train_1803

### largest_late_gain

![Recorded round ledger](../cases/v1_matched_rnn_diverse_noinfluence_seed4_largest_late_gain.png)

Layouts: train_1514;train_0109;train_0019;train_1392;train_0790;train_0264;train_1359;train_1005;train_1075;train_1577;train_0104;train_0037;train_0180;train_1532;train_0002;train_0151;train_1338;train_0739;train_0637;train_1530

