# v2_rnn_single_influence_seed1

Direct input + t0 fix; step penalty 0.01. Final-update training return 17.226.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed1_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_single_influence_seed1_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v2_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.1226 | 13.2644 | -0.4308 | -0.5046 | -0.3592 | decreasing | 371 | 3.9027 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v2_rnn_single_influence_seed1_worst_return.png)

Layouts: train_1412;train_1814;train_1359;train_1285;train_1808;train_1911;train_0574;train_1168;train_0542;train_0363;train_0843;train_0020;train_0938;train_0000;train_1437;train_0801;train_0422;train_0375;train_1952;train_0427

### near_cutoff

![Recorded round ledger](../cases/v2_rnn_single_influence_seed1_near_cutoff.png)

Layouts: train_0563;train_1737;train_0916;train_1597;train_1821;train_1967;train_1499;train_0576;train_0169;train_1673;train_1554;train_1315;train_0246;train_1160;train_0774;train_1070;train_1556;train_0849;train_0460;train_1907

### most_switches

![Recorded round ledger](../cases/v2_rnn_single_influence_seed1_most_switches.png)

Layouts: train_1349;train_0303;train_0176;train_1054;train_1268;train_0263;train_1372;train_1464;train_1518;train_0102;train_1432;train_1348;train_1509;train_0558;train_1534;train_1959;train_0611;train_0587;train_0889;train_0506

### largest_late_gain

![Recorded round ledger](../cases/v2_rnn_single_influence_seed1_largest_late_gain.png)

Layouts: train_0882;train_0857;train_1411;train_0589;train_0158;train_1003;train_0907;train_0527;train_1016;train_0777;train_0726;train_0172;train_0907;train_1432;train_0752;train_1445;train_1950;train_1091;train_0468;train_1952

