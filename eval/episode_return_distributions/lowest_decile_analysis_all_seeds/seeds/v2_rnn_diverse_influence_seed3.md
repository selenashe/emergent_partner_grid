# v2_rnn_diverse_influence_seed3

Direct input + t0 fix; step penalty 0.01. Final-update training return 13.866.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed3_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v2_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_influence_seed3_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v2_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 10.5783 | 14.5283 | 0.0011 | -0.0683 | 0.0686 | approximately flat | 155 | 0.9234 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed3_worst_return.png)

Layouts: train_0196;train_0714;train_0525;train_0524;train_0715;train_1180;train_0921;train_1472;train_0005;train_1351;train_1092;train_0903;train_0786;train_1137;train_0473;train_1242;train_1080;train_0492;train_1682;train_1033

### near_cutoff

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed3_near_cutoff.png)

Layouts: train_0908;train_0235;train_1268;train_0402;train_0300;train_1340;train_0680;train_0980;train_0705;train_0183;train_1907;train_0443;train_0296;train_0986;train_1752;train_1278;train_1019;train_0443;train_1190;train_0419

### most_switches

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed3_most_switches.png)

Layouts: train_1579;train_1151;train_0354;train_1556;train_0542;train_1029;train_0206;train_1888;train_1275;train_0187;train_0037;train_1964;train_1208;train_1030;train_0237;train_0319;train_1143;train_1712;train_1290;train_0713

### largest_late_gain

![Recorded round ledger](../cases/v2_rnn_diverse_influence_seed3_largest_late_gain.png)

Layouts: train_0577;train_0726;train_1509;train_1869;train_1780;train_0609;train_0887;train_1427;train_0634;train_0462;train_1472;train_0245;train_1271;train_0907;train_1750;train_0878;train_1225;train_1967;train_1570;train_0770

