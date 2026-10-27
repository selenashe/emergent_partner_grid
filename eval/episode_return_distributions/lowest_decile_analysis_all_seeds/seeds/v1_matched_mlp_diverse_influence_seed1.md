# v1_matched_mlp_diverse_influence_seed1

Direct input + t0 fix; step penalty 0.01. Final-update training return 13.431.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed1_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed1_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v1_matched_influence_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 9.4153 | 13.7468 | -0.0203 | -0.0754 | 0.0351 | approximately flat | 95 | 0.0000 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v1_matched_mlp_diverse_influence_seed1_worst_return.png)

Layouts: train_1463;train_1637;train_1409;train_0343;train_1620;train_1349;train_1115;train_1746;train_0209;train_0981;train_0271;train_1860;train_0651;train_0153;train_1225;train_1872;train_0544;train_1671;train_0794;train_0002

### near_cutoff

![Recorded round ledger](../cases/v1_matched_mlp_diverse_influence_seed1_near_cutoff.png)

Layouts: train_1183;train_1913;train_1611;train_1396;train_0908;train_1429;train_1881;train_0770;train_0495;train_0175;train_0516;train_0697;train_1275;train_1344;train_1581;train_0102;train_0402;train_1212;train_1204;train_0076

### most_switches

![Recorded round ledger](../cases/v1_matched_mlp_diverse_influence_seed1_most_switches.png)

Layouts: train_1580;train_0508;train_0010;train_0147;train_0819;train_1363;train_0991;train_1735;train_0304;train_0875;train_0137;train_0045;train_0563;train_0921;train_1799;train_1231;train_0065;train_0609;train_1380;train_1396

### largest_late_gain

![Recorded round ledger](../cases/v1_matched_mlp_diverse_influence_seed1_largest_late_gain.png)

Layouts: train_1445;train_0314;train_0210;train_0569;train_1356;train_1622;train_0791;train_0932;train_1905;train_0991;train_1492;train_1120;train_0594;train_0908;train_0319;train_1605;train_1005;train_0540;train_0430;train_1143

