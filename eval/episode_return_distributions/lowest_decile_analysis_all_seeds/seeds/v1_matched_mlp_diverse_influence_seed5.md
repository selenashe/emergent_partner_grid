# v1_matched_mlp_diverse_influence_seed5

Direct input + t0 fix; step penalty 0.01. Final-update training return 13.242.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed5_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v1_matched_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/mlp_diverse_influence_seed5_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v1_matched_influence_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 9.8771 | 13.7762 | -0.0230 | -0.0780 | 0.0296 | approximately flat | 84 | 0.0000 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v1_matched_mlp_diverse_influence_seed5_worst_return.png)

Layouts: train_1514;train_1611;train_1887;train_0473;train_0784;train_1361;train_1649;train_1821;train_0492;train_1018;train_0379;train_1228;train_0689;train_1182;train_1080;train_1464;train_0390;train_1814;train_1228;train_1445

### near_cutoff

![Recorded round ledger](../cases/v1_matched_mlp_diverse_influence_seed5_near_cutoff.png)

Layouts: train_0485;train_0228;train_0609;train_1079;train_1806;train_0730;train_1891;train_1170;train_1865;train_1003;train_0972;train_0900;train_0951;train_1645;train_0803;train_0739;train_0010;train_0139;train_0828;train_1389

### most_switches

![Recorded round ledger](../cases/v1_matched_mlp_diverse_influence_seed5_most_switches.png)

Layouts: train_0534;train_1641;train_0701;train_0651;train_0111;train_0211;train_1117;train_0037;train_1265;train_0523;train_1372;train_1612;train_1891;train_0843;train_0701;train_0959;train_1673;train_0027;train_0689;train_0151

### largest_late_gain

![Recorded round ledger](../cases/v1_matched_mlp_diverse_influence_seed5_largest_late_gain.png)

Layouts: train_1445;train_0314;train_0210;train_0569;train_1356;train_1622;train_0791;train_0932;train_1905;train_0991;train_1492;train_1120;train_0594;train_0908;train_0319;train_1605;train_1005;train_0540;train_0430;train_1143

