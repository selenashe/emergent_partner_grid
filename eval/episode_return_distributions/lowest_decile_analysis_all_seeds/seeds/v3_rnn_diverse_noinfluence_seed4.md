# v3_rnn_diverse_noinfluence_seed4

Direct input + t0 fix; step penalty 0.01. Final-update training return 11.400.

[Checkpoint](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_noinfluence_seed4_t0fix_direct_input.safetensors) · [Saved config](/juice6/u/jshe/emergent_partner_grid/train/train_logs/v3_t0fix_direct_input/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/rnn_diverse_noinfluence_seed4_t0fix_direct_input_config.json) · [Matching source](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/v3_t0fix_direct_input/source)

| seed | tail_mean_return | rest_mean_return | late_minus_early | delta_ci_low | delta_ci_high | trend | tail_timeout_rounds | tail_switches_per_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 7.2968 | 11.9016 | -0.0250 | -0.1028 | 0.0518 | approximately flat | 260 | 0.0000 |

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

## Four distinct recorded tail cases

### worst_return

![Recorded round ledger](../cases/v3_rnn_diverse_noinfluence_seed4_worst_return.png)

Layouts: train_0064;train_1315;train_0366;train_0952;train_1403;train_0216;train_0260;train_1662;train_1952;train_1478;train_1624;train_1047;train_1416;train_0164;train_1720;train_1512;train_0341;train_0341;train_1083;train_1781

### near_cutoff

![Recorded round ledger](../cases/v3_rnn_diverse_noinfluence_seed4_near_cutoff.png)

Layouts: train_1011;train_0675;train_1429;train_1621;train_0677;train_0330;train_0417;train_0151;train_0486;train_0680;train_0276;train_0692;train_0130;train_1936;train_0107;train_1890;train_1622;train_1194;train_0436;train_1033

### most_switches

![Recorded round ledger](../cases/v3_rnn_diverse_noinfluence_seed4_most_switches.png)

Layouts: train_0783;train_0708;train_0437;train_1213;train_0275;train_1749;train_1430;train_1567;train_0576;train_1658;train_1204;train_0579;train_1235;train_0943;train_1410;train_1737;train_0966;train_0974;train_0327;train_1074

### largest_late_gain

![Recorded round ledger](../cases/v3_rnn_diverse_noinfluence_seed4_largest_late_gain.png)

Layouts: train_0153;train_1737;train_0637;train_1015;train_0621;train_1031;train_1231;train_0938;train_0315;train_1988;train_1445;train_0244;train_0559;train_0263;train_1897;train_1508;train_0730;train_0414;train_1447;train_1476

