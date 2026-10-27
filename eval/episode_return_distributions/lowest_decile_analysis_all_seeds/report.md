# Lowest-return decile across all five learner seeds

Direct input + t0 fix · step penalty 0.01 · 60 policies · 55,200 episodes · 5,520 tail episodes · 110,400 tail rounds.

All 60 learners are included. Summary means weight the five independent learners equally, rather than treating their episodes as independent training replicates.

Each policy contributes exactly its lowest 92 of 920 episodes, sorted by return, population and episode index. Boundary ties and tie-inclusive changes are recorded. Trend intervals use 4,000 paired whole-episode bootstrap draws within each policy and condition on tail selection; they do not include seed-selection uncertainty.

All measurements use the original saved 480 familiar and 440 novel episodes per policy. Terminal transitions are included and scan padding is excluded. The reward is R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. An assignment labels the ego goal; the partner takes the other goal. Capability delay d allows a movement opportunity every d+1 ticks. Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. A trend inside an outcome-selected tail is not evidence of learning or memory use. The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.

Geometry-aware oracle comparisons use the matching 1,096-layout corpus and recorded initial assignment. They grant hidden capability information and informed navigation/switching; they are diagnostic ceilings, not an attainable policy baseline. Assignment opportunity gaps in no-influence controls are imposed by the environment, not ego decision errors.

## Current diagnoses

| version | condition | seed | training_return | tail_mean_return | rest_mean_return | late_minus_early | trend | tail_timeout_rounds | tail_switches_per_round | tail_final_partner_faster |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1_matched | rnn_diverse_influence | 1 | 12.3060 | 6.9608 | 14.0128 | 0.0015 | approximately flat | 111 | 0.0000 | 0.0000 |
| v1_matched | rnn_diverse_influence | 2 | 12.5650 | 6.8846 | 13.9725 | 0.0053 | approximately flat | 114 | 0.0000 | 0.0000 |
| v1_matched | rnn_diverse_influence | 3 | 12.8310 | 6.8496 | 14.0423 | -0.0130 | approximately flat | 120 | 0.0000 | 0.0109 |
| v1_matched | rnn_diverse_influence | 4 | 13.1720 | 6.9825 | 13.9789 | 0.0157 | approximately flat | 104 | 0.0000 | 0.0000 |
| v1_matched | rnn_diverse_influence | 5 | 14.0140 | 11.2135 | 15.0884 | 0.0103 | approximately flat | 154 | 0.0000 | 0.7364 |
| v1_matched | mlp_diverse_influence | 1 | 13.4310 | 9.4153 | 13.7468 | -0.0203 | approximately flat | 95 | 0.0000 | 0.2897 |
| v1_matched | mlp_diverse_influence | 2 | 13.4150 | 6.8795 | 13.9453 | -0.0178 | approximately flat | 112 | 0.0000 | 0.0000 |
| v1_matched | mlp_diverse_influence | 3 | 12.9740 | 10.2237 | 13.7555 | -0.0365 | approximately flat | 93 | 0.0000 | 0.4027 |
| v1_matched | mlp_diverse_influence | 4 | 12.9420 | 6.9478 | 13.9805 | 0.0003 | approximately flat | 114 | 0.0000 | 0.0005 |
| v1_matched | mlp_diverse_influence | 5 | 13.2420 | 9.8771 | 13.7762 | -0.0230 | approximately flat | 84 | 0.0000 | 0.3261 |
| v1_matched | rnn_single_influence | 1 | 17.2280 | 6.7514 | 14.0294 | 0.0087 | approximately flat | 122 | 0.0000 | 0.0000 |
| v1_matched | rnn_single_influence | 2 | 17.2830 | 6.9329 | 14.1096 | -0.0334 | approximately flat | 112 | 0.0000 | 0.0000 |
| v1_matched | rnn_single_influence | 3 | 17.3290 | 6.9354 | 14.1751 | -0.0490 | approximately flat | 110 | 0.0000 | 0.0005 |
| v1_matched | rnn_single_influence | 4 | 17.3150 | 6.9197 | 14.1725 | -0.0071 | approximately flat | 115 | 0.0000 | 0.0000 |
| v1_matched | rnn_single_influence | 5 | 17.3330 | 6.6446 | 14.1280 | -0.1143 | decreasing | 127 | 0.0000 | 0.0000 |
| v1_matched | rnn_diverse_noinfluence | 1 | 11.4870 | 7.9314 | 12.4221 | 0.0080 | approximately flat | 217 | 0.0000 | 0.4375 |
| v1_matched | rnn_diverse_noinfluence | 2 | 11.6170 | 8.2643 | 12.4259 | 0.0302 | approximately flat | 208 | 0.0000 | 0.4457 |
| v1_matched | rnn_diverse_noinfluence | 3 | 11.6270 | 8.0095 | 12.3417 | 0.0522 | no clear trend | 231 | 0.0000 | 0.4451 |
| v1_matched | rnn_diverse_noinfluence | 4 | 11.3060 | 7.9795 | 12.3784 | -0.0461 | approximately flat | 229 | 0.0000 | 0.4424 |
| v1_matched | rnn_diverse_noinfluence | 5 | 11.0450 | 7.8761 | 12.3368 | -0.0349 | approximately flat | 225 | 0.0000 | 0.4321 |
| v2 | rnn_diverse_influence | 1 | 13.6250 | 9.4299 | 14.2918 | 0.0134 | approximately flat | 136 | 0.8668 | 0.3859 |
| v2 | rnn_diverse_influence | 2 | 14.2380 | 11.5477 | 14.9413 | -0.0217 | approximately flat | 104 | 0.7875 | 0.6951 |
| v2 | rnn_diverse_influence | 3 | 13.8660 | 10.5783 | 14.5283 | 0.0011 | approximately flat | 155 | 0.9234 | 0.6918 |
| v2 | rnn_diverse_influence | 4 | 13.7260 | 8.6296 | 14.2825 | 0.0009 | approximately flat | 152 | 1.0614 | 0.3076 |
| v2 | rnn_diverse_influence | 5 | 13.0720 | 8.2492 | 14.2711 | -0.0587 | no clear trend | 173 | 1.1446 | 0.2293 |
| v2 | mlp_diverse_influence | 1 | 12.6090 | 5.6757 | 13.5942 | -0.0132 | approximately flat | 147 | 0.5310 | 0.0000 |
| v2 | mlp_diverse_influence | 2 | 11.9440 | 5.6924 | 13.6020 | -0.0116 | approximately flat | 146 | 0.5310 | 0.0000 |
| v2 | mlp_diverse_influence | 3 | 12.3930 | 7.1084 | 13.8380 | -0.0084 | approximately flat | 97 | 0.5185 | 0.0087 |
| v2 | mlp_diverse_influence | 4 | 13.2870 | 9.1499 | 13.7955 | 0.0344 | approximately flat | 67 | 0.5326 | 0.2092 |
| v2 | mlp_diverse_influence | 5 | 12.1230 | 6.9229 | 13.8753 | -0.0271 | approximately flat | 104 | 0.5527 | 0.0103 |
| v2 | rnn_single_influence | 1 | 17.2260 | 2.1226 | 13.2644 | -0.4308 | decreasing | 371 | 3.9027 | 0.0196 |
| v2 | rnn_single_influence | 2 | 17.2840 | 1.2971 | 13.1739 | -0.5618 | decreasing | 425 | 4.7120 | 0.0321 |
| v2 | rnn_single_influence | 3 | 17.3030 | 4.0474 | 13.6055 | -0.3707 | decreasing | 235 | 2.3250 | 0.0125 |
| v2 | rnn_single_influence | 4 | 17.3140 | 4.3511 | 13.7139 | -0.2615 | decreasing | 222 | 1.7549 | 0.0054 |
| v2 | rnn_single_influence | 5 | 17.2900 | 3.8757 | 13.4849 | -0.2188 | decreasing | 268 | 2.6473 | 0.0136 |
| v2 | rnn_diverse_noinfluence | 1 | 10.8600 | 7.3473 | 11.9527 | 0.0298 | approximately flat | 277 | 0.0000 | 0.4446 |
| v2 | rnn_diverse_noinfluence | 2 | 11.5860 | 7.5160 | 12.0741 | -0.0083 | approximately flat | 255 | 0.0000 | 0.4429 |
| v2 | rnn_diverse_noinfluence | 3 | 11.2110 | 7.3553 | 12.0898 | 0.0051 | approximately flat | 247 | 0.0000 | 0.4283 |
| v2 | rnn_diverse_noinfluence | 4 | 11.4000 | 7.2968 | 11.9016 | -0.0250 | approximately flat | 260 | 0.0000 | 0.4261 |
| v2 | rnn_diverse_noinfluence | 5 | 11.7740 | 7.7091 | 11.9624 | -0.0217 | approximately flat | 247 | 0.0000 | 0.4533 |
| v3 | rnn_diverse_influence | 1 | 12.9460 | 7.2087 | 13.9471 | 0.0625 | increasing | 110 | 0.3245 | 0.0293 |
| v3 | rnn_diverse_influence | 2 | 13.2410 | 8.3405 | 14.3195 | -0.0014 | approximately flat | 167 | 1.0304 | 0.3011 |
| v3 | rnn_diverse_influence | 3 | 14.8110 | 11.9923 | 15.4506 | -0.0477 | approximately flat | 79 | 0.2777 | 0.6821 |
| v3 | rnn_diverse_influence | 4 | 13.6910 | 7.2009 | 14.0339 | 0.0091 | approximately flat | 98 | 0.2255 | 0.0092 |
| v3 | rnn_diverse_influence | 5 | 13.2010 | 7.4807 | 14.0390 | 0.1060 | increasing | 103 | 0.4582 | 0.0261 |
| v3 | mlp_diverse_influence | 1 | 12.9060 | 7.0771 | 14.0109 | -0.0014 | approximately flat | 98 | 0.0103 | 0.0011 |
| v3 | mlp_diverse_influence | 2 | 13.1200 | 7.5888 | 14.1137 | -0.0093 | approximately flat | 81 | 0.0478 | 0.0266 |
| v3 | mlp_diverse_influence | 3 | 13.3580 | 7.1793 | 13.9961 | -0.0210 | approximately flat | 98 | 0.0201 | 0.0054 |
| v3 | mlp_diverse_influence | 4 | 13.5160 | 6.7589 | 13.8357 | -0.0111 | approximately flat | 122 | 0.0114 | 0.0000 |
| v3 | mlp_diverse_influence | 5 | 12.7150 | 7.2060 | 13.9665 | -0.0145 | approximately flat | 99 | 0.0473 | 0.0114 |
| v3 | rnn_single_influence | 1 | 17.4310 | 4.8663 | 13.9285 | -0.2911 | decreasing | 228 | 2.0658 | 0.0207 |
| v3 | rnn_single_influence | 2 | 17.4200 | 6.3438 | 14.1578 | -0.1773 | decreasing | 146 | 0.8168 | 0.0033 |
| v3 | rnn_single_influence | 3 | 17.3800 | 4.1935 | 13.8074 | -0.4867 | decreasing | 284 | 3.3103 | 0.0196 |
| v3 | rnn_single_influence | 4 | 17.3500 | 4.9672 | 13.9207 | -0.3967 | decreasing | 221 | 1.9625 | 0.0120 |
| v3 | rnn_single_influence | 5 | 17.4360 | 5.6642 | 14.1752 | -0.3143 | decreasing | 185 | 1.4348 | 0.0130 |
| v3 | rnn_diverse_noinfluence | 1 | 10.8600 | 7.3473 | 11.9527 | 0.0298 | approximately flat | 277 | 0.0000 | 0.4446 |
| v3 | rnn_diverse_noinfluence | 2 | 11.5860 | 7.5160 | 12.0741 | -0.0083 | approximately flat | 255 | 0.0000 | 0.4429 |
| v3 | rnn_diverse_noinfluence | 3 | 11.2110 | 7.3553 | 12.0898 | 0.0051 | approximately flat | 247 | 0.0000 | 0.4283 |
| v3 | rnn_diverse_noinfluence | 4 | 11.4000 | 7.2968 | 11.9016 | -0.0250 | approximately flat | 260 | 0.0000 | 0.4261 |
| v3 | rnn_diverse_noinfluence | 5 | 11.7740 | 7.7091 | 11.9624 | -0.0217 | approximately flat | 247 | 0.0000 | 0.4533 |

[Actor, critic and observation-channel audit](math_audit/report.md)

[Every seed and its four recorded cases](seed_index.md) · [240-page casebook](cases/all_seed_representative_cases.pdf)


![Round returns](plots/round_return.png)

![Round timeouts](plots/timeout_rate.png)

![Round durations](plots/round_steps.png)

![Effective switches](plots/switches_per_round.png)

![Final partner-faster assignment](plots/final_partner_faster.png)

![Every tail episode](plots/all_tail_episode_heatmaps.png)

## Profile composition and switching checks

The profile-standardized reference reweights all episodes to the tail’s capability mixture, including the tail itself. It is descriptive, not an independent control. Lost success credit plus extra tick cost exactly decomposes the rest-minus-tail return gap. High-switch versus zero-switch comparisons match profile, round and initial assignment, but still leave layout and time-at-risk confounding.

