# Updated memory h_t faster-goal decoding

Feature dimension: 128; one vector per episode at each observation index t=0..400. The frozen policy and all actor/critic transformations remain unchanged. Only the affine two-class readout is fitted.

Fit 24 actual policy-training profiles x16 held-in evaluation episodes =384 examples, 192 per class. Test all 90 unequal profiles x4 held-out episodes =360 examples; repetition IDs 1,9,14,15. The other 1056 examples never enter fitting or reported testing. No temporal averaging or feature normalization. Independent full-batch class-balanced Adam1000, lr .01, no weight decay, initialization 10000 with scale 1/sqrt(D).

Five policy seeds for each of v1_matched, v2 and v3; influence and no-influence conditions reported separately. Single-partner control remains omitted because its policy-training support has one faster-goal class. Profile accuracy uses four episodes; category curves give profiles equal weight within seed. Intervals bootstrap five policy seeds, 10000 resamples, RNG30000.

Trajectories reconstructed from original batch sizes, profile order, reset/action RNG seeds and frozen checkpoints. Reconstruction audits compare surviving original behavior, archived decoding outputs, and episode terminals. Inputs are shared with the other representations and archived separately. No replacement episode IDs.

The action_probabilities_rewards_first50_20261023 directory links to the unchanged frozen-policy action/reward analysis; those quantities depend on the policy and episode cohort and are identical for all probe representations.

Temporal boundary rule: Hold each representation's final valid vector forward. Previous memory at terminal observation L-1 is h_(L-2).

Outputs mirror the baseline per-profile/category results and include representation-specific early-behavior diagnostics.
