# Training-capability probes evaluated on all 90 profiles

Read existing post-training evaluation rollouts of the final frozen direct-input, fixed-t0 v1_matched/v2/v3 policies, seeds 1–5. No RL policy retraining and no new rollouts. These are final-policy states on familiar capabilities, not historical states recorded during RL optimization.

Fit a separate affine 128->2 faster-goal classifier at every environment step t=0..400. The input is h_t after the current observation and before its action, with no explicit time averaging. The label is red-faster (0) or blue-faster (1), determined by comparing the two delays. Equal-delay profiles are excluded.

Only the 24 diverse-policy training capability pairs enter probe fitting: 16 episodes per profile, 384 examples, with 192 per class. The frozen reference optimizer uses full-batch cross-entropy, Adam1000 updates, lr=.01, no weight decay, init10000, and raw float32 features. Test every one of the 90 unequal pairs on four separate episodes: repetition IDs 1,9,14,15, zero-based. The remaining 1056 examples (16 for each of the 66 novel profiles) never enter probe fitting or reported testing.

The same held-out episode IDs are used at every timestep, as in the earlier all90-fit analysis. Profile accuracy is correct predictions / 4 for each policy seed, then mean across five seeds. Within one profile this equals recall for its fixed faster-goal class. Shading is the 95% percentile bootstrap over policy seeds (10000 draws, RNG30000). Category means first average profiles within each seed.

At timesteps past termination, carry the final valid hidden state; never use padding. Active test counts are retained. Colors label diverse-policy training exposure.

Single-partner RNN is omitted: its own policy-training support is only (1,4), supplying red-faster labels only. No degenerate binary accuracy curve is fabricated.

Fit profiles [red delay, blue delay]:

(1,4), (1,7), (1,8), (1,9), (2,4), (2,7), (2,8), (2,9), (3,4), (3,7), (3,8), (3,9), (4,1), (4,2), (4,3), (7,1), (7,2), (7,3), (8,1), (8,2), (8,3), (9,1), (9,2), (9,3)

Outputs: all90 combined PNG/PDF; 90 individual profile PNGs and multipage PDF; category averages over steps 0..400 and the first60; summary CSV/NPZ; per-policy fitted readouts, probabilities, partitions, and episode manifests; a final integrity and no-leakage audit.

Final audit passed: all reported metrics were recomputed from saved predictions and the fit/test capability and episode masks were independently verified.
