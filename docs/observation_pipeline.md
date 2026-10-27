# Phase-aware direct-input experiments

The maintained workflow explicitly uses `GRID_ENCODER: flat` and
`INCLUDE_IS_T0: true` for both actor and critic in every RNN and MLP condition.
The grid contributes 245 binary values, allocation contributes three values,
and the round-local phase flag contributes one value. All 249 values enter the
GRU directly; memory is 128-dimensional. No convolution, projection or input
normalization precedes the GRU. The MLP control uses the same observations.

The phase bit is separate from recurrent episode-reset flags. Memory persists
through 20 rounds with one partner. The fix preserves each protocol's actor
masks, initialization, switching/cooldown rules, reward and recurrent PPO replay.

`train/observation_pipeline.py` retains encoder installation and short-prefix
support for validation. The maintained launcher prepares from already-verified
five-seed source variants/configs through `train/prepare_experiment.py`; it no
longer reconstructs models from original CNN experiments or runs encoder pilots.

All three protocols cross four conditions and learner seeds 1–5: 60 policies.
The scientific budget is 915 updates, 59,965,440 actual transitions, 20 rounds per
episode, horizon 100, step penalty 0.01 and success reward 1. CPU/GPU smoke tests
retain the original optimizer schedule and stop with `PILOT_UPDATES=1`.

Training/evaluation uses `bash/submit_t0fix_direct_input_training.py`; independent
behavior and representation stages use `bash/run_t0fix_direct_input_analyses.py`.
Source/config identity and shared-input hashes are sealed independently in
`train/config/direct_input_integrity.json`, after the original equivalence checks
passed. No retired CNN checkpoints, rollouts, configs or source ancestors are
required. The root README documents commands, storage and retained analyses.
