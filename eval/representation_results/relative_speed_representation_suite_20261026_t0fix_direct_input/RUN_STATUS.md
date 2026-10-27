# Completed

All 8 representations and 30 frozen policies per representation completed: 240 policy/representation runs, 96,240 independent classifiers over steps 0–400. All 720 individual-profile plots, 56 early-behavior/motion/correlation figures, and 4 comparison figures (PNG/PDF) were generated. Final audits passed; the original manifest and 30 archived baseline classifiers remain unchanged. See README.md and completion_audit.json.

The verification gate and finalizer were executed within the current CPU allocation; their pending scheduled jobs were cancelled to avoid duplicate work. Recovery and fit arrays completed; no suite jobs remain active.

The following is the historical orchestration log; its progress snapshots are superseded by this completion status.

# Representation probe suite — running

Original baseline: /juice6/u/jshe/emergent_partner_grid/eval/representation_results/relative_speed_train_profiles_to_all90_v1_matched_v2_v3_20261022_t0fix_direct_input

Eight result directories: current_observation, previous_memory, updated_memory, actor_features, critic_features, action_logits, action_probabilities, value_estimate

Fit:24 familiar profiles x16 episodes. Test:all90 profiles x4 episodes, repetitions1,9,14,15. Separate classifiers at401 timesteps for30 policies. No RL policy training.

Recovery uses original stored train/test capability-pool order and original additional-profile order, batch sizes and RNG seed12345. Each cache must pass archived decoder-decision, probability, terminal, and surviving original behavior checks. Original train/test hidden states are preserved exactly.

Active recovery array: 18112186

Fit arrays (aftercorr on each recovered policy):
- fit_current_observation: 18112258
- fit_previous_memory: 18112259
- fit_updated_memory: 18112260
- fit_actor_features: 18112261
- fit_critic_features: 18112262
- fit_action_logits: 18112263
- fit_action_probabilities: 18112264
- fit_value_estimate: 18112265

Aggregate array: 18112266
Finalize job: 18112267

The first policy is also being fitted using up to four processes in the current CPU allocation. Its queued array tasks are held until the local job completes, then released by the local orchestrator. Inspect local_first_policy_fit_results.json and logs/local_*_0.err if interrupted.

Focused checks:15 passed. A baseline-backed full all90 plotting smoke test generated90 individual profile plots and combined/category outputs under /tmp/representation_suite_primary_smoke_20261026. A real-data training smoke test reproduced baseline decisions at0,20,100.

The original test HDF5 pool order differs from the review membership list. First recovery attempt18111989 was cancelled after this order mismatch was caught; the successful train cache was retained. Frozen corrected recovery source is recovery_source_v2/.

Updated-memory retraining is recorded separately from the original archived fit. The first policy matches exact original training states and has3 changed held-out decisions out of144360, all atstep5; current and frozen reference optimizers differ only in accepting arbitrary input width. The per-policy completion files record baseline regressions.

TIMING: all plots use the current observation index t. Previous-memory features are h_(t-1), so they exclude observation t; behavior curves still describe events before observation t. Other representations are evaluated from observation t or its updated state. Each representation holds its own final valid feature vector after episode termination.

Shared policy action/reward plots link to the unchanged baseline action_probabilities_rewards_first50_20261023 directory. Early behavior/motion/correlation outputs are recomputed for each representation.

Monitor: squeue -u jshe; logs/; */policies/*/*/completion.json. The final completion_audit.json is written only after all eight result directories and comparison figures complete.


## Latest orchestration

GPU batch job: 18112596. It runs the unchanged frozen recovery_source_v2 producer in fresh per-policy processes, then clears only that validated policy's pending fit dependencies. Original per-policy GPU array18112186 is retired.

Final verification gate: 18112794; it depends on all eight fit arrays. It verifies240 fitted hashes, original baseline hashes, and all archived validated motion grids; applies caption-only global-time clarifications; then releases held aggregation18112266. Finalize18112267 depends on that aggregation. No manual release is required if the gate passes.

Last snapshot: 18/30 policies recovered; 56/240 fits completed. All15 focused tests passed;536 frozen source hashes matched;55,335 v1 motion observations matched exactly.
