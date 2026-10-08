# CoordinationGrid implementation and experiment audit guide

Reviewed against the working tree on **2026-10-06**; obsolete experiment links removed on **2026-10-07**. This guide maps the methods in `project_log_revised.md` and subsequent active implementations to callable code. The function inventory below includes exact source links and line numbers. The initial review added explanatory comments without changing executable Python logic. A subsequent cleanup removed unused stratified layout selection; its behavior and validation are recorded below. Existing uncommitted implementation work is included in this review.

## Reading the experiment

The question is whether an ego agent learns about a partner's hidden red/blue movement delays through repeated interaction and uses that information to assign goals. One **tick** is one environment transition. One **round** starts with fresh positions/layout and ends in success or timeout. One **partner episode** contains 20 rounds with the same hidden capability. The RNN remembers previous rounds within that episode and resets for the next partner. A **learner seed** identifies an independently trained policy; an evaluation repetition is a different unit.

A capability `(d_R,d_B)` gives wait ticks between partner moves toward red and blue. Lower is faster; after a move the partner waits `d` ticks, giving period `d+1`. The ego has no cooldown. The partner follows a deterministic shortest path; only the ego is learned.

| Condition | Memory | Policy-training partners | Requests control partner goal? | Implemented at |
| --- | --- | --- | --- | --- |
| `rnn_diverse_influence` | 128-dimensional GRU | 24 profiles | Yes | [build_network](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:325), [resolve_training_capability_pool](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:355), [CoordinationGrid.step_env](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:812) |
| `mlp_diverse_influence` | Feedforward replacement | 24 profiles | Yes | [ActorCriticCommMLP.__call__](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:266); same pool/transition |
| `rnn_single_influence` | GRU | `(1,4)` by default | Yes | [resolve_training_capability_pool](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:355); same RNN/transition |
| `rnn_diverse_noinfluence` | GRU | 24 profiles | No | [make_train](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:411) routes `INFLUENCE` to [CoordinationGrid.step_env](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:812) |

The authoritative 24 familiar and 22 novel profiles are constants in [jaxmarl/environments/coordination_grid/capability_populations.py](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/capability_populations.py). Novel profiles include scalar delays 0, 5, and 6 absent from policy training. Policy evaluation reuses familiar layouts: this is partner generalization, not unseen-layout generalization. “Familiar” evaluation includes all 24 training-pool profiles even for the single-partner policy, which actually trained on only one.

## Terms used in the code

| Term | Plain-language meaning |
| --- | --- |
| BFS (breadth-first search) | Explore walkable cells from nearest to farthest to find shortest routes around walls. |
| CNN (convolutional neural network) | Learn visual patterns from the grid, applying the same pattern detector at different positions. |
| GRU / carry / hidden state | A learned memory vector, updated when the agent observes the grid; “carry” is the memory passed to the next tick. |
| Actor / critic | The actor assigns probabilities to legal actions; the critic estimates future reward to help training. |
| PPO | Collect actions using the current policy, then improve its weights while limiting changes in the probabilities of those collected actions. |
| GAE / advantage | Estimate how much better an action's observed outcome was than the old value prediction, combining immediate and later evidence. |
| Oracle / blind baseline | A reference that knows the capability versus a fixed assignment chosen without that capability; here both use static analytical formulas. |
| Probe / decoder | A separately fitted small predictor asking what capability information can be recovered from saved policy memory. It does not train the policy. |
| Discovery / validation / test | Fit parameters on discovery data, choose settings on validation data, then measure performance on reserved test data. The strict released probe deliberately uses a different selection procedure. |
| Mask | A yes/no array selecting legal actions or valid recorded ticks; excluded entries must not affect the calculation. |
| JIT / vmap / scan | JAX facilities for compiling calculations, running them across parallel simulations, and repeating a state update over time. |
| JSON / NPZ / HDF5 / safetensors | Formats for readable configuration, numerical arrays, large structured rollout arrays, and saved network weights, respectively. |
| Bootstrap interval | Repeatedly resample the stated independent units to describe uncertainty; episodes and independently trained policies are different units. |

## Versions, defaults, and frozen evidence

| Experiment branch | Assignment behavior | Corpus / exposure | Source of truth |
| --- | --- | --- | --- |
| Counterbalanced v1/v2 | Each retains its own transition rules and PPO implementation | Shared 1096 corpus, paired packets, global queue | `sbatch_counterbalanced1096_20261002_235609.json`, separate frozen trees |
| Preparation source template | `online_v2`, **categorical policy sampling** | 1096 corpus, **random** sampler in the root trainer | [train/config/ippo_coordination_grid.yaml](/juice6/u/jshe/emergent_partner_grid/train/config/ippo_coordination_grid.yaml), [make_train](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:411) |
| New corpus-preparation branch | No policy retraining implied | Uncapped exact 12/12 training-profile optimum selection when `--n_final` is omitted | [main](/juice6/u/jshe/emergent_partner_grid/data_prep/build_final_corpus.py:162); separate 2262-layout output exists |

The October 3 project log is historical evidence and does not describe every October 6 working-tree addition or subsequent changes to active defaults. Do not infer a completed run's settings from current YAML. Counterbalanced scheduling is installed by [patch_trainer](/juice6/u/jshe/emergent_partner_grid/bash/submit_counterbalanced_training.py) in a **new frozen copy**; merely running the active trainer does not enable it.

[resolve_path](/juice6/u/jshe/emergent_partner_grid/repo_paths.py:38) and [resolve_record_paths](/juice6/u/jshe/emergent_partner_grid/repo_paths.py:78) translate old `dev/`, `analysis/`, `logs/`, and archived paths while leaving original records intact. Preserved historical snapshots are deliberately not annotated: their hashes identify the exact source used for completed runs.

## Step-by-step implementation map

### 1. Generate, filter, balance, and inspect layouts

| Experimental step | Exact implementation | What to inspect / output |
| --- | --- | --- |
| Generate wall grids and distinct starts/goals | [generate_envs](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:397), [_sample_env](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:320), [_sample_grid](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:309) | Seeded proposals; rejection limits; reachability/geometric constraints happen before accepted candidates |
| Compute wall-aware paths and structural metrics | [bfs_distances](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:86), [bfs_path_counts](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:136), [canonical_shortest_path](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:166), [compute_metrics](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:243) | Distances, junctions, dead ends, assignment gap, path overlap, switching-cost geometry; switching-cost metric is a generation screen, not an online optimal controller |
| Deduplicate accepted candidates | [_env_fingerprint](/juice6/u/jshe/emergent_partner_grid/data_prep/build_final_corpus.py:108) called in [main](/juice6/u/jshe/emergent_partner_grid/data_prep/build_final_corpus.py:162) | Exact geometry fingerprint and `n_unique_candidates`; not a separate symmetry-equivalence deduplication |
| Score both fixed assignments across training capabilities | [completion_time](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:59), [_reward_from_time](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:91), [evaluate_layout](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:221) | Oracle versus best fixed allocation, red/blue optimum counts, ties, feasibility, worst completion |
| Observability screen (Filter C) | [passes_observability](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:376) in builder `main` | Partner distance to **both** goals must reach configured minimum |
| Broad allocation-dependence/balance screen (Filter A) | [passes_allocation_balance](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:394) in builder `main` | Non-tied red-optimum fraction within thresholds; bound on tied cases |
| Delta-reward screen (Filter B) | [passes_delta_reward](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:433) and builder `main` | Absolute threshold or quantile calculated on survivors of observability + broad balance |
| Reject long-horizon outliers | [worst_case_oracle_steps](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:443), [horizon_outlier_mask](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:448) | Threshold uses post-delta survivors; boundary/zero-variance handling matters |
| Recommend feasible horizon | [derive_max_steps](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:478) | Recommendation written to manifest; executed horizon remains explicit in run config |
| New uncapped exact-optimum selection | [passes_exact_allocation_balance](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:416) and builder `main` | Keep every final survivor with 12 strictly red-optimal and 12 strictly blue-optimal profiles, no ties |
| Historical/explicit capped selection | [main](/juice6/u/jshe/emergent_partner_grid/data_prep/build_final_corpus.py:162), centroid branch | `--n_final` requires `--centroid_p_opt_red_target`; retained 1000/2000 corpora used target 0.5. The unused stratified alternative has been removed. |
| Apply eight D4 rotations/reflections | [apply_sym_to_env](/juice6/u/jshe/emergent_partner_grid/data_prep/build_final_corpus.py:88), [_sym_position](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:389), [_sym_wall](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:413) | Transform walls, starts, and goals together; recompute/check selection invariance; symmetry counts differ by at most one for uncapped sizes |
| Shuffle, split, and write full selection evidence | Builder [main](/juice6/u/jshe/emergent_partner_grid/data_prep/build_final_corpus.py:162), [env_to_json_dict](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:483), [encode_env](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:436) | `manifest.json`, split JSON/NPZ, `diagnostics/candidates.csv`, stage CSVs and survivor indices; active experiments use train layouts |
| Inspect distance/wall-density distributions | [main](/juice6/u/jshe/emergent_partner_grid/data_prep/plot_layout_distributions.py:56) | Marginal and joint distributions and cutoff feasibility; descriptive diagnostics |
| Build simultaneous distance-balanced subset | [expected_counts](/juice6/u/jshe/emergent_partner_grid/data_prep/balance_ego_distances.py:26), [_fractional_cycle](/juice6/u/jshe/emergent_partner_grid/data_prep/balance_ego_distances.py:87), [round_counts](/juice6/u/jshe/emergent_partner_grid/data_prep/balance_ego_distances.py:127), [main](/juice6/u/jshe/emergent_partner_grid/data_prep/balance_ego_distances.py:160) | Both ego distances 1–8; 137 layouts per distance per goal; maximum-flow feasibility, fractional entropy solution, expectation-preserving integer rounding, uniform bin sampling without replacement |
| Render grids for human inspection | [render_one](/juice6/u/jshe/emergent_partner_grid/data_prep/render_layout_corpus.py:17), [contact_sheet](/juice6/u/jshe/emergent_partner_grid/data_prep/render_layout_corpus.py:41), [render_env](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:496) | Pictures and contact sheets identify actual saved layouts |
| Validate fixed-assignment task pressure / horizon | [load_layouts](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_validation.py:70), [_cap_pool_analysis](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_validation.py:110), [main](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_validation.py:273) | Recomputed BFS, familiar/novel analytical reward gaps and completion bounds; not learned-policy evaluation |
| Exhaustively audit geometry versus optimum | [assignment_times](/juice6/u/jshe/emergent_partner_grid/data_prep/check_geometry_goal_dependence.py:81), [prediction_limit](/juice6/u/jshe/emergent_partner_grid/data_prep/check_geometry_goal_dependence.py:112), [analyze_pool](/juice6/u/jshe/emergent_partner_grid/data_prep/check_geometry_goal_dependence.py:137), [main](/juice6/u/jshe/emergent_partner_grid/data_prep/check_geometry_goal_dependence.py:287) | Verify frozen corpus hash; enumerate all 1096 × 46 cases, ties, coordinate correlations and exact geometry-only ceilings |

The corpus builder retains old numbered comments such as “Step 7” and “Step 13.” Follow its actual calls and saved stage audit: the current branch has no separate basic-validity or feasibility-selection stage. `passes_feasibility` remains a utility, not an active stage in this builder. Exact optimum balancing and exact distance balancing solve different problems and are separate code paths.

### 2. Define the environment and observable evidence

| Step | Exact implementation | Audit meaning |
| --- | --- | --- |
| Read JSON coordinates and precompute partner routes | [_load_layout](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:266), [_bfs_next_actions](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:302), [CoordinationGrid.__init__](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:508) | JSON is `(row,column)`; runtime is `(x,y)`; BFS ties use fixed direction order |
| Encode/decode ego action | [encode_ego](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:214), [decode_ego](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:219) | `a=3*move+allocation`; movement order is UP, DOWN, RIGHT, LEFT, STAY |
| Enforce legal actions | [ego_action_mask](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:166), both actor-critic `__call__` methods, [CoordinationGrid.step_env](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:812) | 15 stored logits; NONE excluded; initialization has exactly one legal action; ten choices later |
| Random initial assignment and physical reset | [CoordinationGrid._build_state_for](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:675), [CoordinationGrid.reset_from_schedule](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:752) | Initialization independent of capability; first tick stationary; scheduled capability/layouts can be supplied externally |
| Joint online movement and goal reassignment | [CoordinationGrid.step_env](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:812) | Update actual assignment before this tick's partner navigation; reload destination delay only on a true goal change |
| Partner goal-specific delay and BFS execution | [CoordinationGrid._partner_next_move](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:792), `step_env` | Repeated request preserves cadence; delay zero permits immediate movement; goal switches to positive delay wait |
| No-influence control | `step_env`, [CoordinationGrid.get_obs](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:1027) | Actual initial assignment fixed; ego request echoed identically in both conditions; requested and actual assignments can differ |
| Walls, boundaries, same-cell collisions, swaps | `step_env` movement block | Reject invalid proposals; shared-cell and swapping proposals leave agents in place |
| Reward, timeout, round/episode boundaries | `step_env`, [CoordinationGrid._build_state_for](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:675) | Success on distinct goals gives +1, other ticks −0.01; round horizon 100; final done only after round 20 |
| Hide capability, expose behavior | [CoordinationGrid.get_obs](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:1027) | Five grid channels plus request one-hot and initialization flag; capability absent; visibility parameter exists but default is fully visible |
| Record completion tick correctly | `step_env` `info['round_time']`, evaluator `rollout_condition.body` | Post-transition completion time; returned physical state can already be the next round |

For a one-move partner route, partner arrival is environment step 2: step 1 is the initialization transition. With route length `P>0` and held assignment, arrival is `2+(P-1)*(d+1)`; ego arrival is `1+E`. Their maximum is the analytical completion time. That formula assumes independent shortest paths and no collisions, so it is not a dynamic online oracle.

**Reward accounting detail discovered during review:** [_reward_from_time](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:91) computes successful analytical reward as `success_reward - step_penalty*T`. The executed environment gives `success_reward - step_penalty*(T-1)` because success replaces that tick's penalty. Thus analytical successful rewards are one penalty (0.01 by default) lower. When both assignments succeed this constant cancels in their reward difference; success/failure comparisons need separate care. Existing behavior and saved statistics are preserved; the helper is now commented to expose the distinction.

### 3. Sample exposure, train, save, and audit

| Step | Exact implementation | Evidence / controls |
| --- | --- | --- |
| Select condition/profile pool | [build_network](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:325), [resolve_training_capability_pool](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:355) | RNN/MLP and diverse/single; `make_train` routes influence |
| Encode observations | [CNN.__call__](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:117), [CommObsEncoder.__call__](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:162) | Six convolutions, 64 grid features + 8 allocation features projected to width 128; no delay label input |
| Preserve memory across rounds | [ScannedRNN.__call__](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:85), [ActorCriticCommRNN.__call__](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:199) | GRU reset flags refer to new partner episodes, not physical round resets |
| Feedforward memory control | [ActorCriticCommMLP.__call__](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:266) | Same encoder/masked actions/PPO; one dense layer replaces recurrence |
| Original random profile/layout scheduling | [build_schedule](/juice6/u/jshe/emergent_partner_grid/train/episode_scheduler.py:49), [initial_episode_cursor](/juice6/u/jshe/emergent_partner_grid/train/episode_scheduler.py:101), [summarize_schedule](/juice6/u/jshe/emergent_partner_grid/train/episode_scheduler.py:115) | Uniform random draws, coverage diagnostic; finite counts need not balance exactly |
| Counterbalanced packet design | [build_schedule](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:50), [BalancedSchedule.summary](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:37) | 20-layout packet given once to each profile; five without-replacement passes close a 1096-layout cycle; independent layout/order RNGs |
| Unique asynchronous episode dispatch | [initial_queue](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:111), [dispatch](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:118), [lookup](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:133) | Global episode IDs; simultaneous completions ordered by slot; allocation imbalance at any prefix ≤1 episode |
| Install balanced sampler in each protocol | [patch_trainer](/juice6/u/jshe/emergent_partner_grid/bash/submit_counterbalanced_training.py) | Patches frozen trainer scheduling/reset/audit sites; active root trainer remains random |
| Initialize policy/optimizer/environment | [make_train.train](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:520), [make_train.create_learning_rate_fn](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:454) | Learner-key initialization; gradient clipping, Adam, 5% warmup/cosine schedule |
| Collect parallel PPO buffer | [make_train.train._update_step._env_step](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:578) | Only ego transitions train; scheduled resets for final-done workers; collected memory/log probabilities stored |
| Correct recurrent replay reset alignment | [Transition](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:380), `_env_step`, loss function below | `reset` is pre-observation; `done` is post-transition; they are distinct fields |
| Compute advantages and value targets | [make_train.train._update_step._calculate_gae](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:722) | Backward GAE; post-transition `done` stops bootstrap across episodes |
| Replay time-ordered minibatches | [make_train.train._update_step._update_epoch](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:754) | Shuffle environment sequences, not timesteps; replay from collected initial carry |
| Optimize clipped PPO/value/entropy losses | [make_train.train._update_step._update_epoch._update_minbatch._loss_fn](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:764) | Normalized advantages, clipped ratio and value, entropy term; partner has no learned optimizer |
| Log return, success, and exposure | [make_train.train._update_step._log_cb](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:1016), [accumulate_audit](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:145) | Logged returns describe rollout before its gradient update; actual audits count starts, completions, and ticks |
| Save weights/config and built-in evaluation | [main](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:1241), [save_params](/juice6/u/jshe/emergent_partner_grid/jaxmarl/wrappers/baselines.py), [evaluate_policy](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:1079) | Safetensors, resolved configuration, train/test summary JSON; full training trajectories are not saved |
| Export partial-cutoff empirical matrices | [write_audit](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:168) | `*_sampling_audit.{json,npz}`, active episode IDs/round times, allocated prefix, profile×layout matrices |
| Independently reconstruct actual exposure | [main](/juice6/u/jshe/emergent_partner_grid/train/audit_counterbalanced_results.py:17) | Reconstruct schedule prefix and subtract active tails; compare starts/completions exactly; unequal elapsed steps expected |
| Restore ignored schedules reproducibly | [rebuild](/juice6/u/jshe/emergent_partner_grid/train/rebuild_counterbalanced_schedules.py:57), [save_original_archive](/juice6/u/jshe/emergent_partner_grid/train/rebuild_counterbalanced_schedules.py:37) | Use frozen generator and require original NPZ byte checksum |
| Plot training curves | [smooth](/juice6/u/jshe/emergent_partner_grid/train/plot_training_curves.py), [main](/juice6/u/jshe/emergent_partner_grid/train/plot_training_curves.py) | Align actual environment steps; exclude missing completed-episode metrics from smoothing |

The main YAML specifies 256 workers × 256 ticks/update, four epochs, 64 minibatches, learning rate 5e−4, clip 0.2, gamma 0.99, GAE lambda 0.95, entropy coefficient 0.01, value coefficient 1, gradient norm 0.25. Integer budgeting gives **915 updates / 59,965,440 transitions**, not exactly 60 million. The schedule seed (2026) is separate from learner seeds 1–5 and evaluation seed 12345.

The global queue balances **allocated episodes**. At the fixed-step cutoff some of 256 workers are unfinished, so started/completed exposure can differ. Slower profiles also require more ticks. The empirical matrices and independent reconstruction are the audit of what happened, while the balanced plan describes what was allocated.

### 4. Freeze, validate, and launch experiments

| Step | Exact implementation | Side effects / boundary |
| --- | --- | --- |
| Prepare paired counterbalanced v1/v2 batch | [prepare](/juice6/u/jshe/emergent_partner_grid/bash/submit_counterbalanced_training.py), [relocate_v1_imports](/juice6/u/jshe/emergent_partner_grid/bash/submit_counterbalanced_training.py) | Recover v1 revision, copy active v2, freeze corpus/schedules/hashes, construct forty training commands; no submission in preparation |
| Preflight preservation and smoke checks | [check_preserved_implementation](/juice6/u/jshe/emergent_partner_grid/train/validate_counterbalanced_training.py:17), [validate](/juice6/u/jshe/emergent_partner_grid/train/validate_counterbalanced_training.py:51) | Checks unchanged scientific implementation, regression tests, bounded training/evaluation, writes preflight record |
| Verify frozen evidence before submission | [verify_manifest](/juice6/u/jshe/emergent_partner_grid/bash/submit_counterbalanced_training.py) | Exact hashes for source, corpus and schedules |
| Submit/resume training and dependent evaluation | [submit](/juice6/u/jshe/emergent_partner_grid/bash/submit_counterbalanced_training.py) | Slurm submission is an actual side effect; persist IDs; counterbalanced evaluation has `afterok` on its twenty policies |
| Map condition presets and run Hydra trainer | [bash/train_final_experiment.sh](/juice6/u/jshe/emergent_partner_grid/bash/train_final_experiment.sh) (`case CONDITION`, final Python command) | Three condition switches, learner seed, protocol, paths and resource settings; supports frozen old script paths |
| Bulk evaluate saved configs | [bash/eval_all_checkpoints.sh](/juice6/u/jshe/emergent_partner_grid/bash/eval_all_checkpoints.sh) (`for cfg` loop) | Configuration/weights pairs; 20 episodes/profile; hidden export for RNNs; skips existing summary files |

This documentation task does not submit jobs, regenerate full corpora, retrain policies, or overwrite completed evaluation artifacts. Cancellations, queue timestamps, cleanup and file relocations in the project log are administrative history recorded in manifests/logs; they are not simulation steps or callable scientific methods. Some were performed interactively and have no standalone function to cite.

### 5. Evaluate realized behavior and compare policies

| Step | Exact implementation | Interpretation |
| --- | --- | --- |
| Restore checkpoint and saved condition | [load_params](/juice6/u/jshe/emergent_partner_grid/jaxmarl/wrappers/baselines.py), [main](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:431) | Architecture/settings come from companion config; evaluator restores influence explicitly and guards protocol |
| Evaluate familiar and novel capability populations | [rollout_condition](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:184), [evaluate_policy](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:1079) | 24×20 familiar and 22×20 novel partner episodes; same training-layout pool |
| Record post-observation action-producing memory | [rollout_condition.body](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:239) | `h_t=RNN(h_(t-1),o_t)` is stored with that action's context; not pre-observation carry |
| Remove scan padding, retain terminal outcome | [_first_done_alive_mask](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:312) | Valid through first final done **inclusive**; scan bound is 20×100=2000 ticks |
| Score realized performance and assignment changes | [summarize](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:330) | Success, returns, successful completion time; exclude initialization from allocation decisions; actual/requested goals separate |
| Historical fixed-allocation optimum/regret | [per_layout_bfs](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:85), [analytical_alloc_rewards](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:117), [optimal_allocation_and_regret](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:151); use frozen v1 evaluator for historical headlines | Static reward-based reference; no-influence ego requests are hypothetical |
| Export reproducible rollout evidence | [_save_hdf5](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:417) | `_train.h5`, `_test.h5`, `_summary.json`; protocol/config attributes; hidden arrays only when requested |
| Five-seed v1/v2 performance comparison | [main](/juice6/u/jshe/emergent_partner_grid/eval/compare_allocation_protocols.py) | Means and sample SDs across trained policies; distinguish original corpus difference from shared-corpus counterbalanced comparison |
| Choose best learner per condition using training | [select_seeds](/juice6/u/jshe/emergent_partner_grid/eval/plot_best_training_seed.py), [main](/juice6/u/jshe/emergent_partner_grid/eval/plot_best_training_seed.py) | Final logged return by default, explicit trailing window optional; no held-out selection |
| Uncertainty for selected-policy held-out behavior | [held_out_metrics](/juice6/u/jshe/emergent_partner_grid/eval/plot_best_training_seed.py) | Resample whole episodes within each of 22 fixed profiles; excludes between-policy variation |
| Choose one common learner seed | [main](/juice6/u/jshe/emergent_partner_grid/eval/plot_common_training_seed.py) | Equal-weight mean training scores across the eight counterbalanced policies, choice finalized before reading held-out outcomes |

Online-v2 faster-goal fractions are **speed adherence**, not dynamic optimal-allocation accuracy. Remaining distances, collisions and switch delays can matter. A high success rate alone does not establish capability inference or causal use of memory.

### 6. Decode and visualize recurrent representations

| Step | Exact implementation | Data / safeguard |
| --- | --- | --- |
| Discover all RNN policies; reject missing/ambiguous inputs | [discover_rollout_files](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Three recurrent conditions × five learner seeds × two population files; MLP excluded |
| Keep protocols separate | [validate_allocation_protocols](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | One protocol per analysis/output tree; old untagged files treated fixed_v1 |
| Validate episode/profile/round evidence | [load_checkpoint_rollouts](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py), [validate_rollout_dataset](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py), [episode_valid_length](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | 920 episodes/checkpoint, 46 profiles ×20 repetitions; fixed capability; valid terminal-inclusive 128-D states |
| Absolute-history features | [prefix_mean_hidden](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Prefix means at 1,50,…,400 ticks, clipped to actual valid episode length |
| Cumulative round-history features | [round_prefix_mean_hidden](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Episode prefix through each round's inclusive endpoint; 20 features/episode |
| Final-history UMAP features | [final50_mean_hidden](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Last 50 valid states, or all if shorter |
| Shared 16/4 episode split within profile | [make_probe_split](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py), [split_masks](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | 736 probe-training /184 probe-test episodes; shared repetition indices; no entire profile held out |
| Independent red/blue affine probes | [train_linear_probe](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py), [run_probes](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | 128→10 softmax classifier, full-batch Adam lr .01 for 1000 steps; separate weights/moments per target/cutoff/policy |
| Distance score, exact accuracy and MAE | [distance_aware_accuracy](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py), [evaluate_probe](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | `1-abs(predicted_delay-true_delay)/9`; high partial-credit score does not imply high exact accuracy |
| Familiar/novel held-out episode scores | `run_probes` subset masks | Novel profiles are unseen in **policy** training but are included in **probe** fitting; not novel-capability decoder transfer |
| Random-feature/true-label control | [run_random_baseline](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Normal(0,1) vectors and real labels, same split/fitter; five vector seeds |
| Shuffled-label diagnostic | `run_probes` shuffled branch | Permute paired `(d_R,d_B)` labels for real t=400 features; diagnostic, not causal capability intervention |
| Policy-seed aggregation and paired contrasts | [summarize_probes](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py), [summarize_condition_differences](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Mean/sample SD/bootstrap over five learner seeds; nominal-seed pairing is descriptive |
| Representative policy selected from behavior | [select_paper_style_seed](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py), [select_recorded_common_seed](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Training-return evidence or explicit familiar proxy; batch-scoped common-seed provenance; no probe-score selection |
| Separate UMAP per policy | [run_umap](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py), [make_umap_figures](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | 920 final-history vectors, `min_dist=1`, `n_neighbors=919`, random state 42; color by `d_B-d_R` or individual delays |
| Save models, provenance, figures and report | [main](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py), [write_metadata](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py), [write_report](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Models/NPZ, episode and split records, all-seed/selected curves, source evidence/software settings |
| Compare allocation versions | [main](/juice6/u/jshe/emergent_partner_grid/eval/compare_representation_versions.py:24) | Per-version fits remain independent; individual-seed and mean curves, endpoint contrasts |
| Inspect task exposure beside final decoding | [main](/juice6/u/jshe/emergent_partner_grid/eval/summarize_decoding_task_exposure.py:20) | Partner red occupancy during valid non-initialization ticks and round endings joined to round-20 scores; not movement counts |

Do not pool hidden-coordinate systems from independently trained policies. UMAP organization or delay decoding establishes information recoverability, not that the actor uses that information to assign goals. The historical log's scientific conclusion remains mixed; documentation and test completion are not scientific replication criteria.

### 7. Separate released-code probe reproduction

The newer [eval/representation_analysis_strict.py](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py) deliberately implements a different analysis from the primary written-method probe. It extracts pinned definitions from local reference source, verifies their hash, and does not modify them.

| Step | Exact implementation | Difference from primary analysis |
| --- | --- | --- |
| Load verified upstream definitions | [load_upstream](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py) | Local pinned JAX/Flax definitions, source-hash guard |
| Instrument original split/evaluation/fitting | [RecordedUpstreamProbe.record_split](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py), [RecordedUpstreamProbe.record_eval](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py), [RecordedUpstreamProbe.fit](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py) | Preserve source RNG/fitted weights while logging IDs/checkpoints |
| Sequence cutoffs with selected-weight warm starts | [run_sequence](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py) | AdamW decay .001, 1001 default updates, scalar-label-stratified splits, best **test** checkpoint; splits change between fits |
| Released baseline and supplementary true-label control | [run_baselines](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py) | Released baseline uses random features **and random integer labels**; true-label control saved separately |
| Score subsets and report overlap | [score_subset](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py), `run_sequence` | Save current test overlap with earlier training IDs; test-selected scores are not independent generalization estimates |
| Orchestrate and disclose extensions | [main](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py), [write_report](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py) | Grid round curves and all-seed summaries; explicit NumPy seeding; UMAP reused from unchanged states |
| Compare analysis methods | [main](/juice6/u/jshe/emergent_partner_grid/eval/compare_representation_protocols.py:15) | Keep analysis-method and environment-version axes distinct; baseline target differences labeled |

The historical fidelity audit in the project log compares source/paper methods; it is not a runtime experiment. Its replay-reset defect applies to historical v1. The active v2 `Transition.reset` separates the flags correctly. Counterbalanced v1 intentionally preserves its original PPO implementation, so documentation must not claim the v2 fix retroactively repaired its trained weights.

### 8. New raw-state partner geometry, driven dynamics, and interventions

The untracked [eval/partner_dynamics](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics) package was changing during this review and postdates the project log. Its README's initial “folder setup only” statement is stale relative to present executable modules and results folders. Treat the functions below as implemented code paths; coverage/completion requires the per-policy artifacts and reports. The attached specification contains broader proposed methods, so a requested method is not necessarily fully executed.

| Step | Exact implementation | Audit meaning |
| --- | --- | --- |
| Dispatch stages and isolate frozen imports | [main](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/__main__.py:10), [load_frozen](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/collect.py:14) | Batch/protocol/condition/seed scoped outputs; subprocess per protocol for frozen-source collection |
| Inventory inputs, hash provenance, split repetitions | [inventory](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/data.py:78), [split_repetitions](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/data.py:29), [sha](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/data.py:18) | Discovery/validation/test episodes; recorded source protocol; no temporal sample split within an episode |
| Event-align **raw** hidden states | [round_boundaries](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/data.py:41), [stream_events](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/data.py:126) | Start/midpoint/end of each of 20 rounds, sixty event samples per episode; not cumulative means |
| Train-only scaling, regularized regressors, validation selection | [Ridge.fit](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/numeric.py:5), [select_ridge](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/numeric.py:20), [r2](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/numeric.py:32) | Fit scaling/coefficient state on discovery data, choose alpha by validation |
| Raw-state PCA, competing factor models, population/layout/time transfer | [analyze](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/geometry.py:16) | PCA/ridge instead of sparse crossed dPCA; capability regressions compared with true-faster-goal conditional prior; target-layout exclusion allows prior-history overlap |
| Cross-validated representational distances | [cross_distance](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/numeric.py:48), geometry `analyze` | Discovery-only diagonal noise precision; independent repetition halves; pairwise profile scores descriptive |
| Grouped uncertainty | [grouped_ci](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/numeric.py:36), [build](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/report.py:9) | Episode groups within policy; seed groups in report; keep unit explicit |
| Select diagnostic layouts and capture richer histories | [diagnostic_layouts](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/collect.py:33), [run](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/collect.py:47) | Default six profiles ×four repetitions, eight diagnostic layouts, four-round histories; before/after carry, encoded input, full observation/state, keys and outcomes |
| Extract/replay original encoder/cell/readout | [PolicyAdapter._full](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/adapters.py:21), [PolicyAdapter._update](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/adapters.py:31), [PolicyAdapter._readout](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/adapters.py:38) | Direct frozen weights; verify probabilities, memory and categorical action replay before analysis |
| Exact local memory/input derivatives | [PolicyAdapter.__init__](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/adapters.py:7), collection `run` | JAX `jacfwd` for hidden/input Jacobians, finite-difference checks, spectral/gain and short-horizon products |
| Learn partner-predictive directions on discovery histories | [partner_projector](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/interventions.py:5) | Ridge targets orientation + delays; SVD basis, at most rank three; validation check separate |
| Perturb recipient memory with controls | [perturb](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/interventions.py:21), collection `run` and nested `branch` | Whole/subspace/attenuation/random/nonpartner/sham changes, paired physical starts and RNG; zero-change identity check |
| Replay memory changes against identical inputs | [run](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/replay.py:12), [main](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/replay.py:77) | Same baseline observations at initialization and next tick; compare first learned allocation for each protocol; separate from free-running rewards |
| Preserve trajectory boundaries in driven fitting | [BoundaryDMDc.compute_svd](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/input_dsa.py:23), [pairs](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/input_dsa.py:61) | Assemble within-round current→next pairs before concatenation; do not invent inter-episode transitions |
| Fit and export input-driven DMDc | [fit](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/input_dsa.py:45), [arrays](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/input_dsa.py:76) | Pinned reference components plus explicit boundary adapter; delay embedding, rank and regularization |
| Validate rank/delay, predict held-out and short-horizon states | [evaluate](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/dynamics.py:13), [run](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/dynamics.py:27) | Discovery projections/fits, validation selection; persistence, input-omitting and input-only comparisons |
| Compare regimes and networks with input-aware metric | [distance](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/input_dsa.py:85), dynamics `run` | Separate joint/intrinsic/input distances; cross-network comparison uses common seeded raw-input projection rather than learned encoder axes |
| Aggregate paired outcomes and emit reports | Dynamics `run`, [build](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/report.py:9) | Predicted-state quality separate from realized round reward/action changes; report per-policy coverage and limitations |

Input timing matters: saved hidden state `h[t]` is post-observation, so the transition to `h[t+1]` is driven by encoded input `e[t+1]`. V2 initialization probabilities remain forced even after memory changes; its first learned allocation occurs later. Whole-carry swaps can transfer other remembered state, so their behavioral effect alone is not capability-specific evidence. Short four-round capture does not cover full late-episode dynamics or capability switching. Optional slow-point analysis is not established by the listed functions.

## Project-log coverage and limits

| Project-log section | Where the implementation is mapped above |
| --- | --- |
| What we are testing; design summary; four conditions | Reading the experiment; step 2 environment; step 3 networks/condition pool |
| Capability semantics; action space/influence; layouts; reward/horizon | Steps 1–2; analytical versus executed reward qualification |
| PPO/network configuration; capability/layout scheduler | Step 3; YAML and random/counterbalanced distinction |
| Counterbalanced launch; balanced-layout launch; how to launch | Step 4; manifests and matching frozen source |
| Checkpoints/config/data provenance; repository reorganization | Versions/frozen evidence; `repo_paths`; artifact map below |
| Evaluation; online-v2 implementation/verification | Steps 2, 3, 5; terminal-inclusive masking; tests below |
| Historical fixed-allocation behavioral evaluation | Matching frozen v1 evaluator; step 5 comparisons |
| Historical representation entry point, features, split, fitting, baselines, UMAP, outputs | Step 6; separate step 7 for subsequent strict released-code reproduction |
| Historical analytical validation | Step 1 `capability_validation` and static-model assumptions |
| Scientific replication criteria and conclusion | Assess behavior, decoding and causal use separately; no pass/fail automated scientific conclusion added |
| Fidelity audit and recommended follow-ups | Historical source comparison; active v2 reset correction; strict probes separately; other recommendations remain proposals unless implemented here |
| Deferred/not-in-this-experiment | Capability **profile** changes mid-episode, unseen-layout policy generalization, communication/local-blindness conditions, larger populations and additional environments are not core completed experiments |

Online reassignment changes the partner **goal**, not the partner **capability**. Shuffled probe labels do not shuffle capabilities in the physical environment. A constructor visibility knob is not an executed blindness condition. The newer memory perturbation routines address some deferred representation-intervention ideas but do not establish completion of every proposed study.

Retired stage A/B/C/D and generator-sweep entry points were removed during prior cleanup; they are not required experimental steps and have no active function to reference. Historical numerical assertions and cluster completion statuses remain evidence in logs/manifests; this review inspects their implementation and does not rerun full training to reproduce those assertions. Only CoordinationGrid remains in the active environment package. Completed frozen sources retain inherited modules to preserve imports and source hashes.

## Artifact map for an audit

| Evidence | Location / writer |
| --- | --- |
| Generator settings, per-candidate filtering and splits | `data_prep/<corpus>/manifest.json`, `diagnostics/`; corpus builder |
| Actual geometry | `data_prep/<corpus>/layouts/train/*.json` and compiled NPZ; generator/builder/balancer |
| Forty-run design, resources, dependencies, hashes | `train/manifests/sbatch_<batch>.json`; batch preparer/submitter |
| Exact experiment sources, copied corpus, shared schedules | `train/run_snapshots/<batch>/`; batch preparer |
| Weights and resolved settings | `train/train_logs/<version>/<batch>/<condition>_seedN{.safetensors,_config.json}`; trainer |
| Collection-time curves | `train/slurm_logs/` plus `train/hydra_outputs/`; trainer/launcher |
| Started/completed/tick exposure | Companion `_sampling_audit.{npz,json}` and `train/sampling_audits/<batch>/`; counterbalanced audit writers |
| Saved evaluation trajectories | `eval/eval_out/<version>/<batch>/*_{train,test}.h5`; standalone evaluator |
| Behavior comparisons and selection evidence | `eval/protocol_comparison/`; comparison/seed-selection scripts |
| Primary probe models/splits, numeric tables, metadata | `eval/representation_results/<batch>/<version>/`; representation analysis |
| Strict method trace and comparison | Separate `<batch>_strict_replication_of_covercooked/`; strict analysis |
| Raw-state and targeted-dynamics evidence | `eval/partner_dynamics_results/<batch>/<protocol>/<condition>_seedN/`; dedicated package |

## How to verify this documentation change

Comments below core functions explain purpose, input/output meaning, algorithm order, and the audit assumptions. Each is marked `# Audit guide:`. The function inventory links to the definitions; those blocks follow existing docstrings. Shell launchers have corresponding execution-block comments.

Executable Python syntax trees are compared with copies taken immediately before annotation, so existing working-tree changes are preserved and the annotations themselves must not change logic. Frozen snapshots/reference implementations remain unchanged. Existing regression suites check environment boundaries and replay, action selection, scheduling/exposure, corpus selection and probe behavior. See the validation record at the end for the checks actually run for this edit.

## Complete callable inventory for the active experiment code

The tables include helpers and nested functions so names in the stage map can be traced to exact definitions. Private helpers are implementation details; `main` is usually a command-line entry point. Class rows identify data containers or networks. Annotation explanations are given for the core functions; existing docstrings supply descriptions for remaining helpers. Framework wrappers are limited to checkpoint serialization used by this experiment.


### Historical counterbalanced source links

| Version | Environment | Trainer | Standalone evaluator |
| --- | --- | --- | --- |
| v1 | [CoordinationGrid.step_env](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/counterbalanced1096_20261002_235609/v1_balanced_training/source/jaxmarl/environments/coordination_grid/coordination_grid.py:757) | [make_train](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/counterbalanced1096_20261002_235609/v1_balanced_training/source/baselines/IPPO/ippo_rnn_coordination_grid.py:382) | [summarize](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/counterbalanced1096_20261002_235609/v1_balanced_training/source/analysis/evaluate_partner_modelling.py:292) |
| v2 | [CoordinationGrid.step_env](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/counterbalanced1096_20261002_235609/v2_balanced_training/source/jaxmarl/environments/coordination_grid/coordination_grid.py:769) | [make_train](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/counterbalanced1096_20261002_235609/v2_balanced_training/source/baselines/IPPO/ippo_rnn_coordination_grid.py:368) | [summarize](/juice6/u/jshe/emergent_partner_grid/train/run_snapshots/counterbalanced1096_20261002_235609/v2_balanced_training/source/analysis/evaluate_partner_modelling.py:293) |

These links point to the completed categorical counterbalanced batch. The frozen v1 code preserves historical fixed-allocation behavior, including its original recurrent replay logic; do not substitute the commented current v2 trainer for reproducing those runs.

### repo_paths.py

| Definition | Purpose / audit detail |
| --- | --- |
| [_archive_path](/juice6/u/jshe/emergent_partner_grid/repo_paths.py:28) | Helper called by this module; inspect its linked body for arguments and return values. |
| [resolve_path](/juice6/u/jshe/emergent_partner_grid/repo_paths.py:38) | Translate historical experiment paths to current active or archived locations only for reading. Prefer a path that already exists, and leave unrelated external paths alone. Preserve original files/config values so their hashes and historical evidence remain intact. |
| [resolve_record_paths](/juice6/u/jshe/emergent_partner_grid/repo_paths.py:78) | Return a recursively reconstructed record whose repository path strings resolve to present locations. Do not mutate the original manifest. This lets current readers consume immutable records that refer to retired dev/analysis/logs locations. |

### data_prep/balance_ego_distances.py

| Definition | Purpose / audit detail |
| --- | --- |
| [expected_counts](/juice6/u/jshe/emergent_partner_grid/data_prep/balance_ego_distances.py:26) | Find fractional counts for each joint red-distance/blue-distance bin with the same target in every row and column. A maximum-flow check first proves the requested quotas are feasible. The entropy solution spreads selection across available layouts without adding partner distances as a selection criterion. |
| [expected_counts.residual](/juice6/u/jshe/emergent_partner_grid/data_prep/balance_ego_distances.py:73) | Helper called by expected_counts; inspect its linked body for arguments and return values. |
| [_fractional_cycle](/juice6/u/jshe/emergent_partner_grid/data_prep/balance_ego_distances.py:87) | Find an alternating loop linking fractional row/column bins. Adjusting opposite edges of such a loop preserves both marginal totals. This structure lets round_counts convert fractional selections to integers without independently rounding away the quotas. |
| [_fractional_cycle.visit](/juice6/u/jshe/emergent_partner_grid/data_prep/balance_ego_distances.py:102) | Helper called by _fractional_cycle; inspect its linked body for arguments and return values. |
| [round_counts](/juice6/u/jshe/emergent_partner_grid/data_prep/balance_ego_distances.py:127) | Move counts along alternating cycles until all are integers. Randomize the direction using the available upward/downward room, preserving expected counts as well as exact marginals. Final bin counts say how many distinct layouts to sample, not how often to repeat layouts. |
| [_write_csv](/juice6/u/jshe/emergent_partner_grid/data_prep/balance_ego_distances.py:153) | Helper called by this module; inspect its linked body for arguments and return values. |
| [main](/juice6/u/jshe/emergent_partner_grid/data_prep/balance_ego_distances.py:160) | Exclude layouts whose ego distances fall outside the chosen range, solve simultaneous marginal quotas, and sample without replacement within each joint bin. Preserve selected source JSON bytes and IDs. Write exclusion reasons and counts so an auditor can distinguish the distance cutoff from the random quota selection. |

### data_prep/build_final_corpus.py

| Definition | Purpose / audit detail |
| --- | --- |
| [_rc_to_xy](/juice6/u/jshe/emergent_partner_grid/data_prep/build_final_corpus.py:80) | Helper called by this module; inspect its linked body for arguments and return values. |
| [_xy_to_rc](/juice6/u/jshe/emergent_partner_grid/data_prep/build_final_corpus.py:84) | Helper called by this module; inspect its linked body for arguments and return values. |
| [apply_sym_to_env](/juice6/u/jshe/emergent_partner_grid/data_prep/build_final_corpus.py:88) | Rotate/reflect walls, starts, and both goal positions together using one of eight square symmetries. Recompute metadata afterward. Geometry should preserve BFS distances and capability-selection statistics; main checks that invariance after transformation. |
| [_env_fingerprint](/juice6/u/jshe/emergent_partner_grid/data_prep/build_final_corpus.py:108) | Hash geometry including both starts and colored goals so duplicate candidate layouts can be recognized. The same fingerprint is used to verify split uniqueness. This exact geometry deduplication is distinct from canonicalizing every rotation/reflection into one equivalence class. |
| [_save_hist](/juice6/u/jshe/emergent_partner_grid/data_prep/build_final_corpus.py:126) | Helper called by this module; inspect its linked body for arguments and return values. |
| [_write_candidate_csv](/juice6/u/jshe/emergent_partner_grid/data_prep/build_final_corpus.py:147) | Helper called by this module; inspect its linked body for arguments and return values. |
| [main](/juice6/u/jshe/emergent_partner_grid/data_prep/build_final_corpus.py:162) | Run the entire selection pipeline and save a per-candidate audit. Generate valid candidates, deduplicate, compute training-capability statistics, screen observability/allocation balance, apply delta-reward and horizon filters, select final survivors, apply balanced D4 transforms, shuffle/split, and save. With no n_final, retain all survivors with exact 50/50 optima and no ties; explicit n_final selects the capped branch. Historical numbered headings contain gaps because separate basic-validity and feasibility stages were removed from the current pipeline. |
| [_reproduce_command](/juice6/u/jshe/emergent_partner_grid/data_prep/build_final_corpus.py:819) | Record the actual generator/selection settings in a rerunnable command. Omit optional settings that were not supplied so the manifest distinguishes uncapped exact selection from historical capped sampling. |

### data_prep/capability_selection.py

| Definition | Purpose / audit detail |
| --- | --- |
| [completion_time](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:59) | Estimate joint completion for a fixed assignment from two independent shortest paths. Include the stationary initialization tick; the partner first moves immediately afterward and inserts delay ticks between later moves. Take the slower agent completion and mark horizon failures infeasible. This is a static analytical model: collisions and later reassignment trajectories are not simulated. |
| [_reward_from_time](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:91) | Use the historical analytical convention: success_reward minus step_penalty times the full completion time. step_env instead charges the penalty only on unsuccessful ticks, so this formula is one penalty lower for successful rounds. Preserve this distinction when auditing saved corpus statistics. Horizon failures pay the penalty on every tick. |
| [layout_bfs_distances](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:113) | Recompute the four wall-aware distances from layout contents, instead of trusting stored metadata. Convert JSON row/column coordinates consistently. The resulting ego-to-red/blue and partner-to-red/blue distances are inputs to fixed-assignment calculations. |
| [layout_bfs_distances.bfs](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:135) | Helper called by layout_bfs_distances; inspect its linked body for arguments and return values. |
| [LayoutStats](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:173) | Analytical selection statistics for one layout across a cap pool. |
| [LayoutStats.to_metadata_dict](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:209) | JSON-safe subset written into per-layout metadata. |
| [evaluate_layout](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:221) | Score both complementary assignments for every capability pair. The capability- aware oracle chooses the better reward separately for each profile; the blind baseline chooses one assignment by expected reward across profiles. Count red/blue optima and ties, then summarize observability, feasibility, and oracle-minus-blind reward. An oracle is an analytical reference here, not a runnable agent. |
| [layout_stats](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:354) | Convenience wrapper reading BFS distances from a metadata dict. |
| [passes_observability](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:376) | Require the partner to travel far enough toward both goals for movement delays to be observable. This filters static distances, not a demonstrated amount of information recovered by the learned RNN. |
| [passes_allocation_balance](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:394) | Check that optimal assignments change across capabilities and are sufficiently balanced, with ties controlled. The broad screen precedes delta-reward and horizon thresholds; moving it changes those thresholds input distributions. |
| [passes_exact_allocation_balance](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:416) | Require exactly half the supplied profiles to strictly favor each ego goal, with zero ties. For the 24-profile training pool this means 12 red and 12 blue optima. This newer final screen is distinct from balancing ego-distance histogram marginals. |
| [passes_feasibility](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:427) | Oracle must complete on essentially every cap pair. |
| [passes_delta_reward](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:433) | Simple absolute threshold on the oracle-vs-blind reward gap. |
| [worst_case_oracle_steps](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:443) | Max oracle completion across the cap pool. NaN if none feasible. |
| [horizon_outlier_mask](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:448) | Keep worst-case oracle completion times strictly below the survivor mean plus the configured number of standard deviations. The existing degenerate-distribution branch handles a common identical value. Calculate this on the intended survivor pool, not on already selected final layouts. |
| [derive_max_steps](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:478) | Convert the maximum retained analytical completion to a recommended integer horizon with margin. The experiment configuration still explicitly sets its executed horizon; recording a recommendation does not update saved training runs. |
| [summarize_distribution](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:500) | Return min/quantiles/max/mean/std of values (NaNs dropped). |
| [summarize_stats](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:525) | Distributions over a list of :class:LayoutStats for the manifest. |
| [summarize_stats.col](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_selection.py:527) | Helper called by summarize_stats; inspect its linked body for arguments and return values. |

### data_prep/capability_validation.py

| Definition | Purpose / audit detail |
| --- | --- |
| [load_layouts](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_validation.py:70) | Read each saved grid and recompute BFS distances from geometry. This checks the actual corpus rather than relying on a generator settings description. The distances are cached for the analytical profile-by-layout calculations. |
| [load_layouts.rc2xy](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_validation.py:89) | Helper called by load_layouts; inspect its linked body for arguments and return values. |
| [_cap_pool_analysis](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_validation.py:110) | Compare capability-aware and fixed partner-blind assignments over every layout and profile in a population. Choose the blind allocation by expected reward, not success rate, because a generous horizon can make success rates almost identical. Report feasibility, reward gaps, allocation flips, and completion bounds under the static model. |
| [_per_cap_success_breakdown](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_validation.py:261) | Helper called by this module; inspect its linked body for arguments and return values. |
| [main](/juice6/u/jshe/emergent_partner_grid/data_prep/capability_validation.py:273) | Run the analytical validation separately for familiar and novel capability pools, with explicitly supplied horizon and reward settings. Save JSON evidence. This checks fixed-assignment task pressure and feasibility; it does not evaluate learned behavior or a dynamic switching oracle. |

### data_prep/check_geometry_goal_dependence.py

| Definition | Purpose / audit detail |
| --- | --- |
| [capability_profiles](/juice6/u/jshe/emergent_partner_grid/data_prep/check_geometry_goal_dependence.py:36) | Helper called by this module; inspect its linked body for arguments and return values. |
| [load_layouts](/juice6/u/jshe/emergent_partner_grid/data_prep/check_geometry_goal_dependence.py:47) | Helper called by this module; inspect its linked body for arguments and return values. |
| [assignment_times](/juice6/u/jshe/emergent_partner_grid/data_prep/check_geometry_goal_dependence.py:81) | Compute uncapped completion times for the two fixed assignments. Avoid turning two horizon failures into an artificial tied optimum. These times ignore agent collisions and online changes of assignment. |
| [correlation](/juice6/u/jshe/emergent_partner_grid/data_prep/check_geometry_goal_dependence.py:102) | Point-biserial Pearson r: red=1, blue=0; equal-time cases excluded. |
| [prediction_limit](/juice6/u/jshe/emergent_partner_grid/data_prep/check_geometry_goal_dependence.py:112) | Group records by the information available to a hypothetical geometry-only classifier and choose the majority optimal goal within each group. This yields an exact best accuracy for those keys under the evaluated profile weights, without fitting a classifier. |
| [analyze_pool](/juice6/u/jshe/emergent_partner_grid/data_prep/check_geometry_goal_dependence.py:137) | Cross every layout with every profile in the requested pool, compare assignment times, and measure ties, optimum frequencies, coordinate correlations, and geometry-only prediction bounds. Independence conclusions apply to this corpus and weighting rather than arbitrary populations. |
| [write_csv](/juice6/u/jshe/emergent_partner_grid/data_prep/check_geometry_goal_dependence.py:175) | Helper called by this module; inspect its linked body for arguments and return values. |
| [save_figure](/juice6/u/jshe/emergent_partner_grid/data_prep/check_geometry_goal_dependence.py:182) | Helper called by this module; inspect its linked body for arguments and return values. |
| [start_cells](/juice6/u/jshe/emergent_partner_grid/data_prep/check_geometry_goal_dependence.py:188) | Helper called by this module; inspect its linked body for arguments and return values. |
| [make_figures](/juice6/u/jshe/emergent_partner_grid/data_prep/check_geometry_goal_dependence.py:204) | Helper called by this module; inspect its linked body for arguments and return values. |
| [main](/juice6/u/jshe/emergent_partner_grid/data_prep/check_geometry_goal_dependence.py:287) | Verify the corpus against its frozen training manifest, recompute wall-aware distances, and run train/test/combined population audits. Save numerical tables, figures, and provenance so the 50 percent geometry-only ceiling is inspectable rather than inferred from synthetic examples. |

### data_prep/env_generator.py

| Definition | Purpose / audit detail |
| --- | --- |
| [GridEnv](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:64) | A single generated environment.  grid stores only walls/empties (values EMPTY/WALL). Agents and goals are tracked as separate coordinates so we can render them on top and encode them into the compiled numpy array. |
| [bfs_distances](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:86) | Measure shortest walking distances with walls respected, visiting cells in increasing distance order. Unreachable cells remain -1. These graph distances, rather than straight-line coordinate distances, feed layout geometry and feasibility checks. |
| [_all_pairs_reachable](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:108) | Helper called by this module; inspect its linked body for arguments and return values. |
| [_traversable_neighbors](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:125) | Helper called by this module; inspect its linked body for arguments and return values. |
| [bfs_path_counts](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:136) | Return (dist, count): count[v] = number of distinct shortest paths src->v. |
| [canonical_shortest_path](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:166) | Deterministic shortest path from src to dst (or [] if unreachable).  Reconstructed backwards from dst; ties broken by _NBR_ORDER. |
| [_structural_counts](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:191) | (num_junctions, num_dead_ends) over traversable cells. |
| [_path_overlap](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:206) | Jaccard overlap on cell-sets of two paths. |
| [_switching_cost](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:216) | Penalty for committing to first_goal for k steps then switching to other_goal:      cost = k' + d(pos_after_k, other) - d(start, other)  where k' = min(k, len(path_to_first_goal)-1) so we don't overshoot the goal when the first path is shorter than k. |
| [compute_metrics](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:243) | Measure geometry for a generated grid: ego/partner goal distances and structural path features. Attach these values as metadata for later filters and plots. These metrics describe static layouts; they are not learned-policy performance measurements. |
| [_sample_grid](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:309) | Helper called by this module; inspect its linked body for arguments and return values. |
| [_sample_env](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:320) | Repeatedly propose walls, starts, and goals until reachability and configured geometric requirements pass. Rejected proposals never enter the candidate corpus. Attach metrics to each accepted layout so downstream selection can be audited independently of generation. |
| [generate_envs](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:397) | Generate a reproducible collection using the configured seed and geometric rejection rules. Capability-sensitive selection happens later in build_final_corpus; a geometrically valid candidate need not yet provide useful partner-modelling pressure. |
| [encode_env](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:436) | Encode one layout as an integer grid suitable for a compiled NPZ corpus. Wall/start/goal codes describe geometry only; they do not encode hidden capability. JSON retains the accompanying metadata. |
| [env_metadata](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:451) | Return the metrics dict, computing it lazily if not already attached. |
| [_dumps_compact_arrays](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:464) | json.dumps with indent, but with the innermost arrays inlined.  Example:     [       [1, 2, 3],       [4, 5, 6]     ] instead of every scalar on its own line. |
| [_dumps_compact_arrays._collapse](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:476) | Helper called by _dumps_compact_arrays; inspect its linked body for arguments and return values. |
| [env_to_json_dict](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:483) | Helper called by this module; inspect its linked body for arguments and return values. |
| [render_env](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:496) | Helper called by this module; inspect its linked body for arguments and return values. |
| [_split_indices](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:532) | Helper called by this module; inspect its linked body for arguments and return values. |
| [save_split](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:549) | Write selected layouts, compiled arrays, and optional renders for a named split. Split sizes describe saved data; actual training/evaluation coverage is controlled by configurations and schedules. |
| [_parse_args](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:590) | Helper called by this module; inspect its linked body for arguments and return values. |
| [main](/juice6/u/jshe/emergent_partner_grid/data_prep/env_generator.py:636) | Parse generator options, generate candidates, partition them, and write geometry artifacts. This standalone generator lacks the capability-dependent filters used by the final-corpus builder. |

### data_prep/plot_layout_distributions.py

| Definition | Purpose / audit detail |
| --- | --- |
| [_summary](/juice6/u/jshe/emergent_partner_grid/data_prep/plot_layout_distributions.py:29) | Helper called by this module; inspect its linked body for arguments and return values. |
| [_style](/juice6/u/jshe/emergent_partner_grid/data_prep/plot_layout_distributions.py:37) | Helper called by this module; inspect its linked body for arguments and return values. |
| [_save](/juice6/u/jshe/emergent_partner_grid/data_prep/plot_layout_distributions.py:43) | Helper called by this module; inspect its linked body for arguments and return values. |
| [_csv](/juice6/u/jshe/emergent_partner_grid/data_prep/plot_layout_distributions.py:49) | Helper called by this module; inspect its linked body for arguments and return values. |
| [main](/juice6/u/jshe/emergent_partner_grid/data_prep/plot_layout_distributions.py:56) | Read saved geometry metadata and plot distance and wall-density distributions plus balancing feasibility diagnostics. These are descriptive corpus checks, not another training run. Inspect both goal marginals to verify a simultaneous distance-balance claim. |

### data_prep/render_layout_corpus.py

| Definition | Purpose / audit detail |
| --- | --- |
| [render_one](/juice6/u/jshe/emergent_partner_grid/data_prep/render_layout_corpus.py:17) | Render walls, both colored goals, and both starting positions for a saved layout. Coordinate placement follows the JSON row/column convention. A visual check can reveal swapped starts/goals that a distribution plot would miss. |
| [digest](/juice6/u/jshe/emergent_partner_grid/data_prep/render_layout_corpus.py:37) | Helper called by this module; inspect its linked body for arguments and return values. |
| [contact_sheet](/juice6/u/jshe/emergent_partner_grid/data_prep/render_layout_corpus.py:41) | Combine existing layout pictures into a labeled overview. The labels identify source layouts for manual review; rendering does not alter training geometry. |
| [main](/juice6/u/jshe/emergent_partner_grid/data_prep/render_layout_corpus.py:61) | Choose saved layouts, write individual renders and contact sheets, and record file digests. This is an inspection utility; it does not regenerate or filter the corpus. |

### train/audit_counterbalanced_results.py

| Definition | Purpose / audit detail |
| --- | --- |
| [main](/juice6/u/jshe/emergent_partner_grid/train/audit_counterbalanced_results.py:17) | Independently reconstruct allocated schedule prefixes for every saved training run. Subtract each active worker unfinished episode/round tail and compare predicted starts/completions with saved matrices. Export verification and actual profile exposure. This validates realized training bookkeeping beyond the sampler mathematical balance guarantee. |

### train/counterbalanced_scheduler.py

| Definition | Purpose / audit detail |
| --- | --- |
| [BalancedSchedule](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:21) | Container / interface; fields: layouts, profile_order, capability_pairs, seed, n_layouts, passes_per_cycle. |
| [BalancedSchedule.n_episodes](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:30) | Helper called by BalancedSchedule; inspect its linked body for arguments and return values. |
| [BalancedSchedule.rounds](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:34) | Helper called by BalancedSchedule; inspect its linked body for arguments and return values. |
| [BalancedSchedule.summary](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:37) | Helper called by BalancedSchedule; inspect its linked body for arguments and return values. |
| [build_schedule](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:50) | Create shuffled, without-replacement layout passes and cut their concatenation into full round packets. Assign each packet once to every profile before moving on. Separate random generators for layouts and profile order let the single and diverse controls share layout prefixes. The gcd calculation chooses enough passes to close a cycle without padding: 1096 layouts and 20 rounds need five passes. |
| [save_schedule](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:81) | Save the numerical plan as NPZ and write a JSON description with its SHA-256 checksum. The checksum identifies exact bytes, not just equivalent array values. The rebuild utility preserves the original archive format for this reason. |
| [load_schedule](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:98) | Helper called by this module; inspect its linked body for arguments and return values. |
| [QueueState](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:104) | Container / interface; fields: next_episode, slot_episode, layout_counts, episode_counts. |
| [initial_queue](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:111) | Helper called by this module; inspect its linked body for arguments and return values. |
| [dispatch](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:118) | Give each finished worker the next unused global episode ID. A cumulative count orders simultaneous completions by worker slot. Workers with unfinished episodes keep their IDs; this avoids overlapping independent cursors and bounds allocated profile imbalance by one. |
| [lookup](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:133) | Translate each global episode ID into a packet index and a position in that packet shuffled profile order. Return the matching fixed capability and shared round layouts. This connects the asynchronous queue to the precomputed balanced design. |
| [accumulate_audit](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:145) | Count actual profile-by-layout exposure from collected transitions: starts when pre-step time is zero, completions when round_done is true, and one environment step for every transition. Count episode starts/completions separately. These empirical counts include unfinished work at the fixed training cutoff. |
| [write_audit](/juice6/u/jshe/emergent_partner_grid/train/counterbalanced_scheduler.py:168) | Persist actual counts together with active worker IDs and their partial-round states. Reconstruct the allocated prefix separately and check total collected steps against the PPO budget. Exact allocation balance does not imply equal completed rounds or equal time exposure, because profiles and policies finish at different speeds. |

### train/episode_scheduler.py

| Definition | Purpose / audit detail |
| --- | --- |
| [EpisodeSchedule](/juice6/u/jshe/emergent_partner_grid/train/episode_scheduler.py:33) | A flat schedule of scheduled partner episodes.  schedule_capability[i]  = the (d_R, d_B) pair for the i-th ep. schedule_layouts[i]     = the (rounds_per_episode,) layout ids. |
| [build_schedule](/juice6/u/jshe/emergent_partner_grid/train/episode_scheduler.py:49) | Materialize one random capability per partner episode and one independently sampled layout per round. Uniform sampling is a distributional promise, not an exact count quota. Storing the array and seed makes the draws reproducible; workers use separate starting cursors into this shared schedule. |
| [initial_episode_cursor](/juice6/u/jshe/emergent_partner_grid/train/episode_scheduler.py:101) | Space worker cursors along the materialized schedule so parallel environments begin at different entries. A worker advances after its own episode ends. This is the historical random sampler; it does not have the unique global dispatch guarantee of the counterbalanced queue. |
| [summarize_schedule](/juice6/u/jshe/emergent_partner_grid/train/episode_scheduler.py:115) | Report observed profile/layout coverage in the random schedule. These counts diagnose finite sampling variation, rather than enforcing exhaustive profile-by- layout balance. |

### train/ippo_rnn_coordination_grid.py

| Definition | Purpose / audit detail |
| --- | --- |
| [ScannedRNN](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:71) | GRU cell that resets its hidden state on each done flag.  Applied via nn.scan over the leading time axis. |
| [ScannedRNN.__call__](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:85) | Process one observation using the previous memory vector. Reset that memory only when the supplied pre-observation reset flag is true. The enclosing scan repeats this cell over time; intermediate physical round resets intentionally retain memory of the same partner. |
| [ScannedRNN.initialize_carry](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:103) | Helper called by ScannedRNN; inspect its linked body for arguments and return values. |
| [CNN](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:110) | The Overcooked-v2 baseline CNN, unchanged. |
| [CNN.__call__](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:117) | Convert a grid image into learned visual features with six convolutions and a final projection. Convolutions apply the same local pattern detector across the grid. This encoder is shared by the recurrent policy and the memory-control policy. |
| [CommObsEncoder](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:142) | Encode the CoordinationGrid dict obs.  grid: (N, H, W, 5) -> CNN -> (N, grid_dim) last_allocation: (N, 3) -> Dense -> (N, msg_dim) concatenate -> Dense(out_dim) -> (N, out_dim)  The final projection to out_dim matters because the downstream ScannedRNN sizes its GRU cell from ins.shape[-1] and re-initializes its carry at that width. If encoder output width ≠ GRU carry width, the reset-on-done jnp.where blows up. So we lock encoder output width to out_dim (== GRU_HIDDEN_DIM at the call site). |
| [CommObsEncoder.__call__](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:162) | Encode the grid and previous allocation separately, concatenate the two feature vectors, and project them to the network width. The allocation is a fixed-meaning goal request. No capability label is provided to the encoder. |
| [ActorCriticCommRNN](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:192) | Single-agent actor-critic with recurrent memory over dict obs. |
| [ActorCriticCommRNN.__call__](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:199) | Encode and normalize observations, update the GRU memory, then produce action probabilities and a value estimate through separate heads. The value estimates future discounted reward for PPO. Mask illegal action logits before constructing probabilities, including the forced initialization action. |
| [ActorCriticCommMLP](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:255) | Feed-forward MLP variant. Same call interface as the RNN model so the trainer can swap them via config: takes (hidden, (obs, dones)) and returns (hidden, pi, value) — but hidden is a dummy that is passed through unchanged. No cross-step memory. |
| [ActorCriticCommMLP.__call__](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:266) | Use the same observation encoder and policy/value heads but replace memory with a dense feedforward transformation. Keep the call signature compatible with the RNN so training and evaluation can share code. Its returned carry is an interface placeholder, not remembered partner experience. |
| [build_network](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:325) | Choose the recurrent or feedforward model from MODEL_TYPE and return a matching memory-initialization function. Conditions change this choice without changing the PPO objective. Reject unknown names rather than silently substituting a model. |
| [build_network.init_hstate](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:344) | Helper called by build_network; inspect its linked body for arguments and return values. |
| [resolve_training_capability_pool](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:355) | Choose either all 24 authoritative training profiles or the configured single partner. This pool trains the policy; standalone evaluation explicitly replaces it with familiar and novel populations. A single-partner policy therefore has less training exposure even when its probe later fits all 46 profiles. |
| [Transition](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:380) | Container / interface; fields: done, reset, action, value, reward, log_prob, obs, info, pre_step_time. |
| [require_current_protocol](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:397) | Guard against evaluating a fixed-v1 checkpoint with online-v2 transition rules. Saved configurations must name the online_v2 protocol for this active trainer. Historical evaluation requires the matching frozen source. |
| [make_train](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:411) | Build a complete training function from the resolved configuration. Count one learned actor per environment because the partner is scripted. Integer division sets the number of full PPO updates; the nominal 60 million transitions become 59,965,440. This active source uses the random scheduler; the batch preparer replaces scheduling in a frozen copy for counterbalanced experiments. |
| [make_train.create_learning_rate_fn](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:454) | Define how the optimizer learning rate changes over gradient updates: first linear warmup, then cosine decay. Multiply PPO update counts by minibatches and epochs to express the schedule in optimizer steps. This keeps the schedule aligned with actual weight updates rather than environment ticks. |
| [make_train._dummy_obs](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:479) | Helper called by make_train; inspect its linked body for arguments and return values. |
| [make_train.train](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:520) | Initialize network weights, optimizer state, parallel environments, per-worker episode cursors, and zero memory. Repeated PPO updates carry these states forward. Random initialization and minibatch ordering follow the learner key; the materialized exposure schedule has its own configured seed. |
| [make_train.train._select_per_slot](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:568) | Per-slot jnp.where(mask, b, a) broadcasting mask_1d (shape (N,)) to whatever leading-N shape a/b have. |
| [make_train.train._update_step](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:576) | Helper called by make_train.train; inspect its linked body for arguments and return values. |
| [make_train.train._update_step._env_step](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:578) | Collect one transition from each parallel environment with current policy weights. Record the memory-reset flag used before this observation separately from the terminal flag produced afterward. Reset only workers that finished their complete partner episode using scheduled capabilities/layouts; leave other workers and memories intact. The stored action, value, log probability, and observation later reconstruct the PPO comparison. |
| [make_train.train._update_step._calculate_gae](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:722) | Compute generalized advantage estimates backwards through the rollout. Advantage measures whether observed reward plus estimated future value exceeded the old value prediction. The post-transition done flag stops value bootstrapping across different partners; it is not the GRU replay reset flag. |
| [make_train.train._update_step._calculate_gae._get_advantages](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:730) | Helper called by make_train.train._update_step._calculate_gae; inspect its linked body for arguments and return values. |
| [make_train.train._update_step._update_epoch](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:754) | Shuffle whole environment sequences into minibatches while retaining their time order. Each sequence starts from its collected memory state. Reordering individual timesteps would destroy the history needed to replay a recurrent policy correctly. |
| [make_train.train._update_step._update_epoch._update_minbatch](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:761) | Helper called by make_train.train._update_step._update_epoch; inspect its linked body for arguments and return values. |
| [make_train.train._update_step._update_epoch._update_minbatch._loss_fn](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:764) | Replay stored observations with their original pre-observation resets and initial memory. Compare new and collected action log probabilities to form the PPO ratio; clip policy and value changes, normalize advantages, and add the entropy term. Only ego data trains these weights. Greedy collection changes how actions were collected, while this probability-based PPO loss remains unchanged. |
| [make_train.train._update_step._log_cb](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:1016) | Print training diagnostics on the host after device computation. Completed-episode returns are the scores later used for behavior-based seed selection. These describe the collection rollout before its associated gradient update, whereas the saved final checkpoint includes that update. |
| [evaluate_policy](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:1079) | Evaluate a trained policy with an explicit capability pool and repeated full partner episodes on familiar layouts. Run a fixed-length scan, then restrict metrics to each episode first final done. Preserve recurrent memory between its rounds; this in-process evaluator returns summaries rather than the standalone HDF5 export. |
| [evaluate_policy.rollout](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:1136) | Helper called by evaluate_policy; inspect its linked body for arguments and return values. |
| [evaluate_policy.rollout.body](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:1137) | Helper called by evaluate_policy.rollout; inspect its linked body for arguments and return values. |
| [main](/juice6/u/jshe/emergent_partner_grid/train/ippo_rnn_coordination_grid.py:1241) | Resolve Hydra configuration, create learner keys, train, save weights and the resolved JSON configuration, then evaluate familiar and novel profiles. Saving configuration next to the checkpoint makes architecture, influence, reward, and action-selection settings auditable later. |

### train/plot_training_curves.py

| Definition | Purpose / audit detail |
| --- | --- |
| [smooth](/juice6/u/jshe/emergent_partner_grid/train/plot_training_curves.py) | Calculate full-width trailing means and preserve missing early episode metrics. A logging placeholder from an update with no completed episodes must not be interpreted as a real zero return. |
| [savefig](/juice6/u/jshe/emergent_partner_grid/train/plot_training_curves.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [main](/juice6/u/jshe/emergent_partner_grid/train/plot_training_curves.py) | Parse training stdout identified by manifest job IDs, align logged update indices with actual environment steps, and plot conditions separately by protocol. Save numerical summaries with source evidence. Learning curves use training observations, not held-out performance. |
| [main.band_start](/juice6/u/jshe/emergent_partner_grid/train/plot_training_curves.py) | Helper called by main; inspect its linked body for arguments and return values. |

### train/rebuild_counterbalanced_schedules.py

| Definition | Purpose / audit detail |
| --- | --- |
| [sha](/juice6/u/jshe/emergent_partner_grid/train/rebuild_counterbalanced_schedules.py:20) | Helper called by this module; inspect its linked body for arguments and return values. |
| [OriginalZipInfo](/juice6/u/jshe/emergent_partner_grid/train/rebuild_counterbalanced_schedules.py:24) | Container / interface; see called methods and the stage map above. |
| [OriginalZipInfo.FileHeader](/juice6/u/jshe/emergent_partner_grid/train/rebuild_counterbalanced_schedules.py:25) | Helper called by OriginalZipInfo; inspect its linked body for arguments and return values. |
| [save_original_archive](/juice6/u/jshe/emergent_partner_grid/train/rebuild_counterbalanced_schedules.py:37) | Write arrays using the historical NPZ archive conventions so regeneration can reproduce exact bytes. Equal numerical arrays alone would not satisfy the manifest file checksum. |
| [rebuild](/juice6/u/jshe/emergent_partner_grid/train/rebuild_counterbalanced_schedules.py:57) | Load the recorded parameters and frozen scheduler/population source, regenerate missing schedule arrays, and require original SHA-256 matches. This restores local ignored arrays for existing experiments without redefining their schedules or resubmitting jobs. |

### train/validate_counterbalanced_training.py

| Definition | Purpose / audit detail |
| --- | --- |
| [check_preserved_implementation](/juice6/u/jshe/emergent_partner_grid/train/validate_counterbalanced_training.py:17) | Compare selected syntax trees in frozen/recovered sources to check that scheduling patches retained the intended protocol and network behavior. Comments do not appear in syntax trees. This guards against changing the scientific condition while patching exposure bookkeeping. |
| [check_preserved_implementation.nodes](/juice6/u/jshe/emergent_partner_grid/train/validate_counterbalanced_training.py:31) | Helper called by check_preserved_implementation; inspect its linked body for arguments and return values. |
| [validate](/juice6/u/jshe/emergent_partner_grid/train/validate_counterbalanced_training.py:51) | Verify sources/schedules, run sampler/protocol regressions and bounded train/evaluation smoke checks, and record preflight results in the manifest. This is preparation validation, not sixty-million-step training or a scientific result. It can write temporary/check artifacts but does not submit Slurm jobs. |

### bash/submit_counterbalanced_training.py

| Definition | Purpose / audit detail |
| --- | --- |
| [relocate_v1_imports](/juice6/u/jshe/emergent_partner_grid/bash/submit_counterbalanced_training.py) | Relocate imports only; retain the original v1 protocol and PPO functions. |
| [sha](/juice6/u/jshe/emergent_partner_grid/bash/submit_counterbalanced_training.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [write](/juice6/u/jshe/emergent_partner_grid/bash/submit_counterbalanced_training.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [replace_once](/juice6/u/jshe/emergent_partner_grid/bash/submit_counterbalanced_training.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [patch_trainer](/juice6/u/jshe/emergent_partner_grid/bash/submit_counterbalanced_training.py) | Transform a recovered trainer source string to use a global counterbalanced queue and empirical exposure audits. Replace scheduling/reset/audit sites using checked anchors, while retaining the original network, loss, and protocol logic. Changes in anchor comments can break this preparer even when Python behavior is unchanged, so keep those anchors intact. |
| [prepare](/juice6/u/jshe/emergent_partner_grid/bash/submit_counterbalanced_training.py) | Create a fresh batch snapshot, verify/copy the shared corpus, and save single/diverse schedules. Recover v1 from its recorded Git revision and copy active v2, then patch both copies with scheduling; training and evaluation retain original categorical action sampling. Write hashes and forty concrete training commands to a manifest; preparation itself does not submit jobs. |
| [verify_manifest](/juice6/u/jshe/emergent_partner_grid/bash/submit_counterbalanced_training.py) | Rehash frozen layouts, schedules, and source files and compare them with the prepared manifest. Resolve relocated paths in memory first. This detects changes to the evidence that a submission claims to use. |
| [submit](/juice6/u/jshe/emergent_partner_grid/bash/submit_counterbalanced_training.py) | Require a passed preflight and intact frozen hashes, then submit only jobs without recorded IDs. Persist each Slurm ID immediately so submission can resume. Queue one evaluation per protocol with afterok dependencies on that protocol twenty training jobs. Executing this function spends cluster resources; reading or documenting it does not launch an experiment. |

### eval/compare_allocation_protocols.py

| Definition | Purpose / audit detail |
| --- | --- |
| [main](/juice6/u/jshe/emergent_partner_grid/eval/compare_allocation_protocols.py) | Gather every requested learner seed from explicit original/extension evaluation directories and summarize familiar or novel population performance separately for fixed v1 and online v2. Plot means and sample standard deviations over learner seeds. Original v1/v2 comparisons also change corpus size; counterbalanced comparisons share a corpus and schedule design. |
| [main.plot_metric](/juice6/u/jshe/emergent_partner_grid/eval/compare_allocation_protocols.py) | Helper called by main; inspect its linked body for arguments and return values. |

### eval/compare_representation_protocols.py

| Definition | Purpose / audit detail |
| --- | --- |
| [main](/juice6/u/jshe/emergent_partner_grid/eval/compare_representation_protocols.py:15) | Compare the written-method and released-code probe outputs for each allocation version without conflating their optimizers, splits, warm starts, or test selection. Keep random-feature baseline labels explicit because the two analysis methods differ in their targets. |

### eval/compare_representation_versions.py

| Definition | Purpose / audit detail |
| --- | --- |
| [main](/juice6/u/jshe/emergent_partner_grid/eval/compare_representation_versions.py:24) | Read separately computed v1/v2 probe tables and compare matched conditions, targets, and history axes. Preserve individual-seed curves alongside aggregate curves and endpoint contrasts. This is descriptive comparison of existing fits, not new probe training or evidence of causal representation use. |

### eval/evaluate_partner_modelling.py

| Definition | Purpose / audit detail |
| --- | --- |
| [per_layout_bfs](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:85) | Precompute four wall-aware distances for historical analytical allocation scoring. Distances reflect static start positions. They cannot be used as an online oracle after the agents move or assignments switch. |
| [analytical_alloc_rewards](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:117) | Compute fixed-assignment rewards for ego-red/partner-blue and its complement over each layout. These helpers remain for historical validation; online-v2 headline metrics use actual decisions and realized outcomes instead. |
| [optimal_allocation_and_regret](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:151) | Compare a proposed fixed allocation with the analytical better alternative and record reward regret plus ties. This is meaningful under the fixed-role model. In a no-influence control the ego request can be hypothetical rather than the actual partner assignment. |
| [rollout_condition](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:184) | Load one policy and run repeated partner episodes with an explicit capability pool. Save post-observation hidden states, the state used for the current action, along with pre-step context and post-step outcomes. The fixed-length scan can continue after final done, so every downstream statistic must use a first-done mask. |
| [rollout_condition.body](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:239) | Advance observation, policy memory, action, and physical state in time order. Pair hidden state with the observation just processed, while round_time is the post-transition time. Logging both requested and actual assignment is necessary when influence is disabled. |
| [_first_done_alive_mask](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:312) | Mark all steps up to and including the first episode-ending transition. Exclude later fixed-scan padding while retaining the terminal success/reward. Removing the terminal tick would bias success, return, and hidden-state features. |
| [summarize](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:330) | Calculate realized returns, success, completion times, and online assignment diagnostics only on valid prefixes. Exclude initialization from learned allocation decisions and exclude equal-delay profiles from faster-goal comparisons. Faster-goal adherence describes speed matching, not an optimal online allocation policy. |
| [summarize.fraction](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:375) | Helper called by summarize; inspect its linked body for arguments and return values. |
| [_save_hdf5](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:417) | Write rollout arrays and configuration/protocol metadata to one HDF5 file. Arrays are evidence for downstream analysis; attributes identify which condition and environment generated them. |
| [main](/juice6/u/jshe/emergent_partner_grid/eval/evaluate_partner_modelling.py:431) | Load saved weights and their resolved configuration, restore influence explicitly, and evaluate familiar and novel profile slices separately. Request hidden export only for recurrent policies. The active implementation rejects a protocol mismatch instead of silently changing historical checkpoints. |

### eval/plot_best_training_seed.py

| Definition | Purpose / audit detail |
| --- | --- |
| [write_csv](/juice6/u/jshe/emergent_partner_grid/eval/plot_best_training_seed.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [select_seeds](/juice6/u/jshe/emergent_partner_grid/eval/plot_best_training_seed.py) | Read complete training logs, validate update counts against saved configurations, and choose the highest training-return learner per condition/protocol. Exclude logged updates without completed-episode metrics from a requested trailing window. Resolve logged score ties deterministically by seed; novel test results never enter selection. |
| [held_out_metrics](/juice6/u/jshe/emergent_partner_grid/eval/plot_best_training_seed.py) | After selecting a policy, compute its novel-population outcomes from valid first-done episode prefixes. Bootstrap whole episodes within each profile, keeping all twenty rounds together and profile weights equal. These intervals describe evaluation sampling for this fixed policy, not variation across trained learner seeds. |
| [save_figure](/juice6/u/jshe/emergent_partner_grid/eval/plot_best_training_seed.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [make_plots](/juice6/u/jshe/emergent_partner_grid/eval/plot_best_training_seed.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [make_plots.metric_plot](/juice6/u/jshe/emergent_partner_grid/eval/plot_best_training_seed.py) | Helper called by make_plots; inspect its linked body for arguments and return values. |
| [load_source](/juice6/u/jshe/emergent_partner_grid/eval/plot_best_training_seed.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [main](/juice6/u/jshe/emergent_partner_grid/eval/plot_best_training_seed.py) | Save training candidates and selections before plotting chosen-policy held-out outcomes. Provenance records make the figure policy choices auditable. |

### eval/plot_common_training_seed.py

| Definition | Purpose / audit detail |
| --- | --- |
| [main](/juice6/u/jshe/emergent_partner_grid/eval/plot_common_training_seed.py) | Score each learner seed by the equal-weight mean training return across the selected experiment/protocol/condition combinations. Choose one seed before reading held-out metrics, then reuse it for every plotted policy. Selection-window options are recorded with the outputs. |

### eval/representation_analysis.py

| Definition | Purpose / audit detail |
| --- | --- |
| [require](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Assertions that remain active even when Python is run with -O. |
| [capability_populations](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [discover_rollout_files](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [episode_valid_length](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Find the first final done and include that transition in the valid length. A file with no final done is incomplete and rejected. All feature averages must use this valid prefix rather than the whole fixed scan. |
| [validate_allocation_protocols](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Read HDF5 protocol attributes and require every input file in an analysis to belong to one allocation design. Historical files without the attribute are classified fixed_v1. Separate output folders protect interpretation of results from different protocols. |
| [prefix_mean_hidden](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Average the hidden vectors from episode start through each requested time cutoff. Clip the cutoff to the valid episode length; later cutoffs of a completed episode reuse its complete valid history. This is one feature vector per episode/cutoff, not independent observations at every tick. |
| [round_prefix_mean_hidden](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Locate all 20 ordered round endings and average the whole episode prefix through each ending. These are cumulative history features, not averages of only the named round. Check the final round endpoint against the first final done. |
| [final50_mean_hidden](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Average at most the last 50 valid states of a partner episode for UMAP. Short episodes contribute all their states. This final-history view is distinct from prefix-mean probe features. |
| [RolloutDataset](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Container / interface; fields: labels, rollout_rep, capability_slice, lengths, timestep_features, round_features, final50_features, episodes, validation. |
| [validate_rollout_dataset](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [load_checkpoint_rollouts](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Validate episode counts, capability constancy, terminal flags, hidden dimensions, and round boundaries while reading HDF5 in small batches. Retain feature means rather than every full trajectory. Order episodes by capability and repetition so one shared split applies consistently across policy seeds. |
| [make_probe_split](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Permute the 20 repetition indices once and reuse 16 for fitting and four for testing within every profile. This holds out episodes, not capability profiles. Repetition indices describe source order rather than twenty globally shared environment seeds. |
| [split_masks](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Translate the saved repetition split into episode masks and verify 16/4 coverage for every profile. A novel policy-test profile is still represented in probe training here; this analysis does not claim novel-profile transfer by the probe. |
| [train_linear_probe](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Fit separate affine hidden-state-to-delay classifiers with independent weights and optimizer moments, even though computation is batched. Train for a fixed 1000 Adam steps on the training mask. Use no test-based checkpoint selection, warm starts across cutoffs, scaling, or regularization in this written-protocol implementation. |
| [distance_aware_accuracy](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Score a predicted delay by 1 minus its absolute error divided by nine. A near miss gets partial credit, so a high score is not a high exact ten-class accuracy. Lower delay labels represent faster partners. |
| [evaluate_probe](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [run_probes](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Fit independent red and blue probes for every timestep/round feature set and save models and scores. Test the same fitted probe on all, familiar, and novel episode subsets. Also shuffle paired delay labels at the last timestep as a diagnostic that breaks their association with real hidden states. |
| [run_random_baseline](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Replace hidden states with independent Normal features while retaining true capability labels and the shared episode split. Fit with identical probe settings. This baseline controls for label frequencies and forgiving distance-aware scoring rather than testing a recurrent policy. |
| [summarize_probes](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Aggregate fitted-probe scores across learner seeds with means, sample standard deviations, and bootstrap intervals. The resampling unit is the trained policy seed; repeated rollout examples are not independent trained models. |
| [summarize_condition_differences](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Compare each control with the diverse influence-enabled RNN using matched nominal learner-seed IDs. Save descriptive paired contrasts and uncertainty. Pairing seed IDs is useful bookkeeping, not proof that every policy received identical experiences. |
| [select_paper_style_seed](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Choose a representative policy using training return evidence associated with its checkpoint, with an explicit familiar-evaluation fallback if complete logs are unavailable. Probe scores and novel evaluation scores do not choose the policy. Save the candidate scores and selection source. |
| [run_umap](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Fit a separate two-dimensional embedding to one network episode-level final-50 features. Use the prescribed neighborhood and reproducibility settings. Coordinates from independently trained networks are separate fitted spaces and should not be pooled or interpreted as causal capability use. |
| [select_recorded_common_seed](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Reuse the behavior-only common-seed selection record only when its experiment provenance matches these rollout files. Reject a record from another batch. This separates selecting a policy from evaluating its representation. |
| [save_figure](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [plotting_style](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [make_probe_figures](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [make_umap_figures](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [write_json](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [write_metadata](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [main](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Discover all three recurrent conditions and five learner seeds, validate protocols, compute feature/probe analyses, and save provenance, figures, and reports. MLPs lack recurrent hidden states and are excluded from this analysis. Reuse existing evaluated trajectories rather than retraining policies. |
| [write_report](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis.py) | Generate the methods and numerical report; qualitative interpretation is added after inspecting the actual figures, never inferred from clustering. |

### eval/representation_analysis_strict.py

| Definition | Purpose / audit detail |
| --- | --- |
| [load_upstream](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py) | Load pinned local upstream probe code and verify its expected source hash. This analysis reproduces that released implementation rather than the primary written-protocol probe. The checked-in reference remains untouched. |
| [RecordedUpstreamProbe](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py) | Observe splits and checks without changing source RNG or fitted models. |
| [RecordedUpstreamProbe.__init__](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py) | Helper called by RecordedUpstreamProbe; inspect its linked body for arguments and return values. |
| [RecordedUpstreamProbe.record_split](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py) | Helper called by RecordedUpstreamProbe; inspect its linked body for arguments and return values. |
| [RecordedUpstreamProbe.record_eval](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py) | Helper called by RecordedUpstreamProbe; inspect its linked body for arguments and return values. |
| [RecordedUpstreamProbe.fit](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py) | Wrap the released fitter to record its split and checkpoint evaluations without changing source RNG behavior or learned weights. The released routine chooses weights by test accuracy; its test score therefore also participates in model selection. |
| [score_subset](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py) | Helper called by this module; inspect its linked body for arguments and return values. |
| [run_sequence](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py) | Run cutoffs sequentially using the released scalar-label-stratified splits and selected-weight warm starts. Record split membership and overlap with earlier fitting examples. Changing splits while carrying weights across cutoffs means a later test example may already have trained earlier weights. |
| [run_baselines](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py) | Reproduce the released Normal-feature/random-label baseline and save a separate true-label supplementary control. Their targets differ, so label them separately in comparison plots. |
| [main](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py) | Reuse evaluated RNN hidden features, run the pinned released fitter for both targets and axes, and write all traces and metadata separately from primary analysis. NumPy seeding is an explicit reproducibility addition. A closer source-code reproduction does not make test-selected scores an independent generalization estimate. |
| [write_report](/juice6/u/jshe/emergent_partner_grid/eval/representation_analysis_strict.py) | Write the strict-method report including checkpoint selection, warm starts, split overlap, and baseline differences. These disclosures explain how its interpretation differs from the independent written-protocol fits. |

### eval/summarize_decoding_task_exposure.py

| Definition | Purpose / audit detail |
| --- | --- |
| [main](/juice6/u/jshe/emergent_partner_grid/eval/summarize_decoding_task_exposure.py:20) | Summarize how much valid interaction occurred by the absolute-time decoding cutoffs. Separate elapsed ticks from completed physical rounds and assignments, because different capabilities/policies finish at different rates. This diagnoses exposure behind probe curves rather than fitting another capability decoder. |

### eval/partner_dynamics/__main__.py

| Definition | Purpose / audit detail |
| --- | --- |
| [main](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/__main__.py:10) | Coordinate the dedicated analysis stages and route frozen-source collection through subprocesses to isolate imports. Keep outputs scoped by batch and allocation protocol. Reuse prior checkpoint evidence; this entry point does not train new policies. |

### eval/partner_dynamics/adapters.py

| Definition | Purpose / audit detail |
| --- | --- |
| [PolicyAdapter](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/adapters.py:6) | Container / interface; see called methods and the stage map above. |
| [PolicyAdapter.__init__](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/adapters.py:7) | Expose the frozen policy encoder, recurrent cell, and action/value readout without converting its weights. Build JAX derivatives separately with respect to memory and encoded input. Exact replay validation is required before interpreting those derivatives. |
| [PolicyAdapter._full](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/adapters.py:21) | Apply the original policy while capturing its normalized encoded input from LayerNorm. Return the new carry, probabilities, value, encoded input, and logits. The memory is after processing the supplied observation. |
| [PolicyAdapter._update](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/adapters.py:31) | Apply only the original GRU cell to previous memory and normalized encoded input. Use the nested frozen checkpoint weights directly. This isolates recurrence from the encoder and action heads. |
| [PolicyAdapter._readout](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/adapters.py:38) | Reproduce actor/critic heads on supplied memory and apply the frozen protocol action mask. A v2 initialization observation forces the action even if memory perturbation changes unmasked logits. Value can still change at that forced tick. |

### eval/partner_dynamics/collect.py

| Definition | Purpose / audit detail |
| --- | --- |
| [load_frozen](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/collect.py:14) | Load the manifest-selected historical trainer and environment in an isolated subprocess context. Check that the imported environment actually comes from that frozen tree. Active online-v2 code cannot substitute for a fixed-v1 checkpoint. |
| [diagnostic_layouts](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/collect.py:33) | Choose physical contexts by coordinate-distance strata before inspecting policy performance. This diversity selection uses Manhattan coordinates rather than BFS distances and does not redefine the corpus or oracle. |
| [run](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/collect.py:47) | Collect short checkpoint histories with detailed inputs, memory, state, and random keys, then verify exact one-step replay. Learn a projector from discovery histories, check derivatives against finite differences, and compare paired carry perturbations on held-out recipients. This routine performs targeted diagnostic evaluation, not policy retraining. Early-history, nuisance-memory, and forced-initialization limitations remain explicit in saved summaries. |
| [run.select](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/collect.py:85) | Helper called by run; inspect its linked body for arguments and return values. |
| [run.body](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/collect.py:91) | Helper called by run; inspect its linked body for arguments and return values. |
| [run.rollout](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/collect.py:103) | Helper called by run; inspect its linked body for arguments and return values. |
| [run.branch](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/collect.py:106) | One physical round, identical recipient state/key across perturbation arms. |
| [run.branch.step](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/collect.py:110) | Helper called by run.branch; inspect its linked body for arguments and return values. |
| [run.save_tree](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/collect.py:142) | Helper called by run; inspect its linked body for arguments and return values. |
| [main](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/collect.py:203) | Helper called by this module; inspect its linked body for arguments and return values. |

### eval/partner_dynamics/data.py

| Definition | Purpose / audit detail |
| --- | --- |
| [sha](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/data.py:18) | Helper called by this module; inspect its linked body for arguments and return values. |
| [write_json](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/data.py:21) | Helper called by this module; inspect its linked body for arguments and return values. |
| [populations](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/data.py:25) | Helper called by this module; inspect its linked body for arguments and return values. |
| [split_repetitions](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/data.py:29) | Partition repetition IDs within profiles into discovery, validation, and test sets. Keep every event from an episode in the same split so nearby hidden states do not leak across fit and test sets. |
| [round_boundaries](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/data.py:41) | Recover ordered round starts and inclusive endings within the first-done prefix. Hidden states are post-observation states, while recorded round time is post-transition; preserve that offset when naming events. |
| [inventory](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/data.py:78) | Identify existing rollout/checkpoint evidence and check protocol, shapes, and provenance before computing geometry. Missing inputs are a collection requirement, not permission to substitute another experimental batch. |
| [stream_events](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/data.py:126) | Read one complete episode at a time and retain raw hidden states at each round start, midpoint, and end. Unlike the standard probe pipeline these are event samples, not cumulative prefix averages. Group all sixty samples of an episode in one split; target-layout exclusions do not remove that layout from all preceding histories. |

### eval/partner_dynamics/dynamics.py

| Definition | Purpose / audit detail |
| --- | --- |
| [evaluate](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/dynamics.py:13) | Predict the next delay-lifted hidden state using both previous lifted memory and its aligned observed input. Score mean squared error against actual next states, and compare with simply keeping the state unchanged. This measures predictive validity on supplied segments rather than environment reward. |
| [run](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/dynamics.py:27) | Fit discovery-only hidden/input projections, split captures by repetition, and choose driven-model rank and delay using validation segments. Compare held-out prediction with persistence, input-omitting, and input-only controls before interpreting regime distances. Fit separate early/late and faster-goal regimes with fixed policy weights. Cross-policy comparisons use the seeded raw-observation input basis; summarize paired carry-intervention outcomes separately from observational dynamics. |
| [run.select](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/dynamics.py:63) | Helper called by run; inspect its linked body for arguments and return values. |

### eval/partner_dynamics/events.py

| Definition | Purpose / audit detail |
| --- | --- |
| [run](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/events.py:13) | Helper called by this module; inspect its linked body for arguments and return values. |

### eval/partner_dynamics/geometry.py

| Definition | Purpose / audit detail |
| --- | --- |
| [savefig](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/geometry.py:13) | Helper called by this module; inspect its linked body for arguments and return values. |
| [analyze](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/geometry.py:16) | Fit PCA and ridge predictors on discovery data, choose penalties on validation episodes, and score test episodes. Compare capability explanations with phase/layout nuisance predictors, transfer across time/populations, and compute cross-validated profile distances. Episode grouping governs uncertainty; pairwise profile distances share profiles and are not independent samples. These geometry measures describe recoverable structure rather than causal use by the policy. |

### eval/partner_dynamics/input_dsa.py

| Definition | Purpose / audit detail |
| --- | --- |
| [BoundaryDMDc](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/input_dsa.py:22) | Container / interface; see called methods and the stage map above. |
| [BoundaryDMDc.compute_svd](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/input_dsa.py:23) | Construct current-state, next-state, and input pairs inside each contiguous trajectory first, then concatenate the valid pairs. Joining raw trajectories before pairing would invent a transition between unrelated rounds/episodes. Use singular-value decompositions of state-plus-input predictors and next states to obtain the bases for driven linear dynamics. |
| [fit](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/input_dsa.py:45) | Lift each trajectory into delay-coordinate states and fit the boundary-safe DMDc model with an explicit rank and regularization. Short segments that cannot supply valid delayed pairs are excluded. Rank/delay selection belongs to validation evidence in dynamics.run, not held-out test scores. |
| [pairs](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/input_dsa.py:61) | Create delayed current/next-state pairs for scoring using the same within-segment boundaries as fitting. Align the control with the transition it drives. Concatenate only completed pairs so episode boundaries never become training examples. |
| [arrays](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/input_dsa.py:76) | Export both reduced-coordinate and full lifted-coordinate state/input operators as NumPy arrays. The reduced operators support similarity comparisons; full operators predict the delay-lifted trajectory. |
| [distance](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/input_dsa.py:85) | Compare two driven systems using the pinned controllability metric and an explicit input basis. Rename its joint/state/control return order to joint/intrinsic/input. Learned encoder coordinates from separate networks cannot be treated as identical inputs without an alignment or a common raw-input basis. |

### eval/partner_dynamics/interventions.py

| Definition | Purpose / audit detail |
| --- | --- |
| [partner_projector](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/interventions.py:5) | Fit delay/orientation prediction directions on supplied discovery histories, convert standardized coefficients to memory coordinates, and orthonormalize with SVD. Return at most three directions plus the discovery mean. Label-predictive directions can also encode nuisance memories, so control perturbations are necessary. |
| [perturb](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/interventions.py:21) | Construct a carry change using whole donor difference, the partner-predictive subspace, attenuation, or matched random/complement directions. Multiply by the prespecified magnitude. Magnitude zero must preserve the recipient state exactly; compare paired behavioral branches with identical physical starts and random keys. |

### eval/partner_dynamics/numeric.py

| Definition | Purpose / audit detail |
| --- | --- |
| [Ridge](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/numeric.py:4) | Container / interface; see called methods and the stage map above. |
| [Ridge.fit](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/numeric.py:5) | Standardize predictors using this fit data only and solve regularized least squares for one or several targets. Save means/scales/offsets for unchanged application to held-out examples. The penalty controls coefficient size; test data must not estimate the scaling. |
| [Ridge.predict](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/numeric.py:17) | Helper called by Ridge; inspect its linked body for arguments and return values. |
| [select_ridge](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/numeric.py:20) | Try the fixed penalty grid using discovery fits and validation prediction error. Return the chosen discovery-fitted model; do not fit or tune on the test mask. |
| [r2](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/numeric.py:32) | Helper called by this module; inspect its linked body for arguments and return values. |
| [grouped_ci](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/numeric.py:36) | Reduce values to group means and bootstrap those groups. For trajectory analyses the group is an episode so correlated event samples stay together. This interval does not automatically measure uncertainty across learner seeds. |
| [cross_distance](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/numeric.py:48) | Multiply differences in profile means from independent splits and weight coordinates by supplied noise precision. Unlike a squared Euclidean distance the resulting cross-validated estimate can be negative. Precision must be estimated from training evidence. |

### eval/partner_dynamics/replay.py

| Definition | Purpose / audit detail |
| --- | --- |
| [run](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/replay.py:12) | Replay recipient and perturbed memories against the same baseline observations at initialization and the next tick. Use the later learned-action probabilities for v2 because its initialization choice is forced; use the initialization choice for v1. Reconstruct the same donor/magnitude/random-control perturbations as capture, verify zero-change identity, and distinguish this fixed-input action response from paired free-running reward outcomes. |
| [main](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/replay.py:77) | Load the recorded targeted-capture configuration and select one frozen protocol for replay. Write replay outcomes beside capture artifacts. This diagnostic uses existing checkpoint weights and collected histories; it does not train another policy. |

### eval/partner_dynamics/report.py

| Definition | Purpose / audit detail |
| --- | --- |
| [build](/juice6/u/jshe/emergent_partner_grid/eval/partner_dynamics/report.py:9) | Assemble existing data, geometry, replay, derivative, and intervention evidence into the analysis report. Distinguish analyses actually supported by emitted artifacts from deferred methods such as full InputDSA fitting. A configured or proposed method alone is not a completed result. |

### jaxmarl/environments/coordination_grid/coordination_grid.py

| Definition | Purpose / audit detail |
| --- | --- |
| [Actions](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:122) | Container / interface; see called methods and the stage map above. |
| [Allocations](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:130) | Ego-goal allocation; the partner takes the complementary goal.  NONE is a reserved historical code, excluded from every policy action. |
| [ego_action_mask](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:166) | Turn the current observation into a yes/no list of allowed actions. At the initialization tick only STAY with the supplied assignment survives; later, each movement can accompany RED or BLUE. The five NONE codes remain forbidden even though the network has 15 outputs. Audit both this mask and step_env: direct environment calls must obey the initialization rule too. |
| [encode_ego](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:214) | (move, alloc) -> flat ego action id in [0, 15). Pure Python; JAX-safe. |
| [decode_ego](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:219) | Flat action -> (move, alloc). JAX-compatible; accepts scalar or array. |
| [State](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:234) | Container / interface; fields: agent_pos, wall_map, red_goal, blue_goal, time, terminal, capability, partner_goal, partner_move_ctr, last_ego_allocation, partner_assignment, layout_idx, round_idx, episode_layout_seq. |
| [_load_layout](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:266) | Read wall cells and the two starts/goals from JSON. Stored coordinates are row, column, while runtime positions are x, y (column, row). This conversion matters: exchanging the conventions silently changes navigation and all distance calculations. |
| [_load_layout.rc_to_xy](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:284) | Helper called by _load_layout; inspect its linked body for arguments and return values. |
| [_bfs_next_actions](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:302) | Work backwards from the goal with breadth-first search, visiting nearer cells before farther cells. Then choose a neighboring cell one distance unit closer. This table scripts the partner; it does not learn a policy or predict ego behavior. Fixed direction order resolves equally short routes reproducibly. |
| [bfs_distance_map](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:353) | Numpy multi-source BFS from goal. Returns (H, W) int distance-to-goal (-1 for wall/unreachable). Used by the pre-PPO validation script to compute per-cell shortest-path lengths. |
| [_sym_position](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:389) | Apply D4 symmetry g to a single (x, y) position on an n×n grid. |
| [_sym_wall](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:413) | Apply D4 symmetry g to a (H, W) wall array (indexed [y, x]). |
| [_augment_layout](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:434) | Helper called by this module; inspect its linked body for arguments and return values. |
| [_resolve_layout_paths](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:456) | Collapse the three ways to specify layouts into a concrete list. |
| [_canonicalize_capability_pairs](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:477) | Return an (K, 2) int32 numpy array of capability *delay* pairs.  Delay semantics: entry d_X >= 0 = number of wait steps between consecutive partner moves while pursuing goal X (d=0 = every step). |
| [CoordinationGrid](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:500) | CoordinationGrid with capability-vector scripted partner.  The env still supports a pool of layouts; per-layout geometry and BFS tables are stacked along a leading K axis and indexed by state.layout_idx. Layouts must all be the same H×W. |
| [CoordinationGrid.__init__](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:508) | Load equally sized layouts and stack them into arrays so parallel simulations can index a layout cheaply. Precompute a red and blue navigation table for every layout. Configure delays, influence, reward, and round count independently: the experiment YAML overrides the smaller constructor defaults. |
| [CoordinationGrid._build_state_for](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:675) | Start one physical round with both agents back at their starting cells and a fresh random assignment. A supplied capability and episode layout sequence are carried forward by callers across rounds. Only physical state and the movement counter reset here; recurrent policy memory belongs to the trainer. |
| [CoordinationGrid.reset](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:725) | Start an entirely new partner episode: choose a capability and all round layouts. The partner keeps this capability through the episode. Training usually uses reset_from_schedule instead, so the scheduler determines these draws. |
| [CoordinationGrid.reset_from_schedule](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:752) | Accept a capability and complete round-layout sequence chosen outside the environment. The random key still chooses the initial assignment. This separation lets two protocols reuse the same scheduled exposure without exposing capability to the policy. |
| [CoordinationGrid.reset_to_layout](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:773) | Deterministic reset to a specific layout — for eval loops that need to visit every val/test layout N times. Also accepts an explicit capability so per-profile eval loops can force it. |
| [CoordinationGrid._partner_next_move](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:792) | goal ∈ {UNSET, RED, BLUE}. Returns int32 move ∈ {0..4}.  UNSET → STAY. This is the *single* point where the BFS tables are consulted; the cooldown mechanic wraps this in step_env. |
| [CoordinationGrid.step_env](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:812) | Apply one simulation tick in order: interpret the ego request, update the actual assignment if influence allows it, compute the delayed BFS move, resolve physical movement, then score success and termination. Both agents stay on tick zero. Intermediate round endings change physical state but keep capability and do not signal final done; final done after the last round triggers the trainer memory reset. Info describes the completed transition, even when the returned state already belongs to the next round. |
| [CoordinationGrid.get_obs](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:1027) | Construct five image channels: walls, red goal, blue goal, ego, partner. Add the previous ego request (or initial default) and an initialization flag. Capability and the internal partner assignment are deliberately absent, so a learned agent must infer delays from behavior rather than read the answer directly. |
| [CoordinationGrid.get_obs.one_hot](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/coordination_grid.py:1045) | Helper called by CoordinationGrid.get_obs; inspect its linked body for arguments and return values. |

### jaxmarl/environments/coordination_grid/capability_populations.py

| Definition | Purpose / audit detail |
| --- | --- |
| [_validate](/juice6/u/jshe/emergent_partner_grid/jaxmarl/environments/coordination_grid/capability_populations.py:60) | Check the population definition when this module loads. Familiar and novel pairs must be disjoint, and the novel scalar delays 0, 5, and 6 must occur in the test population. Lower delay means faster movement; the numerical labels are waiting times, not speeds. |

### jaxmarl/wrappers/baselines.py

| Definition | Purpose / audit detail |
| --- | --- |
| [save_params](/juice6/u/jshe/emergent_partner_grid/jaxmarl/wrappers/baselines.py) | Flatten the JAX/Flax parameter tree and serialize its arrays in safetensors format. This stores learned weights, not environment configuration or training trajectories. The trainer saves resolved JSON configuration separately. |
| [load_params](/juice6/u/jshe/emergent_partner_grid/jaxmarl/wrappers/baselines.py) | Load saved tensor arrays and reconstruct their original nested parameter tree. The evaluator must instantiate the same network architecture from the companion configuration before applying these weights. |

## Existing regression checks by exact test function

Tests check code behavior; they do not independently establish the scientific hypothesis. Each link identifies the assertion-bearing test used for audit. The environment harness also has script-level invocation outside pytest.

**tests/analysis/test_partner_dynamics.py**

[test_terminal_inclusive_and_no_fabricated_rounds](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_partner_dynamics.py:12), [test_episode_splits_are_deterministic_disjoint](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_partner_dynamics.py:22), [test_preprocessing_fits_train_only_and_null_labels_fail](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_partner_dynamics.py:30), [test_crossvalidated_null_distances_are_unbiased](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_partner_dynamics.py:38), [test_zero_intervention_identity_and_norm_matched_controls](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_partner_dynamics.py:43), [test_input_dsa_recovers_driven_operators_without_boundary_transitions](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_partner_dynamics.py:62), [test_post_observation_input_alignment_and_delay_windows](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_partner_dynamics.py:79), [test_original_flax_adapter_derivatives_match_directional_differences](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_partner_dynamics.py:91).

**tests/analysis/test_representation_analysis.py**

[test_allocation_protocols_cannot_be_mixed](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_representation_analysis.py:16), [test_terminal_inclusive_prefix_and_round_means](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_representation_analysis.py:57), [test_malformed_rollouts_fail_loudly](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_representation_analysis.py:74), [test_split_is_seeded_and_never_splits_capability_profiles](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_representation_analysis.py:90), [test_distance_metric_uses_ordered_delays_not_exact_accuracy](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_representation_analysis.py:102), [test_batched_probes_equal_independent_fits_and_decode_linear_signal](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_representation_analysis.py:110), [test_behavior_selection_prefers_final_training_return](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_representation_analysis.py:130), [test_bootstrap_unit_is_five_policy_seeds](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_representation_analysis.py:151), [test_common_seed_selection_is_scoped_to_version_and_batch](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_representation_analysis.py:163), [test_condition_contrasts_preserve_paired_seed_variation](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_representation_analysis.py:189), [test_discovery_excludes_mlp_and_requires_all_fifteen](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_representation_analysis.py:204).

**tests/analysis/test_representation_analysis_strict.py**

[test_recording_preserves_source_weights_predictions_rng_and_best_checkpoint](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_representation_analysis_strict.py:13), [test_warm_start_resets_optimizer_and_split_changes](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_representation_analysis_strict.py:40), [test_distance_score_uses_label_range_and_reports_exact_accuracy_separately](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_representation_analysis_strict.py:61).


**tests/coordination_grid/test_capability_env.py**

[test_train_test_populations](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:101), [test_action_masks](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:119), [test_reset_and_capability_placement](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:136), [test_capability_absent_from_obs_content](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:149), [test_capability_fixed_across_rounds](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:159), [test_capability_changes_on_episode_reset](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:182), [test_exact_cooldown_cadence_symmetric](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:284), [test_exact_cooldown_asymmetric_uses_correct_delay](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:308), [test_d_r_used_only_for_red_d_b_only_for_blue](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:337), [test_delay_zero_is_legal](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:363), [test_negative_delay_rejected](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:370), [test_influence_true_alloc_drives_partner](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:382), [test_influence_false_alloc_keeps_random_default](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:399), [test_influence_false_partner_goal_invariant_to_ego_alloc](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:411), [test_influence_true_partner_goal_flips_with_ego_alloc](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:425), [test_observation_schema_matches_across_influence](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:437), [test_jit_vmap_smoke](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:458), [test_gru_reset_only_at_full_episode_end](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:476), [test_info_round_time_is_post_transition](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:511), [test_analytical_optimal_allocation_hand_crafted](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_capability_env.py:532).

**tests/coordination_grid/test_counterbalanced_scheduler.py**

[test_complete_cycles_have_exact_pair_counts_and_full_without_replacement_passes](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_counterbalanced_scheduler.py:18), [test_every_episode_queue_prefix_is_profile_balanced_and_packets_are_paired](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_counterbalanced_scheduler.py:37), [test_seed_reproducibility_and_layout_prefix_shared_with_single_partner_control](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_counterbalanced_scheduler.py:49), [test_jitted_asynchronous_dispatch_never_reuses_or_skips_an_episode](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_counterbalanced_scheduler.py:60), [test_audit_conserves_steps_and_distinguishes_started_from_completed](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_counterbalanced_scheduler.py:78), [test_serialization_and_capacity](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_counterbalanced_scheduler.py:106), [test_invalid_plans_fail_before_training](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_counterbalanced_scheduler.py:119).

**tests/coordination_grid/test_evaluation_layouts.py**

[test_standalone_eval_uses_saved_layouts_unless_overridden](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_evaluation_layouts.py:15).

**tests/coordination_grid/test_online_allocation.py**

[test_random_initialization_is_visible_reproducible_and_capability_independent](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_online_allocation.py:49), [test_both_stay_at_zero_then_movement_and_assignment_are_joint_choices](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_online_allocation.py:67), [test_repeating_assignment_preserves_cadence_and_switch_uses_destination_delay](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_online_allocation.py:83), [test_no_influence_freezes_default_but_observation_echoes_requests](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_online_allocation.py:106), [test_each_round_resamples_default_and_only_full_episode_terminates_under_jit](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_online_allocation.py:118), [test_network_excludes_none_and_has_no_initial_assignment_choice](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_online_allocation.py:140), [test_legacy_config_is_rejected](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_online_allocation.py:161), [test_unchanged_policy_replays_collection_across_episode_reset](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_online_allocation.py:166), [test_ppo_update_and_evaluation_use_online_decisions_and_finite_losses](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_online_allocation.py:195), [test_summary_distinguishes_requested_and_realized_assignments_and_masks_padding](/juice6/u/jshe/emergent_partner_grid/tests/coordination_grid/test_online_allocation.py:223).

**tests/data_prep/test_build_final_corpus.py**

[test_exact_balance_requires_no_ties_and_both_goals](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_build_final_corpus.py:20), [test_builder_saves_every_exact_survivor_and_resolves_symmetries](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_build_final_corpus.py:41), [test_builder_rejects_removed_stratification_before_generation](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_build_final_corpus.py:94).

**tests/data_prep/test_capability_selection.py**

[test_allocation_balance_rejects_lopsided](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_capability_selection.py:51), [test_allocation_balance_accepts_flipping_layout](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_capability_selection.py:59), [test_observability_rejects_short_partner_paths](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_capability_selection.py:74), [test_delta_reward_identity](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_capability_selection.py:83), [test_fixed_baseline_is_fixed_across_caps](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_capability_selection.py:93), [test_ties_do_not_count_as_red_or_blue](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_capability_selection.py:113), [test_horizon_outlier_rejects_at_or_beyond_3sd](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_capability_selection.py:125), [test_horizon_outlier_reject_is_at_or_beyond](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_capability_selection.py:137), [test_derive_max_steps_covers_worst_case](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_capability_selection.py:145), [test_layout_bfs_distances_matches_expectation](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_capability_selection.py:155), [test_end_to_end_build_small](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_capability_selection.py:167).

**tests/data_prep/test_geometry_goal_dependence.py**

[test_fastest_assignment_uses_the_opposite_partner_goal_delay](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_geometry_goal_dependence.py:15), [test_equal_times_are_ties_not_red_labels](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_geometry_goal_dependence.py:22), [test_ties_are_excluded_from_correlations_and_separately_credited](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_geometry_goal_dependence.py:33), [test_zero_linear_correlation_can_hide_perfect_geometry_prediction](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_geometry_goal_dependence.py:40), [test_per_grid_capability_balance_rules_out_any_geometry_only_classifier](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_geometry_goal_dependence.py:48), [test_authoritative_pools_and_actual_training_corpus_remain_balanced](/juice6/u/jshe/emergent_partner_grid/tests/data_prep/test_geometry_goal_dependence.py:54).

## Validation performed for this edit

- Existing CPU regressions: **94 passed** in 117.78 seconds. Command: `JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/grid_audit_mpl /nlp/scr/jshe/miniconda3/envs/emergent_partner_model/bin/python -m pytest tests/coordination_grid tests/data_prep tests/analysis -q`. The run emitted existing dependency deprecation warnings. The concurrently added `test_partner_dynamics.py` appeared after this run and is not included in that count.
- Added plain-language audit blocks to **162 core functions in 42 Python files**, plus explanatory blocks in both shell launchers. Syntax trees were equal immediately before/after each annotation insertion.
- Final comparison: **40 of 42 annotated Python files** still have identical executable syntax to their annotation-time copies. `eval/partner_dynamics/__main__.py` and `eval/partner_dynamics/dynamics.py` changed independently during the review; those later functional edits are outside this comment-only verification. Their present sources compile, but the 94-test run does not certify their new analysis results.
- In-memory Python compilation passed for all 48 active/support modules examined, without writing bytecode. Both edited shell launchers pass `bash -n`; `git diff --check` passes.
- All local source links and linked definition line numbers in this guide were checked against the working tree when generated. Future edits can shift line numbers; the linked function names remain the lookup key.
- Completed frozen snapshots, pinned reference source, weights, corpora, and historical numerical outputs were not edited by this documentation work. Tracked bytecode regenerated by pytest was restored to its task-start contents.



## Subsequent cleanup: unused stratified layout selection (2026-10-06)

The only active implementation caller of the removed sampler was the builder branch for `--n_final` without a centroid target; the other active caller was its unit test. The retained 1000/2000 source corpora used capped centroid selection, the 1096 subset used separate ego-distance balancing, and the newer 2262 corpus used uncapped exact selection. Their manifests do not record use of the stratified layout sampler.

Removed that alternative branch, its quantile-bin helper, import, dedicated test, and the unused `--stratify_seed` / `--n_bins_per_feature` CLI and new-manifest settings. Capped runs now require an explicit `--centroid_p_opt_red_target`; uncapped exact selection remains the default. Existing corpus manifests, frozen source copies, schedules, and completed training/evaluation artifacts remain unchanged. Scalar-label-stratified **probe** splits are a separate analysis procedure and remain implemented.

Validation: **23 preparation regression tests passed** in 23.61 seconds with `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/grid_audit_mpl /nlp/scr/jshe/miniconda3/envs/emergent_partner_model/bin/python -m pytest tests/data_prep -q`. The run covers both retained selection modes, split uniqueness, symmetry handling, and rejection of removed options before generation. Exact function links were refreshed after deletion shifted definition lines. `git diff --check` passed.

## Original-method extension to ten learner seeds (October 6, 2026)

The active trainer and evaluator use categorical policy sampling. Completed counterbalanced source trees and their hashes remain intact.

`counterbalanced1096_20261007_001529_seeds6to10` adds seeds 6–10 in four conditions
for each of v1 and v2: 40 policies. Each config is copied from that condition's
saved original seed-1 configuration; seeds 1–5 are checked to have identical
hyperparameters. Only `SEED` and `SAVE_PARAMS_PATH` change, after old directory
names are resolved in memory. Each policy retains 60M nominal steps, 915 PPO
updates, 59,965,440 actual steps, original categorical sampling, no runtime D4
augmentation, 1096 frozen layouts and schedule seed 2026. The original v1 PPO
recurrent replay behavior is retained; substituting current v2 code would change
the method and invalidate direct seed aggregation.

| Extension step | Exact implementation | Audit meaning |
|---|---|---|
| sha | [sha](/juice6/u/jshe/emergent_partner_grid/bash/extend_counterbalanced_training.py:34) | Fingerprint file bytes to detect altered evidence. |
| write | [write](/juice6/u/jshe/emergent_partner_grid/bash/extend_counterbalanced_training.py:43) | Atomically save manifest updates so launch can resume without duplicating recorded job IDs. |
| load | [load](/juice6/u/jshe/emergent_partner_grid/bash/extend_counterbalanced_training.py:51) | Read JSON records while resolving historical paths in memory. |
| relocated_config | [relocated_config](/juice6/u/jshe/emergent_partner_grid/bash/extend_counterbalanced_training.py:55) | Read saved original settings and resolve moved paths without rewriting historical files. |
| verify_frozen_base | [verify_frozen_base](/juice6/u/jshe/emergent_partner_grid/bash/extend_counterbalanced_training.py:65) | Hash original sources, layouts and schedules; reject non-categorical or changed experiments. |
| prepare | [prepare](/juice6/u/jshe/emergent_partner_grid/bash/extend_counterbalanced_training.py:96) | Freeze extension infrastructure, construct forty configs and explicitly map every evaluation seed to its original or new directory. No jobs are launched by preparation. |
| verify | [verify](/juice6/u/jshe/emergent_partner_grid/bash/extend_counterbalanced_training.py:170) | Require exactly forty distinct new policies, exact original scientific settings, intact infrastructure and seeds 1–10 for evaluation. |
| environment | [environment](/juice6/u/jshe/emergent_partner_grid/bash/extend_counterbalanced_training.py:210) | Isolate imports to original frozen source; CPU selection is limited to preflight. |
| training_command | [training_command](/juice6/u/jshe/emergent_partner_grid/bash/extend_counterbalanced_training.py:220) | Call the original Hydra trainer with an external resolved YAML config; production uses unchanged original compute settings. |
| validate | [validate](/juice6/u/jshe/emergent_partner_grid/bash/extend_counterbalanced_training.py:255) | Run eight short CPU training checks covering every protocol and condition; require queue audits and built-in evaluation outputs. |
| run_training | [run_training](/juice6/u/jshe/emergent_partner_grid/bash/extend_counterbalanced_training.py:296) | Train one policy with its assigned new seed and original frozen trainer. |
| run_evaluation | [run_evaluation](/juice6/u/jshe/emergent_partner_grid/bash/extend_counterbalanced_training.py:310) | Use original standalone evaluator, seed 12345, 20 episodes for each of 24 familiar and 22 novel profiles; save RNN hidden states. |
| run_aggregation | [run_aggregation](/juice6/u/jshe/emergent_partner_grid/bash/extend_counterbalanced_training.py:340) | Invoke comparison for familiar and novel partners using all ten seeds; each learner seed receives equal weight and sample SD uses ddof=1. |
| submit | [submit](/juice6/u/jshe/emergent_partner_grid/bash/extend_counterbalanced_training.py:357) | Record forty Slurm IDs; queue one evaluation after each protocol's twenty new policies, then aggregate after both evaluations succeed. |

The Slurm entry point is
[bash/run_counterbalanced_seed_extension.sh](/juice6/u/jshe/emergent_partner_grid/bash/run_counterbalanced_seed_extension.sh).
It restores the same conda environment and deterministic GPU kernel settings as
the original batch, then invokes the frozen extension runner. Kernel determinism
does not turn categorical policy sampling into greedy selection.

[eval/compare_allocation_protocols.py](/juice6/u/jshe/emergent_partner_grid/eval/compare_allocation_protocols.py)
accepts `--extension-manifest` and `--population train|test`. It rejects missing
seed files, checks terminal-inclusive rollout lengths and return consistency,
saves 80 per-policy rows for each population, and averages ten rows per
condition/protocol. Means are not selected using a best-performing seed.
The original five-seed default remains available for historical comparisons.

[tests/analysis/test_ten_seed_comparison.py](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_ten_seed_comparison.py)
checks ten-seed means and sample SD across mixed original/new directories for
both populations, verifies padded transitions are excluded and rejects a missing
new seed. Together with scheduler, online-allocation and evaluator-layout
regressions: **26 tests passed**. CPU training preflight and actual submission
status are recorded in the extension manifest.

Submission verified: v1 training jobs **17741713–17741732**, v2 training jobs
**17741733–17741752**, original-method evaluator jobs **17741753** and **17741754**,
and ten-seed aggregation job **17741755**. All 43 jobs were confirmed by Slurm;
training was pending cluster priority and analysis was pending its dependencies
at the recorded scheduler check. Eight short training runs and two standalone
evaluator runs passed; the latter checked both population slices and 128-unit
RNN hidden-state exports. Completed full-training results are not claimed by
these preflight checks. The manifest stores individual IDs, launch commands,
source/config hashes and dependency chains.

### October 7 launcher failure and guarded relaunch

The first forty policies (17741713–17741752) all failed before entering the
trainer: Slurm copied the wrapper into its spool directory, so `BASH_SOURCE`
resolved the Python runner relative to the wrong directory. Dependent evaluation
and aggregation jobs were cancelled. No additional policy weights or ten-seed
results were produced by that attempt. Its full manifest and Slurm accounting
are preserved under the extension snapshot's `failed_launch_attempt_1/`;
the original frozen experiment and initial extension infrastructure remain intact.

The wrapper now receives an absolute Python-runner path as its first argument,
checks that it exists, then forwards the stage arguments. Every production,
evaluation and aggregation submission passes that explicit path. Eight launcher
regression tests cover a copied script, paths containing spaces, invalid relative
paths, all three submitted stages, success dependencies and duplicate avoidance.
The actual copied wrapper also passed local verification using real conda.

GPU launcher checks 17747086 and 17747087 are queued for v1 and v2. Forty
replacement production policies require `afterok` success of both checks before
training starts. Their current IDs are recorded individually in the extension
manifest. Replacement evaluation jobs are 17747145 and 17747146; all-ten
aggregation is 17747147. Scientific source hashes, all forty policy configs,
original layouts/schedules and categorical action sampling are unchanged.
The repaired production infrastructure is frozen in `infrastructure_launch_v3/`.
The queued GPU checks use the separately preserved `infrastructure_launch_v2/`.
At relaunch verification, both checks were pending cluster priority and the
43 production/analysis jobs were pending their success dependencies.

Launcher regression source:
[tests/train/test_seed_extension_launcher.py](/juice6/u/jshe/emergent_partner_grid/tests/train/test_seed_extension_launcher.py).

### GPU check failure and production-dimension diagnostics

Small-batch GPU preflight 17747086 (v1) completed training, checkpoint export,
sampling audit and both built-in population evaluations. Check 17747087 (v2)
failed in cuDNN convolution backward-filter execution on sphinx3. Both checks
used the original GPU runtime settings. Every guarded production job was
cancelled before starting, and no extension-policy results were generated.
The second attempt's manifest/accounting is preserved under
`failed_launch_attempt_2/`; its source/config files are untouched.

The new `production-smoke` runner mode retains original production tensor
sizes: 256 environments, 256 rollout steps, four PPO epochs and 64 minibatches,
with the original 100-step horizon. It shortens only total compute to one PPO
update (65,536 environment transitions) and evaluation to one episode per
capability. Outputs go under `slurm_production_preflight/`, leaving both earlier
CPU and tiny GPU checks intact. This tests whether the GPU error depends on
small-check dimensions; no diagnosis about its cause is claimed before results.
Diagnostic jobs are 17747263 (v1) and 17747264 (v2), using frozen infrastructure
`infrastructure_launch_v4/`. The original frozen trainers and all forty full
production configs remain unchanged. Eleven launcher/configuration regressions
passed, including checks that production-shape diagnostics do not override
rollout, minibatch, epoch or environment-horizon settings.

Third guarded submission: training jobs **17747295–17747334**, evaluator jobs
**17747335/17747336**, and all-ten aggregation **17747337**. Every training job
requires successful completion of **17747263 and 17747264**. The v1 check
completed on sphinx5; its saved config confirms the original rollout/PPO tensor
dimensions, and its sampling audit records 65,536 collected transitions. Both
built-in population evaluations completed. The v2 check was still pending at
this verification. The manifest records runtime package versions and actual
Slurm dependency evidence. The cuDNN failure's root cause remains unconfirmed;
no experiment algorithm or full-training hyperparameter was changed.

### Verified completion of the ten-seed extension

All forty full policies completed, as did the two standalone evaluations and
all-ten aggregation. Slurm reports `COMPLETED`, exit `0:0` for all 43 final jobs.
All forty saved configs confirm the assigned actual learner seed (6–10), one
policy per job and 915 PPO updates; each sampling audit records exactly
59,965,440 collected transitions and allocated profile-episode imbalance ≤1.
Original source, layout and schedule hashes were rechecked. The forty new
summaries cover 480 familiar and 440 novel episodes per policy; all eighty HDF5
population files exist. Each comparison CSV contains 80 rows (2 protocols ×
4 conditions × 10 learner seeds), and every aggregate has `n_seeds=10`.

[Completed results summary](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/RESULTS_SUMMARY.md) records both populations' means, sample
SD, episode returns and steps with links to per-seed CSVs and figures. V2's
single-partner seed 8 (50.05% novel round success) remains in the mean; no
performance-based seed filtering was applied. Completion evidence is stored
in `completion_check` in the extension manifest. Checked 2026-10-07T19:23:21.550724+00:00.

## Per-profile performance grids

`eval/plot_profile_performance.py` reconstructs performance from the saved
rollout scalar fields; it does not retrain policies or rerun evaluation. Twenty
episodes × twenty rounds contribute to each policy/profile. A per-policy average
is computed first, then all ten learner seeds are averaged equally with sample
SD (`ddof=1`). Both protocols and all four conditions remain represented. The
novel output has 22 profile panels; the shared training population has 24.
The single-partner control itself was trained only on `(1, 4)`.

| Step | Exact function | Audit meaning |
|---|---|---|
| profile_metrics | [profile_metrics](/juice6/u/jshe/emergent_partner_grid/eval/plot_profile_performance.py:34) | Read scalar HDF5 fields; match explicit profile indices to RED/BLUE delays; check physical capabilities and complete round indices; include the first terminal transition and exclude scan padding; measure each policy/profile. |
| check_pooled_summary | [check_pooled_summary](/juice6/u/jshe/emergent_partner_grid/eval/plot_profile_performance.py:106) | Verify that separating profiles reconstructs the saved overall success, return, episode/round counts, and success curve for every policy. |
| aggregate_profiles | [aggregate_profiles](/juice6/u/jshe/emergent_partner_grid/eval/plot_profile_performance.py:127) | Require the full protocol × condition × seed × profile grid; reject missing/duplicate rows; compute equally weighted seed means and sample SD for scalar metrics and round curves. |
| profile_grid | [profile_grid](/juice6/u/jshe/emergent_partner_grid/eval/plot_profile_performance.py:162) | Give every ordered RED/BLUE delay pair its own panel with a shared vertical scale. |
| save_figure | [save_figure](/juice6/u/jshe/emergent_partner_grid/eval/plot_profile_performance.py:176) | Save a PNG and vector PDF from the same plotted data. |
| plot_profiles | [plot_profiles](/juice6/u/jshe/emergent_partner_grid/eval/plot_profile_performance.py:183) | Compare v1/v2 means within all four conditions for success, return and episode steps; show SD and individual learner seeds. |
| plot_round_profiles | [plot_round_profiles](/juice6/u/jshe/emergent_partner_grid/eval/plot_profile_performance.py:230) | Draw one per-profile grid for each protocol; show four condition curves through rounds 1–20 with seed SD bands. |
| run | [run](/juice6/u/jshe/emergent_partner_grid/eval/plot_profile_performance.py:266) | Read explicit original/new evaluation directories and frozen population constants; validate before exporting data and figures with provenance. |

[Profile-metric regressions](/juice6/u/jshe/emergent_partner_grid/tests/analysis/test_profile_performance.py)
passed **10 tests**. Checks cover interleaved profile ordering, terminal success
and reward inclusion, arbitrary scan padding, physical-profile disagreement,
missing/repeated rounds or profiles, missing/duplicate seed rows, all-ten means
and sample SD, and reconstruction of pooled summaries. Real saved results passed
all grouping/count/reconstruction checks for both populations. The CSVs contain
1,760 novel and 1,920 familiar policy/profile rows, respectively.

[Novel figures and data](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/novel/per_profile/README.md),
[familiar figures and data](/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/familiar/per_profile/README.md).
Figures were visually inspected for panel labels, shared scales, legends and
readability. Source and exported figure hashes are recorded in the extension
manifest's `per_profile_analysis` entry; original pooled figures remain intact.
