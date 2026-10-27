# Current evaluation and analysis scripts

The root scripts support the five-seed direct-input, t0-fixed v1_matched/v2/v3
cohort: 60 policies, step penalty 0.01. Extended capability and representation
studies reuse these policies; they do not constitute additional training cohorts.
Some files are imported helpers or validators rather than standalone producers.
Immutable producing sources inside retained result/snapshot directories remain
part of their provenance. The root scripts provide maintained entry points.

| File | Role | What it produces or supports |
| --- | --- | --- |
| [__init__.py](__init__.py) | Package marker | Allows imports from eval; no output. |
| [evaluate_partner_modelling.py](evaluate_partner_modelling.py) | Evaluation utilities | Standalone rollout HDF5s and summary JSON; production launchers use the checkpoint-matching frozen copy. |
| [compare_allocation_protocols.py](compare_allocation_protocols.py) | Protocol comparison | All-five-seed familiar/novel summaries, return dots, success and duration plots. |
| [plot_episode_return_dots.py](plot_episode_return_dots.py) | Comparison plotting helper | Protocol/condition return plots from a saved comparison.json. |
| [plot_profile_performance.py](plot_profile_performance.py) | Per-profile performance | Familiar/novel per_profile tables and plots using actual current job tags. |
| [plot_round_return_progression.py](plot_round_return_progression.py) | Round-return progression | Three population views of cumulative/per-round returns; five-seed and pooled-episode variability. |
| [plot_step_distributions.py](plot_step_distributions.py) | Duration diagnostics | Episode/round duration distributions from saved comparison inputs. |
| [plot_overcooked_paper_match.py](plot_overcooked_paper_match.py) | Training-selected returns | Best-of-five training selection, episode-bootstrap intervals and companion plots. |
| [analyze_assignment_behaviors.py](analyze_assignment_behaviors.py) | All-seed behavior | Round/profile assignment and effective-switch tables and figures. |
| [plot_best_seed_assignment_behaviors.py](plot_best_seed_assignment_behaviors.py) | Selected-seed behavior | Assignment diagnostics using verified training-only seed selection. |
| [plot_selected_assignment_dashboard.py](plot_selected_assignment_dashboard.py) | Selected-seed dashboard | Combined return, assignment and switching panels. |
| [representation_analysis.py](representation_analysis.py) | Shared representation utilities | Validated rollout loading, terminal-inclusive state features and episode splits; imported by maintained producers. |
| [relative_speed_representation.py](relative_speed_representation.py) | RNN capability decoding | Faster-red/blue probes, controls, metrics and UMAPs for the original 46 profiles. |
| [run_relative_speed_representation.py](run_relative_speed_representation.py) | Representation runner | Prepared manifests, frozen producer sources and worker execution. |
| [run_relative_speed_pipeline.py](run_relative_speed_pipeline.py) | Representation orchestration | Probe fitting, independent audits and cumulative/round-end state comparison. |
| [round_state_prediction_comparison.py](round_state_prediction_comparison.py) | State-feature comparison | Predicted-class fractions using cumulative means versus individual round-end GRU states. |
| [export_round_state_comparison_pngs.py](export_round_state_comparison_pngs.py) | State-comparison export | Per-seed and combined comparison PNGs from saved scores. |
| [mlp_relative_speed_representation.py](mlp_relative_speed_representation.py) | MLP capability decoding | Replay-validated actor-layer activations, linear probes, controls and plots. |
| [verify_mlp_relative_speed_controls.py](verify_mlp_relative_speed_controls.py) | Independent MLP audit | Recomputed negative-control and aggregate metric checks. |
| [full_capability_evaluation.py](full_capability_evaluation.py) | All-90-profile extension | Reuses the original 46 profiles and evaluates 44 additional profiles with the same frozen policies; representation reports. |
| [plot_capability_category_decoding.py](plot_capability_category_decoding.py) | Category decoding | Four-category summaries and figures from all-90-profile results. |
| [plot_capability_category_returns.py](plot_capability_category_returns.py) | Category performance | Actual 20-round returns grouped by capability category. |
| [analyze_early_decoding.py](analyze_early_decoding.py) | Early prefix decoding | Dense early probe fitting and temporally ordered event diagnostics. |
| [plot_early_decoding.py](plot_early_decoding.py) | Early prefix aggregation | Equal-seed summaries, decoding onset windows and event figures. |
| [analyze_early_goal_exposure.py](analyze_early_goal_exposure.py) | Goal-exposure diagnostics | Goal pursuit excluding initialization STAYs; paired decoding gains without completed-round feedback. |
| [analyze_instantaneous_decoding.py](analyze_instantaneous_decoding.py) | Single-state decoding | Independent faster-goal probes at each saved observation, t=0..400. |
| [plot_instantaneous_decoding.py](plot_instantaneous_decoding.py) | Single-state aggregation | Audited per-profile/category summaries and figures. |
| [analyze_training_capability_decoding.py](analyze_training_capability_decoding.py) | Familiar-to-all90 decoding | Fit on familiar-profile episodes, test independently across all 90 profiles; default excludes single-partner RNNs. |
| [analyze_decoding_behavior.py](analyze_decoding_behavior.py) | Decoding/behavior alignment | Early held-out decoding and behavior summaries on matching episodes. |
| [plot_early_action_probabilities.py](plot_early_action_probabilities.py) | Actor diagnostics | Frozen actor probabilities and recorded rewards over the first 50 observations. |
| [replay_early_partner_motion.py](replay_early_partner_motion.py) | Validated motion replay | Early positions/actions reconstructed against saved states and behavioral fields; accepted prefixes only. |
| [analyze_replayed_motion.py](analyze_replayed_motion.py) | Motion analysis | Motion events, trajectories and summaries from validated reconstructions. |
| [analyze_decoding_behavior_correlations.py](analyze_decoding_behavior_correlations.py) | Descriptive correlations | Within-panel associations, differences, lags and seed-bootstrap uncertainty on matched episodes. |
| [optimal_training_return.py](optimal_training_return.py) | Exact task oracle | Per-layout/profile completion times, corpus ceilings and initial-information bounds for familiar or held-out partners. |
| [plot_training_return_ceiling.py](plot_training_return_ceiling.py) | Oracle comparison | Current five-seed final-training or held-out returns versus validated oracle ceilings. |
| [episode_return_analysis.py](episode_return_analysis.py) | Episode-return extraction | Terminal-inclusive episode/round/switch caches, exact lowest decile and oracle gaps. |
| [episode_return_math.py](episode_return_math.py) | Actor/critic reconstruction | Frozen-network values/probabilities, discounted targets, observation counterfactuals and reference checks. |
| [episode_return_decoding.py](episode_return_decoding.py) | Decision-time decoding | Five round cutoffs and a shuffled-label control for six training-selected RNNs. |
| [episode_return_reports.py](episode_return_reports.py) | Episode-return reports | Return distributions, selected/all-seed tail reports and casebooks, and single-partner generalization. |
| [rerun_episode_return_analyses.py](rerun_episode_return_analyses.py) | Episode-return runner | Staged extraction, mathematical audits, reporting, frozen sources and completion validation. |

## Common regeneration commands

Run from the repository root in the existing representation environment.
Oracle validation additionally needs JAX/Flax and Numba; the existing
`/juice6/u/nlp/climblab/tmp/epg_repr_all10_20261008/bin/python` contains these.
Production policy evaluation uses the training environment and matching frozen
sources, through the maintained launchers.

```bash
python eval/compare_allocation_protocols.py --population train
python eval/compare_allocation_protocols.py --population test
python eval/plot_profile_performance.py
python eval/plot_round_return_progression.py
python train/plot_final_training_return_dots.py
python eval/optimal_training_return.py --population train
python diagnostics/validate_optimal_training_return.py --population train
python eval/plot_training_return_ceiling.py --population train
python eval/optimal_training_return.py --population test
python diagnostics/validate_optimal_training_return.py --population test
python eval/plot_training_return_ceiling.py --population test
```

Comparison/profile/progression results go under
`eval/protocol_comparison/v1_matched_v2_v3_20261022_080155_t0fix_direct_input/all_5_seeds`.
Oracle results go under `eval/{training,heldout}_return_ceiling/v1_matched_v2_v3`.
These tables describe a fully informed task oracle; corpus averages are
population references, not per-episode bounds for sampled layout sequences.

The profile producer accepts `--manifest` with the canonical training manifest
and reads actual job tags. The former comparison wrapper and seed-extension
interface have been removed. Round-return plots use 100 single-partner training
profile episodes, 2,400 familiar-reference episodes or 2,200 held-out episodes
per protocol/condition, across five learners. Their error bars show pooled
episode SD; seed SD is saved separately.

See the [episode-return analysis guide](episode_return_distributions/README.md)
for the four regenerated report families. Representation extension scripts
accept their own analysis manifests; use the retained manifests with
`t0fix_direct_input` in their result paths.
