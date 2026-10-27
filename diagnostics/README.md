# Experiment diagnostics

The approved twelve-policy cohort and the approved implementation protocol are
documented in [the seed debugging plan](/juice6/u/jshe/emergent_partner_grid/diagnostics/seed_debugging_plan.md).

Put active debugging scripts, sanity checks, validation tools, and diagnostic
interventions in this directory. Training and scientific evaluation entry points
remain in `train/`, `eval/`, and `data_prep/`; their diagnostic tools live here.
Regression tests remain in `tests/`.

The exact-oracle validator also needs the oracle's `numba` dependency; it is not
installed in the current training environment. Its CLI help remains available
without that dependency. Moving the scripts does not change these runtime
requirements.

Run commands from the repository root using the existing
`emergent_partner_model` environment. Both `python diagnostics/<script>.py` and
package imports use the repository root. Existing result directories, frozen
experiment sources, source copies saved with results, and historical command
records retain their original paths and bytes. `relocations.json` records the
old and new locations and source hashes. Do not edit archived source copies to
make their historical commands look current.

## Seed learning and behavior

| Script | Purpose |
| --- | --- |
| `diagnose_seed_variability.py` | Audit the 120 training histories, final behavior, and empirical sampling. |
| `select_debugging_seeds.py` | Rank the 90 included final training endpoints and propose a behavior-diverse held-out debugging cohort; excludes RNN without influence. |
| `scan_return_divergence.py` | Scan all 915 logged updates for the twelve targets, locate sustained return gaps, check smoothing sensitivity, and distinguish temporary gaps from differences persisting to the endpoint. |
| `plot_targets_against_best_training.py` | Plot each of the twelve targets against its original best-of-ten training reference, with consistent selection and recomputed onset markers; show best-target/runner-up contrasts separately. |
| `analyze_initial_seed_stimuli.py` | Compare initial scheduled cases and optionally reproduce the first rollout. |
| `diagnose_switch_interventions.py` | Evaluate frozen v2/v3 policies with switching held fixed. |
| `analyze_completed_learning.py` | Validate completed RNN checkpoint ledgers, localize behavior changes, and estimate paired assignment-intervention effects using shared-packet bootstrap intervals. |
| `analyze_failure_mechanisms.py` | Inspect partner-profile failures, switch timing/clocks, early own-policy batches, and replayable cases. |
| `plot_learning_analysis.py` | Plot individual seed pairs with sampled divergence brackets, transfer decay, and profile-specific success. |
| `render_switch_failure.py` | Render a recorded baseline/intervention round, reconstructing terminal movement with success checks. |
| `report_completed_learning.py` | Build the scientific HTML report from the validated analysis outputs. |
| `record_learning_diagnostics.py` | Full-horizon frozen PPO recorder, exact presentation ledger, complete trajectories, checkpoint states, and historical fidelity checks. |
| `evaluate_learning_diagnostics.py` | Fixed-packet checkpoint timelines with complete per-round/step traces and paired assignment interventions. |
| `validate_learning_diagnostics.py` | Pilot artifact checks and exact full-state resume gate. |
| `learning_diagnostics.py` | Shared target provenance, checkpoint schedule, and source loading. |
| `smoke_learning_diagnostics.py` | Small CPU functional fixture, explicitly separate from historical reproduction. |
| `run_learning_diagnostics.sh` / `diagnostic_pipeline_gate.sh` | Scheduler entry points; cap concurrent GPU jobs at submission and gate expansion on validated pilots. |
| `replay_exposure_snapshots.py` | Reproduce v1_matched seeds 1/2 with minimal counter snapshots at updates 12 and 20. |
| `audit_exposure_snapshots.py` | Primary updates-13–20 exposure audit, historical metric checks, and shared-case outcome bounds. |
| `replay_early_exposure.py` | Auxiliary full-trajectory reproduction and shared input constants. Its recorded runs diverged from historical metrics after update 3. |
| `audit_early_window.py` | Build the independent difficulty reference or analyze auxiliary full trajectories. |

The primary exposure audit is in
`train/seed_diagnostics/v1_matched_v2_v3/early_window_13_20/counter_snapshots/report.html`.
The parent directory contains a separately labeled auxiliary replay report; it
does not establish the historical window's per-grid outcomes or encounter order.

```bash
python diagnostics/diagnose_seed_variability.py --help
python diagnostics/replay_exposure_snapshots.py --help
python diagnostics/audit_exposure_snapshots.py
```

The last command regenerates the primary audit from existing counter snapshots
and the saved difficulty reference. It validates historical scalar metrics,
schedule/counter reconstruction, and input hashes. The replay commands perform
training computations: use a fresh output directory and the original frozen
runtime and deterministic GPU settings. Consult saved submission records before
launching them.

Instrumentation can change JAX compilation and therefore training trajectories.
Check a new recorder against the original uninstrumented run before interpreting
its traces as historical evidence. Preserve the full original learning-rate
schedule and distinguish a failed reproduction from a diagnosis of the old run.

## Other sanity checks

| Script | Purpose |
| --- | --- |
| `audit_counterbalanced_results.py` | Reconstruct final empirical exposures from schedules and unfinished worker tails. |
| `validate_counterbalanced_training.py` | Validate frozen sources and run bounded preparation smoke checks. |
| `audit_v1_matched_results.py` | Audit v1_matched outputs and comparisons with historical fixed_allocation. |
| `verify_relative_speed_results.py` | Validate categorical probe artifacts, partitions, and scores. |
| `verify_round_state_comparison.py` | Validate round-end states and reported prediction fractions. |
| `validate_optimal_training_return.py` | Check the task oracle against exhaustive search and frozen environments. |
| `check_geometry_goal_dependence.py` | Audit geometry versus optimal goal across the saved corpus and capability profiles. |
| `capability_validation.py` | Analytical checks of the capability design and reward differences. |

Full recording coverage, field meanings, animation reconstruction, storage, and validation are in [the trace schema](learning_trace_schema.md).
