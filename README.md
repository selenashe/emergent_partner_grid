# CoordinationGrid partner experiments

For a step-by-step map from the experimental methods to exact code files and
functions, see [the implementation and audit guide](/juice6/u/jshe/emergent_partner_grid/IMPLEMENTATION_GUIDE.md).
Core functions include plain-language audit comments describing their assumptions.

This repository runs the four CoordinationGrid conditions for three allocation designs:

- **v1:** choose an assignment at round start and retain it for the round.
- **v2:** random initial assignment, followed by allocation decisions during movement.
- **v3:** ego chooses RED or BLUE at initialization, followed by v2's allocation and switching rules.

The conditions are diverse-partner RNN + influence, diverse-partner MLP + influence,
single-partner RNN + influence, and diverse-partner RNN without influence. Each
partner episode contains 20 rounds. The diverse training pool has 24 capability
profiles; evaluation includes 22 novel profiles on familiar layouts.

## Repository layout

| Directory | Contents |
| --- | --- |
| `data_prep/` | Grid generation, analytical filtering, distance-distribution plots, balancing, rendering, and the source/current layout corpora with their preparation audits. |
| `train/` | PPO preparation templates, counterbalanced scheduling, preflight validation, training curves, empirical sampling audits, checkpoints, and frozen experiment sources. |
| `eval/` | Checkpoint evaluation, performance comparisons and representation analysis, saved rollouts, summaries, figures, and evaluation Slurm logs. |
| `bash/` | Slurm training/evaluation entry points and batch submission scripts. |
| `jaxmarl/` | Shared JAX interface and the CoordinationGrid environment; capability populations are defined in `jaxmarl/environments/coordination_grid/capability_populations.py`. |
| `tests/` | Environment, sampler, preparation, and analysis tests. |
| `archive/relocations.json` | Compatibility index; obsolete experiment archives have been deleted. |

There is no active `dev/`, `baselines/`, `analysis/`, or `notebooks/` directory.
The relevant former IPPO baseline is now `train/ippo_rnn_coordination_grid.py`.
The unrelated MAPPO/QLearning algorithms and other IPPO configurations were removed.

## Preparation

The provenance chain is:

1. `data_prep/env_generator.py` generates geometrically valid candidate grids.
2. `data_prep/build_final_corpus.py` applies capability-sensitive filters and saves
   layouts, manifests, per-candidate filter audits, and diagnostic plots.
3. `data_prep/plot_layout_distributions.py` examines shortest-path distances and wall density.
4. `data_prep/balance_ego_distances.py` samples a subset with simultaneous red/blue
   ego-distance quotas. The retained 1096 layouts have exactly 137 examples at
   each distance 1–8 for each goal.
5. `data_prep/capability_validation.py` checks analytical feasibility and allocation rewards;
   `data_prep/render_layout_corpus.py` renders saved grids and contact sheets.

The retained source corpus is `data_prep/grids_capability_selected_2000` (selection
before balancing). Counterbalanced v1/v2 both use
`data_prep/grids_capability_selected_balanced_1096`. The newer
`data_prep/grids_capability_selected_exact_balanced_2262` is separate preparation
work and was not used by the completed training batches.
Their `manifest.json` files preserve the original generation parameters and filter
survival counts. Historical paths inside those records are preserved.

New runs of `build_final_corpus.py` retain every survivor with exactly 12 of
24 training capability pairs favoring each ego goal, with no ties. There is no
fixed layout cap unless `--n_final` is supplied. Reachability and geometric
requirements remain part of generation, and deduplication is recorded as
`n_unique_candidates`; there is no separate basic-validity or feasibility
selection stage. Observability, the broad allocation-balance screen, delta
reward, and horizon filtering still precede the final exact-balance screen, so
their threshold distributions use the same pools as before. D4 symmetry counts
differ by at most one when the corpus size is not divisible by eight.

To generate an uncapped corpus with all layouts assigned to training:

```bash
python data_prep/build_final_corpus.py \
  --n_candidates 20000 --train_val_test_ratio 1 0 0 \
  --out_dir data_prep/grids_capability_selected_exact_balanced \
  --skip_render --skip_plots
```

For a capped corpus, supply both `--n_final` and
`--centroid_p_opt_red_target` (typically `0.5`). The unused stratified layout
sampler and its `--stratify_seed` / `--n_bins_per_feature` options were removed.

The existing 2000/1096 corpora and training snapshots retain their original
manifests. The new output directory contains layouts, a compiled NPZ, and full
candidate and stage audits. Use a fresh output directory for each run.

For example:

```bash
python data_prep/plot_layout_distributions.py \
  --corpus_dir data_prep/grids_capability_selected_balanced_1096
python data_prep/render_layout_corpus.py --help
python data_prep/build_final_corpus.py --help
```

`data_prep/check_geometry_goal_dependence.py` exhaustively checks the shared
counterbalanced corpus against all 46 authoritative partner profiles. It
recomputes BFS distances, verifies the frozen training-corpus hash, and reports
start-coordinate correlations, optimal-goal frequencies per grid and capability,
and exact limits on classification using geometry without capability input.

```bash
python data_prep/check_geometry_goal_dependence.py
python -m pytest tests/data_prep/test_geometry_goal_dependence.py -q
```

Outputs are in
`data_prep/grids_capability_selected_balanced_1096/diagnostics/geometry_goal_dependence/`.
All 1096 grids have 23 red-optimal and 23 blue-optimal assignments across the
46 profiles (12/12 on training profiles, 11/11 on held-out profiles), with no
ties. Every starting coordinate has zero correlation with optimal goal, and
any geometry-only classifier has a 50% accuracy ceiling under uniform profile
weighting. On this corpus, the analytical best assignment always lets the
partner take its faster goal. These conclusions use the established static
independent-shortest-path formula; they do not simulate collisions or v2
switching trajectories, and do not assert independence for other capabilities
or different profile frequencies. The existing synthetic selection tests did
not previously verify this entire final corpus; the new regression test does.

## Training and queued evaluation

Use the existing `emergent_partner_model` Conda environment. The Slurm scripts
activate it themselves. The nominal budget remains 60M steps (915 PPO updates,
59,965,440 actual environment transitions).

For counterbalanced v1/v2, preparation freezes both protocols, the common corpus,
and the paired layout schedules. Preparation and validation do not submit jobs:

```bash
python bash/submit_counterbalanced_training.py --prepare
python train/validate_counterbalanced_training.py --manifest train/manifests/<manifest>.json
python bash/submit_counterbalanced_training.py --submit-manifest train/manifests/<manifest>.json
```

The final command submits four conditions × five learner seeds × two protocols
and queues one evaluation job per protocol with `afterok` dependencies. Submission
is resumable from the manifest. Completed manifests are not resubmitted.

The root trainer/config and `train/episode_scheduler.py` are preparation templates:
`submit_counterbalanced_training.py` patches their copied sources to use the paired
counterbalanced scheduler. Direct CLI training rejects this unprepared template.
Launch training from a prepared snapshot. The v1 protocol is recovered from its
recorded Git revision; the v2 template remains in the active tree.

`bash/eval_all_checkpoints.sh` evaluates a checkpoint directory using its matching
frozen source through `REPO_ROOT` and its corpus through `LAYOUTS_DIR`. The batch
submitter supplies both paths. Completed counterbalanced snapshots retain their
original bytes and internal historical directory names for exact reruns.

## Results and logs

- `train/train_logs/`: weights, resolved training configs, built-in evaluation
  summaries, and empirical sampling-audit JSON/NPZ files.
- `train/manifests/`: Slurm submission manifests, job IDs, dependencies, resources,
  source hashes, schedules, and completion records.
- `train/slurm_logs/`: training Slurm stdout/stderr, including PPO learning curves.
- `eval/slurm_logs/`: standalone evaluation Slurm stdout/stderr.
- `train/hydra_outputs/`: Hydra's resolved YAML configs, overrides, and Python
  logger output. This was the former root `outputs/`, separate from Slurm logs.
- `train/run_snapshots/`: immutable sources and frozen input data used by completed
  training batches. For the counterbalanced batch it also contains the shared schedules.
- `train/sampling_audits/`: independent verification and CSVs of actual training exposure.
- `train/training_curves/`: saved learning-curve figures and numerical summaries.
- `eval/eval_out/`: per-policy familiar/novel rollout HDF5s and evaluation summaries.
- `eval/protocol_comparison/`: v1/v2 performance and round-by-round comparisons.
- `eval/representation_results/`: hidden-state decoding and representation analysis.

Generated schedule NPZs are ignored by Git because the single-profile array
exceeds GitHub's file limit. Schedule JSON parameters, checksums, frozen sampler
source, and actual exposure audits remain tracked. On a fresh clone, restore
the arrays with:

```bash
python train/rebuild_counterbalanced_schedules.py \
  --manifest train/manifests/sbatch_counterbalanced1096_20261002_235609.json
```

The script uses the frozen sampler and capability population and requires the
regenerated arrays to match the original SHA-256 checksums.

The completed counterbalanced batch is `counterbalanced1096_20261002_235609`.
Inspect its actual exposure with:

```bash
python train/audit_counterbalanced_results.py counterbalanced1096_20261002_235609
python eval/compare_allocation_protocols.py \
  --counterbalanced-batch counterbalanced1096_20261002_235609
```

To compare one policy per condition, selected by the highest final logged
training episode return among seeds 1–5:

```bash
python eval/plot_best_training_seed.py
```

This uses the counterbalanced batch above and saves plots, all 40 training
selection scores, and the eight selected policies' held-out metrics under
`eval/protocol_comparison/counterbalanced1096_20261002_235609/best_training_seed/`.
Selection uses the final logged PPO rollout, before its last gradient update;
evaluation uses the final checkpoint. Held-out scores do not affect selection.
Error bars resample whole evaluation episodes within each fixed partner profile
and exclude variation across training seeds. A different explicitly specified
trailing selection window is available with `--selection-window-updates`;
use `--output-dir` to keep alternative analyses separate.

To select one common training seed across all eight policies in the retained
five-seed counterbalanced batch:

```bash
python eval/plot_common_training_seed.py
```

The selection maximizes mean final logged training return across four conditions
and both allocation protocols; held-out results do not affect selection. It
selects seed 5. Scores and provenance are saved within the batch's
`common_training_seed/`, with figures under `common_training_seed_5/`.

Representation decoding for the categorical counterbalanced batch is saved under
`eval/representation_results/counterbalanced1096_20261002_235609/`, separately
for `v1_balanced_training` and `v2_balanced_training`. Each version includes
all three RNN conditions and all five seeds. The analysis reuses saved evaluation
hidden states, fits separate red/blue capability probes, and saves UMAPs for
every network. All-seed summaries retain weak policies; the additional
selected-policy view reuses common seed 5 from the behavior-only selection above.

```bash
for version in v1 v2; do
  python eval/representation_analysis.py \
    --eval-dir eval/eval_out/${version}_balanced_training/counterbalanced1096_20261002_235609 \
    --out-dir eval/representation_results/counterbalanced1096_20261002_235609/${version}_balanced_training \
    --layout-count 1096 --analysis-seed 0 --device cpu --threads 1 \
    --common-seed-record eval/protocol_comparison/counterbalanced1096_20261002_235609/common_training_seed/selection.json \
    --umap-all-seeds
done
python eval/compare_representation_versions.py
```

Install `requirements/representation.txt` in an analysis environment first.
Version reports record software versions, input provenance, split and fitted
probe settings. The `comparison/` directory contains v1/v2 mean curves,
individual-seed curves, endpoint tables and descriptive version contrasts.
Probe testing holds out episodes within each of the 46 capability profiles;
it does not hold out entire capability profiles from probe fitting.

A separately labeled released-code analysis lives under
`eval/representation_results/counterbalanced1096_20261002_235609_strict_replication_of_covercooked/`.
It loads the pinned upstream JAX/Flax probe functions in `eval/reference_code/`
and matches AdamW decay, warm starts, scalar-label-stratified splits, and
best-test-checkpoint selection. It includes all five RNN seeds for both versions
and retains the preceding results. NumPy seeding is an explicit reproducibility
addition; the reference script leaves that generator unseeded.
`requirements/representation_strict.txt` specifies a CPU analysis environment;
the exact versions used are also recorded with the result artifacts.

```bash
for version in v1 v2; do
  JAX_PLATFORMS=cpu python eval/representation_analysis_strict.py \
    --eval-dir eval/eval_out/${version}_balanced_training/counterbalanced1096_20261002_235609 \
    --out-dir eval/representation_results/counterbalanced1096_20261002_235609_strict_replication_of_covercooked/${version}_balanced_training
done
python eval/compare_representation_versions.py \
  --results-root eval/representation_results/counterbalanced1096_20261002_235609_strict_replication_of_covercooked
python eval/compare_representation_protocols.py
```

Strict reports save every fit's split and checkpoint-selection trace. The
released random-feature/random-label baseline is reproduced, with a separate
true-label baseline as a supplementary control. Round-index curves and summaries
across all five policy seeds extend the original routine to our grid experiment.
UMAPs reuse the preceding analysis because the RNN states did not change.
Test-score selection and changing splits while warm-starting can introduce
cross-cutoff training/test overlap; the strict reports record that overlap.

Obsolete random-sampling runs, greedy-action experiments, their derivations,
and unrelated games were deleted on 2026-10-07. Completed counterbalanced
snapshots retain inherited game modules and configs because reruns import them
and their manifests verify every recorded source hash. Their dependencies remain
in `requirements/requirements.txt` for exact reruns. The active package
registers only CoordinationGrid. The scientific training/evaluation settings
and completed counterbalanced checkpoints remain unchanged.

## Tests

The active test folders are `tests/coordination_grid/` (environment, training
and evaluation behavior), `tests/data_prep/` (layout-selection calculations),
`tests/train/` (launchers and the counterbalanced entry point), and
`tests/analysis/` (representation-analysis validation). These import and
verify the implementation in `jaxmarl/`, `train/`, `data_prep/`, and `eval/`;
they do not contain duplicate experiment implementations or generated results.
Inherited tests for unrelated JaxMARL tasks have been removed.

```bash
JAX_PLATFORMS=cpu python -m pytest tests -q
python tests/coordination_grid/test_capability_env.py
```

Cleanup verification on 2026-10-07 passed 119 tests, the standalone environment
checks, and eight short training/evaluation checks from a freshly prepared
counterbalanced batch. Recorded hashes for 101 retained checkpoints/manifests
were unchanged, and both batches passed full frozen-source and input checks.

The former stage A/B/C/D and generator-sensitivity `run_sweep.py` pipeline was
retired before the final experiments. Those stages are not required to prepare,
train, or evaluate the current v1/v2 designs.

## Additional counterbalanced seeds

`counterbalanced1096_20261007_001529_seeds6to10` extends the base
`counterbalanced1096_20261002_235609` batch with learner seeds 6–10 in all four
conditions and both allocation protocols (40 additional policies). It reuses
that batch's exact frozen trainer, environment, evaluator, layouts, schedules
and resolved hyperparameters. Both training and evaluation retain categorical
action sampling.

The launcher is `bash/extend_counterbalanced_training.py`; its `prepare`,
`validate` and `submit MANIFEST` stages create the configs, check eight short
training runs, and launch the jobs. Each protocol's evaluation waits for its
20 new policies. A dependent comparison job combines the base five and additional
five seeds with equal seed weights, reporting means and sample SD for familiar
and novel partners separately. All ten seeds are required; no seed is selected
by performance.

The manifest is
`train/manifests/sbatch_counterbalanced1096_20261007_001529_seeds6to10.json`.
Combined results are saved in
`eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/{familiar,novel}/`.
Base five-seed results remain intact. The implementation and validation
mapping is documented in `IMPLEMENTATION_GUIDE.md`.

All 40 additional policies completed successfully, with actual learner seeds
6–10 and 59,965,440 transitions (915 PPO updates) each. Both evaluations and
ten-seed aggregation also completed. All-ten results retain categorical
sampling and equal seed weights, including poorly performing seeds.

Combined familiar/novel results and figures are in
`eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/`.
See `RESULTS_SUMMARY.md` there for means, sample SD, episode returns and steps.
Novel-partner diverse-RNN success is 97.77 ± 1.56% for v1 and 95.96 ± 4.56%
for v2; episode returns are 15.07 ± 1.14 and 13.07 ± 2.48, respectively.
V2 single-partner RNN performance varies considerably across seeds (89.35 ±
14.92% novel success). The manifest records completion of all 43 jobs; earlier
failed launcher/check attempts remain preserved in the snapshots and logs.

Per-profile performance grids are also available in each population's
`per_profile/` subfolder: success, episode return, episode steps, and separate
v1/v2 success curves across rounds. There is one subplot per profile (22 novel,
24 in the shared training population), using all ten learner seeds. PNG/PDF
figures, metrics, and an index are saved together. Regenerate them with
`python eval/plot_profile_performance.py --extension-manifest PATH`.

## V3 counterbalanced runs

V3 lets the ego choose STAY + RED or STAY + BLUE at every round's t=0.
Both agents stay on that transition. The choice is an initial assignment,
leaving the partner cooldown at zero, so retaining that assignment permits
its first move at t=1. Later goal changes retain v2's destination-delay cost.
The no-influence control keeps v2's uniform random initial assignment,
forced initialization action, and ignored allocation requests.

Before an influence-enabled v3 initialization, the previous-allocation channel
is NONE and the partner goal is unset. After initialization, observations,
movement, reward, collision handling, and round termination match v2.
`ALLOCATION_PROTOCOL=online_v3` selects v3 in the active environment/trainer;
the default remains `online_v2` for existing preparation workflows.

`bash/submit_v3_counterbalanced_training.py` prepares 40 policies: four
conditions, learner seeds 1–10, 60 million nominal transitions per policy
(59,965,440 actual transitions; 915 PPO updates). Its frozen sources copy
the completed v2 trainer and evaluator byte for byte and change only the
environment's initialization rule and protocol tag. Layouts, counterbalanced
schedules, resolved condition settings, architecture and PPO are reused.

```bash
python bash/submit_v3_counterbalanced_training.py prepare
python bash/submit_v3_counterbalanced_training.py validate MANIFEST
python bash/submit_v3_counterbalanced_training.py submit MANIFEST
```

The prepared batch is `counterbalanced1096_20261008_035848_v3`, recorded in
`train/manifests/sbatch_counterbalanced1096_20261008_035848_v3.json`.
Submission first queues two GPU checks using the production tensor dimensions,
then 40 dependent training jobs, four evaluation jobs (one per condition), and
a comparison/plot job. Every Slurm ID is saved immediately; repeating submission
skips already-recorded jobs.

Evaluation retains categorical sampling, seed 12345, 20 episodes per profile,
24 familiar profiles, 22 novel profiles, and RNN hidden-state export. The final
job combines v3 seeds 1–10 with the existing v1/v2 seeds 1–10. Pooled success,
returns, episode steps, round curves, and per-profile PNG/PDF plots include all
three protocols, using equal learner-seed weights and sample SD. Results will
be written under
`eval/protocol_comparison/counterbalanced1096_20261008_035848_v3/all_10_seeds/{familiar,novel}/`;
per-profile plots are in each population's `per_profile/` folder. Plotting
requires all requested results and fails on missing seeds.
