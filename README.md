# CoordinationGrid partner experiments

This repository runs the four CoordinationGrid conditions for two allocation designs:

- **v1:** choose an assignment at round start and retain it for the round.
- **v2:** random initial assignment, followed by allocation decisions during movement.

The conditions are diverse-partner RNN + influence, diverse-partner MLP + influence,
single-partner RNN + influence, and diverse-partner RNN without influence. Each
partner episode contains 20 rounds. The diverse training pool has 24 capability
profiles; evaluation includes 22 novel profiles on familiar layouts.

## Repository layout

| Directory | Contents |
| --- | --- |
| `data_prep/` | Grid generation, analytical filtering, distance-distribution plots, balancing, rendering, and the 1000/2000/1096-layout corpora with their preparation audits. |
| `train/` | PPO trainer and config, random and counterbalanced samplers, preflight validation, training curves, empirical sampling audits, checkpoints, and frozen experiment sources. |
| `eval/` | Checkpoint evaluation, performance comparisons and representation analysis, saved rollouts, summaries, figures, and evaluation Slurm logs. |
| `bash/` | Slurm training/evaluation entry points and batch submission scripts. |
| `jaxmarl/` | JaxMARL framework and the CoordinationGrid environment; capability populations are defined in `jaxmarl/environments/coordination_grid/capability_populations.py`. |
| `tests/` | Environment, sampler, preparation, and analysis tests. |
| `archive/` | Previously retired development artifacts, outside the active experiment pipeline. |

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

The retained corpora are `data_prep/grids_capability_selected` (original v1, 1000),
`data_prep/grids_capability_selected_2000` (selection before balancing), and
`data_prep/grids_capability_selected_balanced_1096` (v2 and counterbalanced v1/v2).
Their `manifest.json` files preserve the original generation parameters and filter
survival counts. Historical paths inside those records are preserved.

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

`bash/submit_balanced_training.py` retains the original v2 batch workflow: the
**layout corpus** is balanced, while profiles/layouts are sampled randomly by
`train/episode_scheduler.py`. `train/counterbalanced_scheduler.py` supplies the
new paired profile/layout allocation. Neither is a hyperparameter sweep.

For an individual current v2 policy:

```bash
CONDITION=rnn_diverse_influence SEED=1 sbatch bash/train_final_experiment.sh
```

`bash/eval_all_checkpoints.sh` evaluates a checkpoint directory. For v1, use its
frozen source through `REPO_ROOT` and supply its actual relocated layouts with
`LAYOUTS_DIR`; the batch submitter sets these explicitly. Original completed
snapshots remain unchanged, including their internal historical directory names.

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

For the original versions that sampled profiles and layouts randomly:

```bash
python eval/plot_best_training_seed.py --original-versions
```

These plots and selection records live under
`eval/protocol_comparison/balanced1096_20261002_022924/best_training_seed/`.
Each analysis includes a combined comparison and separate v1/v2 performance
plots in PNG and PDF. Original v1 used 1000 layouts and original v2 used 1096;
the figure labels retain this distinction.

To use one common training seed for every condition in all four completed
original/counterbalanced v1/v2 experiments:

```bash
python eval/plot_common_training_seed.py
```

The common seed maximizes the equal-weight mean final logged training return
across the 16 condition/version/experiment combinations. Held-out results do
not affect selection. For the completed pre-greedy runs, this selects seed 5.
Plots are saved under each batch's `common_training_seed_5/` directory, with
joint selection scores and provenance under
`eval/protocol_comparison/common_training_seed/`. The earlier per-condition
best-seed plots remain separate.

Representation decoding for the non-greedy counterbalanced batch is saved under
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
    --common-seed-record eval/protocol_comparison/common_training_seed/selection.json \
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
Original-v1 representation results remain in `eval/representation_results/`.

The archived batch
`counterbalanced1096_20261004_212351_greedy_action_selection` uses the same
1096-layout corpus, counterbalanced schedules, four conditions, five learner
seeds per version, and 60M nominal environment steps. It has separate frozen
sources, checkpoints, logs, and evaluations. All of its artifacts now live in
`archive/2026-10-05-greedy-action-selection/`, with the original `train/` and
`eval/` directory structure preserved. The archive includes all 42 jobs' Slurm
logs, Hydra outputs, preflight checks, schedules, sampling audits and plots.
`manifest.json` records each move and verifies that every file was preserved;
`archive/relocations.json` lets analysis readers resolve historical paths.
Its manifest is
`archive/2026-10-05-greedy-action-selection/train/manifests/sbatch_counterbalanced1096_20261004_212351_greedy_action_selection.json`.

`ACTION_SELECTION=greedy_random_ties` chooses the action with the highest
policy probability in training and evaluation. Randomness affects action choice
only when multiple actions have exactly equal maximum probabilities; there is
no numerical tolerance. Learner seeds still control initial weights, PPO
minibatch shuffling, and v2's random round-start assignment. Schedule randomness
uses its separate fixed seed of 2026. The PPO loss and entropy coefficient are
unchanged; greedy collection is a deliberate departure from PPO's usual
sampling of actions from the policy distribution. Older saved configs without
this setting retain categorical action sampling when evaluated.

To prepare another batch with this behavior:

```bash
python bash/submit_counterbalanced_training.py --prepare \
  --action-selection greedy_random_ties
```

Preparation prints the new manifest path. Validate it with
`train/validate_counterbalanced_training.py --manifest PATH`, then submit with
`bash/submit_counterbalanced_training.py --submit-manifest PATH`; submission
queues an evaluation job for each version with dependencies on its 20 training
jobs. Use `--action-selection categorical` to prepare the original stochastic
collection procedure instead.

The greedy batch completed all 40 training jobs and both queued evaluations.
Each policy collected 59,965,440 environment steps (915 PPO updates); empirical
sampling audits match the allocated schedules for every run. Regenerate its
evaluation and training figures with:

```bash
python eval/compare_allocation_protocols.py \
  --counterbalanced-batch counterbalanced1096_20261004_212351_greedy_action_selection
python eval/plot_common_training_seed.py --batch-only --selection-window-updates 46 \
  --counterbalanced-manifest train/manifests/sbatch_counterbalanced1096_20261004_212351_greedy_action_selection.json
python train/plot_training_curves.py \
  --manifest train/manifests/sbatch_counterbalanced1096_20261004_212351_greedy_action_selection.json
```

The common-seed comparison selects seed 5 across all eight policies by the
equal-weight average training return over the final 46 updates (3.01M steps).
Updates with no completed episodes have placeholder returns and are excluded;
held-out outcomes do not enter selection. Its figures and selection records
are in `archive/2026-10-05-greedy-action-selection/eval/protocol_comparison/counterbalanced1096_20261004_212351_greedy_action_selection/common_training_seed_5/`.
The parent directory contains five-seed means and sample SDs; learning curves
are in `archive/2026-10-05-greedy-action-selection/train/training_curves/counterbalanced1096_20261004_212351_greedy_action_selection/`.

The audit matrices count rounds started, rounds completed, and environment steps
for every profile × layout pair. Allocation is balanced within one episode;
unfinished episodes at the fixed-step cutoff cause small actual-round differences.
Environment steps differ with partner speed. Full training observation/action
trajectories were not saved.

`repo_paths.py` resolves old paths from immutable records to their current
locations. It does not rewrite original configs, source snapshots, rollout
attributes, or historical submission commands. `train/repo_reorganization/manifest.json`
records the moves and removals. `project_log_revised.md` retains the detailed
experiment history.

## Tests

The active test folders are `tests/coordination_grid/` (environment, training
and evaluation behavior), `tests/data_prep/` (layout-selection calculations),
and `tests/analysis/` (representation-analysis validation). These import and
verify the implementation in `jaxmarl/`, `train/`, `data_prep/`, and `eval/`;
they do not contain duplicate experiment implementations or generated results.
Inherited tests for other JaxMARL tasks, including the MPE-only API test, are
preserved in `archive/2026-10-05-unrelated-tests/` with SHA-256 verification.

```bash
JAX_PLATFORMS=cpu python -m pytest \
  tests/coordination_grid/test_action_selection.py \
  tests/coordination_grid/test_counterbalanced_scheduler.py \
  tests/coordination_grid/test_evaluation_layouts.py \
  tests/coordination_grid/test_online_allocation.py \
  tests/data_prep/test_capability_selection.py \
  tests/analysis/test_representation_analysis.py
python tests/coordination_grid/test_capability_env.py
```

The former stage A/B/C/D and generator-sensitivity `run_sweep.py` pipeline was
retired before the final experiments. Those stages are not required to prepare,
train, or evaluate the current v1/v2 designs.
