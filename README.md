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

```bash
JAX_PLATFORMS=cpu python -m pytest \
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
