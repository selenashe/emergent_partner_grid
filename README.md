# CoordinationGrid partner experiments

The maintained experiments are the **five-seed, t0-fixed, direct-input** versions
of **v1_matched**, **v2**, and **v3**. Each protocol has diverse RNN, diverse MLP,
single-partner RNN, and no-influence RNN conditions: 60 policies, including 45 RNNs.
The original ten-seed CNN experiments and their dedicated launchers are retired.
The sliding-observation studies and input-encoder pilot snapshots, manifests,
checkpoints and external artifacts have also been removed. Shared preparation
and smoke-test code remains part of the maintained pipeline.

- **v1_matched:** choose an assignment at round start and retain it for the round;
  PPO replays the GRU with pre-observation episode-reset flags. Its no-influence
  control draws a fresh uniform assignment each round.
- **v2:** random initial assignment followed by allocation decisions during movement.
- **v3:** ego chooses the initial assignment, followed by v2 switching mechanics.

The phase fix adds the round-local `is_t0` input to actor and critic. Direct input
concatenates 245 binary grid values, three allocation values, and the phase bit.
All 249 values enter the 128-dimensional GRU directly. The MLP control uses the
same observations with its feedforward temporal module. Partner capabilities
remain hidden. Memory persists through the 20 rounds with each partner.

## Inputs and integrity

The canonical training manifest is
`train/manifests/sbatch_v1_matched_v2_v3_20261022_080155_t0fix_direct_input.json`.
`train/config/direct_input_integrity.json` seals all 60 scientific configs,
four condition-specific protocol source inventories, the layout corpus hash,
and both schedule hashes. The original config/source transformation checks
passed before this independent contract was created. Verification no longer
requires original CNN manifests, configs, sources, weights, logs, or rollouts.

Existing trained policy sources, saved configs and weights remain unchanged.
Preparation copies the retained, verified configs and protocol sources into a
new batch; only output destinations change. Both v1 influence and corrected
v1 no-influence source variants remain distinct.

The shared scientific inputs contain 1096 layouts, paired counterbalanced
schedules, 24 training profiles and 22 held-out profiles. For compatibility with
immutable saved configs, the shared layouts and schedules retain their paths
under `train/run_snapshots/counterbalanced1096_20261002_235609/{corpus,schedules}`.
Those paths now serve the maintained cohort; the old training results and source
ancestors are not required. Active corpus preparation code is in `data_prep/`.

Large artifacts use `/juice6/scr6/nlp/interp-models/emergent_partner_grid` with
repository-relative paths. `GRID_ARTIFACT_ROOT` selects another absolute root.
`repo_paths.py` resolves saved historical paths and publishes compatibility
links. Removing a repo symlink does not itself remove the scratch artifact.

On a fresh clone, rebuild ignored schedule arrays with the retained manifest:

```bash
python train/rebuild_counterbalanced_schedules.py \
  --manifest train/manifests/sbatch_v1_matched_v2_v3_20261022_080155_t0fix_direct_input.json
```

## Training and evaluation

Use the existing `emergent_partner_model` environment. Preparation submits no jobs:

```bash
python bash/submit_t0fix_direct_input_training.py prepare
python bash/submit_t0fix_direct_input_training.py verify MANIFEST
python bash/submit_t0fix_direct_input_training.py validate MANIFEST
python bash/submit_t0fix_direct_input_training.py submit MANIFEST
```

Each policy retains 915 PPO updates, 59,965,440 actual transitions (60M nominal),
20 rounds per partner episode, step penalty 0.01, and the verified optimizer,
masking, environment and scheduling settings. CPU/GPU smoke runs preserve the
full optimizer schedule while stopping after a short prefix. Full runs have no
prefix limit. Slurm submissions and dependencies are resumable.

Evaluation always uses the checkpoint's matching frozen trainer/environment.
Seed 12345 and 20 episodes per capability yield 480 familiar-reference and 440
held-out episodes per policy. The familiar-reference aggregate for a
single-partner policy includes profiles it was not trained on. Reports and
training curves contain only the direct-input cohort.

## Behavior and representation

The [evaluation script guide](eval/README.md) lists each maintained root script,
its outputs, and commands for regenerating protocol and oracle plots.

The base analysis manifest is
`train/manifests/sbatch_analyses_20261022_172833_t0fix_direct_input.json`.

```bash
python bash/run_t0fix_direct_input_analyses.py prepare TRAINING_MANIFEST
python bash/run_t0fix_direct_input_analyses.py validate ANALYSIS_MANIFEST
python bash/run_t0fix_direct_input_analyses.py submit ANALYSIS_MANIFEST
```

Behavior analyses cover all 60 policies and training-only best-of-five
selections. They measure assignments, faster-task matching, effective switches
and returns, including terminal transitions and excluding padding. The
representation pipeline covers 45 RNNs with the original 46 specialist profiles,
736 fitting/184 test episodes per policy, relative-speed probes, shuffled-label
and random-feature controls, UMAPs, and cumulative-versus-round-end GRU states.
Its source freezer copies an explicit dependency closure rather than every
analysis and historical diagnostic script.

Additional retained evaluation pipelines are:

- [Episode-return diagnostics](eval/episode_return_distributions/README.md):
  distributions, selected/all-seed lowest-decile casebooks and actor/critic audits,
  decision-time capability decoders, and single-partner generalization for the
  current 60 policies. Reproduce with `eval/rerun_episode_return_analyses.py`.
- `eval/full_capability_evaluation.py`: all 90 unequal capability profiles,
  reusing the original 46 and collecting 44 additional profiles for the same
  saved policies; no policy retraining.
- `eval/mlp_relative_speed_representation.py`: MLP activation probes and controls.
- `eval/analyze_training_capability_decoding.py`: familiar-profile fitting with
  evaluation on all 90 profiles; the saved default analysis excludes
  single-partner RNNs and includes 30 policies.
- Early/instantaneous decoding, capability-category returns, action probabilities,
  behavior alignment, and validated motion reconstruction in `eval/`.

**90 capability profiles does not mean 90 learner policies.** Retained result
roots carry `_t0fix_direct_input`; they preserve their producing scientific code,
input mappings, splits, controls, figures and audits.

## Tests

Training/environment/preparation tests use the training environment; analysis
tests use the existing representation environment with
`requirements/representation.txt`. Scientific tests retain recurrent replay,
random no-influence assignment, masks, scheduling, terminal handling, splits,
controls, equal-seed weighting, and independent integrity/tampering checks.

```bash
JAX_PLATFORMS=cpu python -m pytest tests/train tests/coordination_grid tests/data_prep -q
python -m pytest tests/analysis -q
```
