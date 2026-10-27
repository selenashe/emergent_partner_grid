# Maintained five-seed direct-input pipeline

The scientific cohort is defined in `train/experiment.py`; its sealed contract
is `train/config/direct_input_integrity.json`. The three allocation protocols,
four conditions and seeds 1–5 form 60 policies. See the root README for commands.

## Preparation and execution

`bash/submit_t0fix_direct_input_training.py` is the single maintained training
entry point. `train/prepare_experiment.py` verifies the independent contract,
copies the existing four protocol source variants and 60 scientific configs,
changes output locations, and verifies the new batch. It reads no original CNN
experiment. `train/cohort_integrity.py` verifies exact source hashes, config
settings, condition-specific source selection, layout bytes and schedule hashes.

`train/experiment_runtime.py` provides hashing, durable manifests, isolated
frozen-source environments, checkpoint publication and evaluation commands.
`bash/run_counterbalanced_seed_extension.sh` is a generic Conda execution wrapper;
its historical name does not select an old experiment.

The trained implementations live in each retained batch's frozen source variants.
The active trainer supplies the common observation modules, while execution uses
the matching frozen trainer/evaluator for each policy. v1 influence conditions
and the uniform-random no-influence condition have separate source variants.

## Scientific implementation

- `jaxmarl/environments/coordination_grid/`: active environment interfaces,
  online v2/v3 transitions, masks and authoritative specialist populations.
- `train/ippo_rnn_coordination_grid.py`: shared encoder/network definitions and
  trainer template; `train/observation_pipeline.py` installs observation modules
  while preserving protocol-specific actor masks and recurrent replay.
- `train/counterbalanced_scheduler.py`: paired layout packets, asynchronous
  capability queue and empirical exposure audits.
- `train/rebuild_counterbalanced_schedules.py`: exact-checksum restoration from
  the retained frozen sampler and population constants.
- `data_prep/`: generation, analytical capability selection, balancing,
  distribution diagnostics and rendering.

Flat observations contain 7×7×5 grid values, three allocation values and `is_t0`:
249 inputs without convolution, projection or input normalization. RNN memory is
128-dimensional. The MLP condition uses the same direct observations. The phase
bit is distinct from the episode-reset flag. PPO replay uses pre-observation
reset flags; GAE uses post-step termination. Each partner episode has 20 rounds.

## Evaluation and analysis

`eval/evaluate_partner_modelling.py` provides standalone evaluation utilities;
the launcher invokes the corresponding frozen evaluator with explicit saved
config, weights, layout directory and evaluation seed. Behavior and hidden-state
analyses reuse saved rollout HDF5s.

`bash/run_t0fix_direct_input_analyses.py` prepares private canonical-name aliases
and submits behavior, representation and completion stages. The explicit
analysis closure is frozen by `bash/step_penalty_downstream.py`; that historical
filename now provides only the maintained 0.01-penalty, five-seed source freezer.

Behavior includes `plot_overcooked_paper_match.py`, assignment analysis and
selected-seed dashboards. Representation includes the binary relative-speed
producer, shared feature/split helpers, runner, cumulative/round-end comparison,
exports and independent validators. `eval/full_capability_evaluation.py` and its
companions support the retained 90-profile expansion and subsequent analyses.

Saved policies, scientific configs, evaluation rollouts and producing model
sources are preserved. Scientific snapshots and manifests retain the source
and input hashes needed to verify and reproduce each policy.
