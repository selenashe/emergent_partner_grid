# CoordinationGrid — Online task allocation (protocol `online_v2`)

Last design change: **2026-10-01** (random initialization and within-round allocation).
Last documentation update: **2026-10-03** (counterbalanced results and repository
reorganization verified).

## Repository paths after the 2026-10-03 reorganization

The active workflow is now grouped into root-level `data_prep/`, `train/`,
`eval/`, and `bash/`. Grid generation/filtering/balancing/rendering scripts,
the original 1000-layout corpus, its 2000-layout successor, and the balanced
1096-layout corpus are in `data_prep/`. The trainer, config, both samplers,
preflight validation, checkpoints, learning curves, and sampling audits
are in `train/`. Evaluation code, rollout HDF5s, performance figures, and
representation analysis are in `eval/`.

The root `outputs/` used to hold Hydra's resolved YAML configs, overrides,
and Python logger output; it is now `train/hydra_outputs/`. Root `logs/`
used to combine Slurm stdout/stderr with launch manifests; those records
are now split between `train/slurm_logs/`, `eval/slurm_logs/`, and
`train/manifests/`. Saved weights and resolved training configs are in
`train/train_logs/`, and saved evaluation rollouts/summaries are in
`eval/eval_out/`. Frozen completed sources moved to `train/run_snapshots/`
without rewriting their contents or source hashes. All 10,133 moved files
retained their inode and byte size.

The current trainer/config/samplers were moved out of `baselines/`; 57
individually verified unrelated baseline files and tests were deleted.
The empty root `notebooks/`, `dev/`, `analysis/`, and `baselines/` directories
were removed. The README and Makefile now describe CoordinationGrid.
The retired `run_sweep.py` was a generator-parameter sensitivity sweep;
stage A/B/C/D scripts were earlier development artifacts. Neither is part
of the current full experiment. The 37 specifically identified archived
stage/sweep scripts, weights, diagnostics, and logs were subsequently
deleted; other historical archive contents were retained. The original
v1/v2 random sampler is still needed and was
renamed from `sweep_scheduler.py` to `train/episode_scheduler.py`; the new
balanced sampler is `train/counterbalanced_scheduler.py`.

Historical experiment records and the entries below retain their original
paths. `repo_paths.py` resolves those paths in memory to current locations.
Moves/removals are recorded in `train/repo_reorganization/manifest.json`.
The independent 40-run exposure audit now writes to
`train/sampling_audits/counterbalanced1096_20261002_235609/`.
Validation passed 49 regression tests, the full environment test harness,
all eight version/condition training smoke runs, standalone evaluation
with hidden-state/HDF5 export for both versions, and regeneration of the
learning curves and held-out comparison. The original random sampler's
arrays and initial cursors matched the frozen v2 implementation exactly.
Checks are recorded in `train/repo_reorganization/validation.json`. No
Slurm jobs were submitted during the reorganization; the temporary
validation source/schedule batch was removed after its checks completed.

## Current status

- **Latest counterbalanced v1/v2 batch:** all 40 training jobs and both queued
  evaluations completed successfully. Results are in
  `eval/protocol_comparison/counterbalanced1096_20261002_235609/`; exact
  empirical exposure verification is in
  `train/sampling_audits/counterbalanced1096_20261002_235609/`. The older
  balanced-layout batch below retains its original random profile sampler.
- The active design is **online_v2**: random assignment at round reset,
  both agents stationary at t=0, and movement/allocation decisions at
  every t>=1. NONE is never a legal policy action.
- **Current balanced-layout batch (2026-10-01, 19:31 PDT):** training jobs
  **17688017–17688036** use 1,096 layouts with exactly 137 examples at each
  ego distance 1–8 for both goals. Four conditions × seeds 1–5, 60M nominal
  timesteps per policy. **All 20 training jobs and evaluation job 17688037
  completed successfully (exit 0:0).** The last training job ended at
  05:33:58 PDT on October 2; evaluation ran automatically from 05:34:30
  to 06:16:14 PDT. Completion verification checked 20 checkpoint/config/
  summary sets and 40 HDF5 files, all using `online_v2` and the frozen
  1,096-layout corpus. Code and layouts are copied into
  a batch snapshot so later workspace edits do not alter these jobs.
  See [the balanced launch manifest](logs/sbatch_online_v2_balanced1096_20261002_022924.json).
- Capability stays fixed across 20 rounds and recurrent memory persists
  across intermediate round boundaries. Only the ego controls assignments;
  the partner executes BFS navigation with goal-specific delays.
- The previously completed 20 policies and representation results belong
  to the **historical fixed-allocation design**. They have not been
  retrained or re-evaluated and are not results for online_v2.
- The previous online_v2 batch was submitted on **2026-10-01 at 16:54 PDT**:
  four conditions × five seeds, jobs **17686978–17686997**. At submission
  verification all were pending scheduler priority; no new results are
  available yet. Evaluation job **17686998** waits for all 20 jobs to succeed.
- **Cancellation update (2026-10-01):** at the user's request, all queued
  jobs with names starting with `cg` were canceled: training jobs
  **17686979–17686997** and evaluation job **17686998** (20 jobs total).
  Job **17686978** was already running and was preserved. Verification
  found no remaining queued `cg` jobs; the evaluation dependency will no
  longer run. The launch manifest records this cancellation and status.
- The user subsequently interrupted **17686978** as well; Slurm confirmed
  cancellation at **17:21:08 PDT**. All 21 submitted jobs are now canceled.
  Its stdout/stderr and Hydra directory `outputs/2026-10-01/16-57-39/`
  were removed, and three tracked bytecode files it regenerated were restored.
  It saved no checkpoint or evaluation artifacts. The launch manifest
  retains the administrative cancellation/cleanup record; no online_v2
  training or evaluation results remain from this batch.
- New checkpoints/configs are written beneath `dev/train_logs/online_v2/`;
  the bulk evaluator uses `dev/eval_out/online_v2/`. Saved configs must
  contain `ALLOCATION_PROTOCOL: online_v2`; old/untagged configs are
  rejected by current training/evaluation to prevent silent protocol changes.
- The redesign also fixes the previously audited GRU replay reset
  mismatch: PPO stores the pre-observation reset separately from the
  post-transition terminal flag used for GAE.

Historical methods, numerical results and figures remain in
[the representation report](analysis/representation_results/README.md)
and in the explicitly historical sections below. Their mixed scientific
result is preserved rather than relabeled as evidence for the new design.

## What we are testing

> When an ego agent interacts repeatedly with partners that differ in
> hidden task-specific capabilities, does recurrent training with diverse
> partners cause the ego to infer partner capability and use that
> representation to allocate tasks appropriately?

Everything in this repo — env, trainer, config, tests, evaluation —
targets that question and only that question. This is the grid analogue
of Mon-Williams et al.'s Overcooked partner-modelling experiment.

## Design summary

### Environment — `jaxmarl.environments.coordination_grid.CoordinationGrid`

- 7x7 grid, one ego (learned) and one scripted partner.
- At reset of **each round**, a uniform random RED/BLUE allocation is
  sampled independently of partner capability and shown to the ego via
  `last_allocation`. It also sets the partner's complementary goal.
- At **t=0 both agents stay**. The policy has one forced action:
  STAY plus the supplied allocation. It cannot choose or override that
  allocation during initialization, including through direct env calls.
- At **every t>=1**, the ego jointly chooses movement and RED/BLUE
  allocation. Under influence=True the request sets the partner's goal
  before that step's navigation; the ego can reassign it during the round.
  The partner cannot choose/reassign goals and follows its BFS path.
- The round ends on success or `max_steps`.
- **20 rounds per partner episode.** Hidden capability `(d_R, d_B)` is
  **fixed** for all 20 rounds. RNN hidden state persists across rounds;
  physical round state resets; layout may change round-to-round from the
  same 1096-layout corpus.
- `done['__all__']` fires **only** at the end of the 20th round — that's
  where the GRU hidden state resets and a fresh capability is drawn.

A new partner's capabilities are initially unknown. The initial assignment
is environmental randomization, not a learned commitment. From t=1 the
GRU combines current observations with its history to choose movement and
allocation. It receives no explicit capability profile or supervised
partner-model objective. It can revise an allocation after seeing behavior
later in the same round; in later rounds it also retains previous-round
behavioral evidence. A fresh partner episode resets this memory.

### Capability semantics (reference-style delays)

Partner capability is a pair `(d_R, d_B)` where each entry is the number
of wait steps between successive partner moves while pursuing the
corresponding goal:

    d = 0  -> moves every step   (fastest)
    d = k  -> moves once every (k + 1) steps

This matches the reference **evaluation** counter in
`baselines/IPPO/test_ippo_rnn_overcooked_v2.py`. It does **not** exactly
match the released reference trainer: that trainer decrements a newly
loaded counter on the same step, producing an action period of
`max(d, 1)` without task switches, rather than `d+1`. Grid training and
evaluation consistently use `d+1`; the reference has a train/eval timing
discrepancy. See the audit below before interpreting nominal cooldowns
as equivalent executed speeds.

Capability populations live in
`jaxmarl/environments/coordination_grid/capability_populations.py`:

    FAST_DELAYS = (1, 2, 3)
    SLOW_DELAYS = (4, 7, 8, 9)

    TRAIN_CAPABILITY_PAIRS (24)  =  FAST x SLOW  ∪  SLOW x FAST
    TEST_CAPABILITY_PAIRS (22)   built from scalars including 0, 5, 6
                                 that never appear in training.

Test pairs are **genuinely novel** — not merely unseen combinations of
familiar values.

### Action space and allocation influence

The legal learned action space at t>=1 is:

    {UP, DOWN, LEFT, RIGHT, STAY} x {ALLOC_RED, ALLOC_BLUE}

- `ALLOC_RED` means ego RED / partner BLUE; `ALLOC_BLUE` means ego BLUE /
  partner RED. The ego chooses its movement freely; its path is not fixed.
- The existing storage encoding `a = 3*move + alloc` and 15 output logits
  are retained, but NONE codes `(0,3,6,9,12)` are **always masked out**.
  Keeping reserved storage codes does not make NONE a legal action.
- At t=0 only id 13 or 14 is legal, whichever encodes the supplied default.
  There is no learned choice: log probability and entropy are zero.
- At t>=1 legal ids are `(1,2,4,5,7,8,10,11,13,14)`, giving ten choices.
  Both the RNN and feedforward policy use the same observation-dependent mask.
- Under **INFLUENCE=true**, every t>=1 request controls the partner's goal.
- Under **INFLUENCE=false**, the random initial partner assignment is held
  fixed throughout the round; subsequent ego requests have no causal effect.
  Round parity is no longer used. The default is sampled identically in
  both conditions, so only the ability to reassign differs.

`last_ego_allocation` supplies the random default at reset/t=0 and echoes
previous ego requests thereafter, with identical semantics in both
conditions. `partner_assignment` records the actual allocation driving
navigation and can diverge from the request under no-influence.
Capability remains absent from observations.

**Cooldowns on reassignment:** repeating an allocation preserves the
counter. When the actual partner goal changes, its counter is loaded with
that goal's delay before movement; the partner waits/decrements if positive.
Delay zero permits movement immediately. A switch to delay d>0 therefore
waits d ticks, then movement follows the usual d+1 cadence. Reloading only
on a true change prevents requests from granting free moves or repeatedly
resetting the counter when an assignment is held constant.

### Layouts

Current corpus: `dev/grids_capability_selected_balanced_1096/` (1096 layouts).

Same 1096 layouts are used for training AND evaluation. There is no
layout-generalization experiment. There are no val/test layout splits.

This is a sampled subset of the 2,000-layout current-capability corpus.
Requiring both ego distances in 1–8 leaves 1,597 layouts; randomized joint
distance balancing retains 1,096, with exactly 137 layouts at each distance
for each goal. Selection uses only ego distances. The subset contains
559 partner-equidistant layouts (51.0%) and mean wall density 24.94%.
The source layouts and metadata are preserved. The subset manifest and
`diagnostics/selection_audit.csv` record seed 2026 and all exclusions.

### Reward + horizon

- `success_reward = 1.0`, `step_penalty = 0.01` (validated setting).
- `max_steps = 100` per round. Historical analytical completion for a fixed allocation
  under the training+test capability populations on the selected corpus
  is 50 steps (see `dev/train_logs/capability_validation_ms100.json`),
  so the unchanged horizon admits a successful constant-allocation strategy.
  This is not a dynamic-reallocation oracle or a guarantee for arbitrary switching.

## The four experimental conditions

All four use the same trainer, the same env, the same reward, the same
layout distribution, and the same test evaluation. They differ only in
these config switches:

| Condition                        | MODEL_TYPE | PARTNER_REGIME | INFLUENCE |
|----------------------------------|------------|----------------|-----------|
| Multi-partner RNN (main)         | rnn        | diverse        | true      |
| Multi-partner MLP (memory ctrl)  | mlp        | diverse        | true      |
| Single-partner RNN (diversity ctrl) | rnn     | single         | true      |
| No-influence RNN (influence ctrl)| rnn        | diverse        | false     |

The single-partner control fixes the training capability to
`DEFAULT_SINGLE_PARTNER = (1, 4)`. It is still evaluated on the same
novel test partners as every other model.

Submitted for online_v2: **5 independent training seeds per condition**, seeds 1–5.
The previously completed five-seed runs used fixed allocation and remain historical.
Policy files follow `<condition>_seed<seed>` with condition identifiers
`rnn_diverse_influence`, `mlp_diverse_influence`,
`rnn_single_influence`, and `rnn_diverse_noinfluence`.

## PPO / network configuration

Matches the reference recurrent config:

    LR              = 5e-4  (linear warmup 0.05 → cosine)
    NUM_ENVS        = 256
    NUM_STEPS       = 256
    UPDATE_EPOCHS   = 4
    NUM_MINIBATCHES = 64        # was 8 pre-refactor
    TOTAL_TIMESTEPS = 60_000_000

    CLIP_EPS        = 0.2   ENT_COEF     = 0.01
    GAMMA           = 0.99  GAE_LAMBDA   = 0.95
    VF_COEF         = 1.0   MAX_GRAD_NORM = 0.25

    FC_DIM_SIZE     = 128
    GRU_HIDDEN_DIM  = 128
    ACTIVATION      = relu

MLP variant replaces the GRU cell with one extra Dense layer of the same
width; every other component is shared.

## Capability / layout scheduler

`baselines/IPPO/sweep_scheduler.build_schedule` samples uniformly:
one capability pair per 20-round partner episode from
`TRAIN_CAPABILITY_PAIRS`; independently sampled layout per round from the
1096-layout corpus. The materialized schedule is long enough
(`N_EPS_TOTAL = 131072`) that a training run does not wrap in practice.

There is **no** exhaustive capability × layout balancing. The old
`sanity_check_schedule` is kept as an alias to `summarize_schedule` which
just reports coverage stats.

## Active files

The following entry points and supporting files are wired into the
current experiment. Earlier stage/sweep and Overcooked plotting scripts
and their historical outputs were moved to
`archive/2026-10-02-retired-analysis/` on 2026-10-02. Original paths are
preserved beneath that directory; its `manifest.json` records every
original path, size, and SHA-256 checksum. The archive contains 19 retired
code/config/notebook files and their outputs (21,459 files, 111.03 MiB).
These files are preserved for history and are not active entry points.
All current v1/v2 code, checkpoints, rollouts, completed-run logs, layout
provenance, frozen v2 source, and representation/comparison/learning-curve
results were retained. Generated data and generic supporting modules
are not retired merely because they are absent from this list.

    Environment:
      jaxmarl/environments/coordination_grid/coordination_grid.py
      jaxmarl/environments/coordination_grid/capability_populations.py
      jaxmarl/environments/coordination_grid/__init__.py

    Trainer + scheduler:
      baselines/IPPO/ippo_rnn_coordination_grid.py     # unified trainer
      baselines/IPPO/sweep_scheduler.py                # simplified sampler
      baselines/IPPO/config/ippo_coordination_grid.yaml

    Launchers:
      bash/train_final_experiment.sh                    # CONDITION=... SEED=... sbatch
      bash/eval_all_checkpoints.sh                      # standalone evaluation of saved configs

    Analytical validation + selection:
      dev/capability_validation.py                      # oracle vs blind gap
      dev/capability_selection.py                       # completion_time primitive
      dev/build_final_corpus.py, dev/env_generator.py   # layout generator

    Tests:
      dev/test_capability_env.py
      tests/coordination_grid/test_online_allocation.py
      tests/analysis/test_representation_analysis.py    # representation/protocol regressions

    Evaluation:
      analysis/evaluate_partner_modelling.py

    Representation analysis:
      analysis/representation_analysis.py               # current grid-specific entry point
      requirements/representation.txt                  # standalone analysis dependencies
      analysis/representation_results/README.md         # methods, results, interpretation
      analysis/representation_results/                  # numerical tables, metadata, figures, NPZs

## Checkpoints, configs and data provenance

### Counterbalanced v1/v2 training — submitted 2026-10-02

The `v1_balanced_training` and `v2_balanced_training` experiments use the
same frozen 1096-layout corpus and preallocated sampling schedule. This
also changes v1's corpus from its original 1000 layouts. Each version
retains its own original environment, allocation rules, network, PPO
implementation, and evaluation functions; only sampling and exposure
audit bookkeeping are patched into its frozen trainer. The v1 source
comes from Git revision `541452613cdc531c98dfc5e0b8f00efe34fae3de`.

The sampler lives in `baselines/IPPO/counterbalanced_scheduler.py`.
Each shared 20-layout packet is allocated once to each of the 24 diverse
profiles, in shuffled profile order, through one global asynchronous
queue. Every layout pass is without replacement. Five passes form a
complete cycle: 274 episodes / 5480 rounds per profile, 6576 episodes /
131520 rounds total, and exactly five visits to every profile-layout pair.
The single-partner control retains only `(1,4)` and uses the same layout
stream. There is no queue wrap or independent worker cursor overlap.

Training still stops after 915 PPO updates: 59,965,440 actual environment
steps under the nominal 60M-step budget, with five learner seeds per
condition. Preallocated episode counts differ by at most one between
profiles at every queue prefix. Completed exposure can differ because of
unfinished episodes at this fixed-step cutoff. Each checkpoint writes
`*_sampling_audit.json` and `*_sampling_audit.npz`, including actual
rounds started/completed and environment steps for every profile-layout
pair, episode counts, the allocated queue prefix, and active episode IDs.

The launch is reproducible with `bash/submit_counterbalanced_training.py`
(`--prepare`, then `--submit-manifest`). Its manifest is
`logs/sbatch_counterbalanced1096_20261002_235609.json`; frozen source,
corpus, schedules, and preflight results are beneath
`dev/run_snapshots/counterbalanced1096_20261002_235609/`.

| Condition | v1 jobs (seeds 1–5) | v2 jobs (seeds 1–5) |
|-----------|---------------------|---------------------|
| Diverse RNN + influence | 17697345–17697349 | 17697365–17697369 |
| Diverse MLP + influence | 17697350–17697354 | 17697370–17697374 |
| Single RNN + influence | 17697355–17697359 | 17697375–17697379 |
| Diverse RNN, no influence | 17697360–17697364 | 17697380–17697384 |

Evaluation jobs **17697385** (v1) and **17697386** (v2) each depend on
successful completion of their version's 20 training jobs. They retain
20 episodes per familiar/novel capability, 20 rounds per episode, and
evaluation seed 12345. Checkpoints/configs/audits and evaluation outputs
are separated under `dev/train_logs/{v1,v2}_balanced_training/` and
`dev/eval_out/{v1,v2}_balanced_training/`, respectively.

Preflight passed 21 sampler/protocol regression tests, end-to-end training
and audit-export smoke runs for all eight version/condition combinations,
and standalone evaluation/HDF5 export smoke runs for both versions. Source
and schedule hashes, original network/environment/evaluation functions,
and launcher syntax were verified before submission. Existing experiment
weights, rollouts, code snapshots, and numerical results were retained.

**Completed results (checked 2026-10-03):** all 40 training jobs and both
queued evaluations completed with exit code `0:0`. The v1 evaluation
finished at 00:54:35 PDT and v2 at 02:31:42 PDT. Each policy collected
59,965,440 environment steps. Held-out results below average five learner
seeds; each seed evaluates 22 novel profiles × 20 episodes × 20 rounds.
Layouts are familiar, so this measures partner generalization.

| Condition | v1 success, mean ± SD | v2 success, mean ± SD | v1 episode return | v2 episode return | v1 episode steps | v2 episode steps |
|-----------|-----------------------|-----------------------|-------------------|-------------------|------------------|------------------|
| Diverse RNN + influence | 98.37 ± 0.47% | 96.06 ± 4.66% | 15.43 | 12.80 | 444.3 | 660.7 |
| Diverse MLP + influence | 98.58 ± 1.09% | 94.54 ± 4.81% | 13.66 | 11.76 | 624.9 | 733.8 |
| Single RNN + influence | 98.74 ± 0.63% | 93.62 ± 6.71% | 13.82 | 11.59 | 612.2 | 732.3 |
| Diverse RNN, no influence | 89.81 ± 11.35% | 90.14 ± 9.91% | 9.80 | 9.80 | 833.6 | 841.2 |

Counterbalanced v2 diverse RNN leads the mean success and return rankings;
its success increased 11.22 percentage points relative to the preceding
v2 run (84.84% → 96.06%) and episode steps decreased 26.3%. Diverse MLP
improved 7.03 points, no-influence improved 15.25 points, and the single
RNN mean changed only 0.10 points. These rankings are descriptive;
substantial seed variation remains. In v1 the three influence conditions
all approach 98–99% success, while diverse RNN has higher return and fewer
steps. Comparing new and original v1 also changes the corpus from 1000
to 1096 layouts, so that improvement cannot be attributed only to sampling.

Independent reconstruction of each schedule prefix, subtracting the exact
256 active episodes at the cutoff, matched every saved profile-layout
matrix of rounds started and completed in all 40 runs. All trained profiles
saw all 1096 layouts. Allocated episodes differ by at most one per profile;
actual started counts for any one layout differ by at most two across
profiles. Environment-step exposure is not balanced across profiles because
round durations differ. This verification is reproducible with
`analysis/audit_counterbalanced_results.py counterbalanced1096_20261002_235609`.

Plots and per-seed/aggregate JSON/CSV results are in
`analysis/protocol_comparison/counterbalanced1096_20261002_235609/`.
Generate the three-panel comparison, episode-step plot, and round-success
plot with `analysis/compare_allocation_protocols.py --counterbalanced-batch
counterbalanced1096_20261002_235609`. The launch manifest now records the
completed scheduler states and the sampling verification report.
The verifier also exports `actual_exposure_by_run.csv` (40 runs) and
`actual_exposure_by_profile.csv` (730 run/profile rows) in that results
directory. These distinguish allocated episodes from the empirical
started/completed episodes, rounds, and environment steps. The original
audit NPZs retain all profile-by-layout counts, including the unfinished
tail; full training observation/action trajectories were not saved.

**Current protocol:** checkpoint/config paths are beneath
`dev/train_logs/online_v2/`; standalone HDF5s/summaries go beneath
`dev/eval_out/online_v2/`. Config JSON and HDF5 attributes identify
`allocation_protocol = online_v2`. Fresh training has been submitted. No existing
weights, rollouts or numerical results were changed by the redesign.

### Balanced 1096-layout online_v2 Slurm launch — 2026-10-01

| Condition | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 |
|-----------|--------|--------|--------|--------|--------|
| `rnn_diverse_influence` | 17688017 | 17688018 | 17688019 | 17688020 | 17688021 |
| `mlp_diverse_influence` | 17688022 | 17688023 | 17688024 | 17688025 | 17688026 |
| `rnn_single_influence` | 17688027 | 17688028 | 17688029 | 17688030 | 17688031 |
| `rnn_diverse_noinfluence` | 17688032 | 17688033 | 17688034 | 17688035 | 17688036 |

The batch uses the existing PPO settings and resource requests: one 80G GPU,
4 CPUs, 32 GB memory, 8-hour training limit, account `nlp`, partition `sphinx`,
excluding `sphinx9`. All four conditions retain their original semantics;
the no-influence control freezes the random initial partner assignment.
The protocol is explicitly `online_v2`, with random defaults at t=0 and
online movement/allocation decisions from t=1 in influence conditions.

Standalone evaluation **17688037** has an `afterok` dependency on all 20
training jobs. It checks that all 20 checkpoint/config pairs exist and runs
20 episodes per capability for all 24 familiar and 22 novel profiles, with
seed 12345 and hidden-state recording for RNN policies. Its layouts default
to each saved training config, fixing the old evaluator's hard-coded
1,000-layout default. An explicit `--layouts_dir` override remains available.

Frozen source and layouts live under
`dev/run_snapshots/balanced1096_20261002_022924/source/`.
Checkpoints/configs live under
`dev/train_logs/online_v2/balanced1096_20261002_022924/`; standalone evaluation
outputs live under `dev/eval_out/online_v2/balanced1096_20261002_022924/`.
Each policy has its own Hydra output directory. The launch manifest records
source hashes, the layout aggregate hash, exact commands, job IDs and the
verified scheduler dependencies. `bash/submit_balanced_training.py` prepares
and submits reproducible batches, recording submissions incrementally.

Preflight: 11 online-protocol tests, two evaluator corpus-selection tests,
shell syntax checks, and a finite small PPO update using the complete
frozen 1,096-layout corpus passed. The small run imported the snapshot's
trainer/environment and had zero t=0 entropy. These are preflight and
submission checks, not evidence that full training or evaluation completed.

### Previous online_v2 Slurm launch — 2026-10-01 (canceled)

This table records the original submissions. Jobs 17686979–17686998 were
subsequently canceled at the user's request before starting; only training
job 17686978 was running at the cancellation check. The user then canceled
that job too, and its generated run files were cleaned as recorded above.

| Condition | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 |
|-----------|--------|--------|--------|--------|--------|
| `rnn_diverse_influence` | 17686978 | 17686979 | 17686980 | 17686981 | 17686982 |
| `mlp_diverse_influence` | 17686983 | 17686984 | 17686985 | 17686986 | 17686987 |
| `rnn_single_influence` | 17686988 | 17686989 | 17686990 | 17686991 | 17686992 |
| `rnn_diverse_noinfluence` | 17686993 | 17686994 | 17686995 | 17686996 | 17686997 |

Each job uses `bash/train_final_experiment.sh`, one GPU with the `80G`
constraint on `sphinx`, account `nlp`, 4 CPUs, 32 GB RAM and an 8-hour
limit. Each trains one policy (`NUM_SEEDS=1`) at the documented PPO
settings with a nominal 60,000,000-step budget. Integer rollout batching
gives 915 updates × 65,536 transitions = **59,965,440 actual steps** per
policy. Tags are exactly `<condition>_seed<seed>` beneath the online_v2
artifact directories; preflight confirmed no matching artifacts existed.
W&B is disabled. Layouts, capability populations, episode length,
schedule seed, optimizer settings and reward follow the configuration above.

The training launcher now uses the evaluator's existing cuDNN settings
(`TF_CUDNN_USE_AUTOTUNE=0`, `TF_CUDNN_DETERMINISTIC=1`,
`--xla_gpu_deterministic_ops=true`) following earlier convolution failures
on multiple nodes. Both launchers exclude `sphinx9`, where earlier runs
also failed. These are execution settings, not environment-design changes.

**Standalone evaluation job 17686998** runs
`bash/eval_all_checkpoints.sh` with an `afterok` dependency on all 20
training jobs and a 2-hour limit. An invalid dependency cancels evaluation;
a failed training job therefore requires repair/resubmission before the
full evaluation can run. Training also performs its existing in-process
evaluation before exiting. The standalone evaluator uses seed 12345 and
20 episodes per capability for all 24 familiar and 22 novel profiles,
writes summaries and train/test HDF5s beneath `dev/eval_out/online_v2/`,
and saves hidden states for the 15 RNN policies. Representation analysis
has not been rerun on this protocol yet.

The [launch manifest](logs/sbatch_online_v2_20261001_235402.json) records
all job IDs, exact submission commands, hyperparameters, Git HEAD and
SHA-256 hashes of the active source files. The submitted working tree
includes the uncommitted online_v2 implementation. Slurm captures each
batch script at submission; Python source is read from the shared repository
when the job starts. Training logs are `logs/cg_cap_<jobid>.out/.err`;
standalone evaluation logs are `logs/cg_eval_all_17686998.out/.err`.
Shell syntax and `git diff --check` passed before submission. Scheduler
inspection confirmed all 21 jobs and all 20 evaluation dependencies.
These are submission checks, not evidence of completed training.

**Historical fixed-allocation artifacts:** the committed full experiment includes, for each of the 20 policies:

    dev/train_logs/<condition>_seed<seed>_config.json
    dev/train_logs/<condition>_seed<seed>_eval.json
    dev/eval_out/<condition>_seed<seed>_summary.json

The corresponding weights and rollout scans are local files, excluded
from Git by `.gitignore`:

    dev/train_logs/<condition>_seed<seed>.safetensors
    dev/eval_out/<condition>_seed<seed>_train.h5
    dev/eval_out/<condition>_seed<seed>_test.h5

There are 40 rollout files across all four conditions; representation
analysis uses only the **30 RNN files**. Training stdout/stderr and Hydra
output directories are also ignored. The committed
`logs/sbatch_*.txt` manifests record launches/relaunches. A fresh checkout
therefore includes configs and summaries but needs the local weights/HDF5s
to reproduce downstream analysis, and local completed training logs to
repeat exact final-training-return checkpoint selection.

The analysis phase did not modify the environment, trainer, capability
populations, reward, layouts, task structure or trained checkpoints.

## How to launch each condition

    # Main multi-partner RNN, seed 1
    TAG=rnn_diverse_influence_seed1 sbatch bash/train_final_experiment.sh

    # Other conditions (same launcher, one env var):
    CONDITION=mlp_diverse_influence TAG=mlp_diverse_influence_seed1 sbatch bash/train_final_experiment.sh
    CONDITION=rnn_single_influence TAG=rnn_single_influence_seed1 sbatch bash/train_final_experiment.sh
    CONDITION=rnn_diverse_noinfluence TAG=rnn_diverse_noinfluence_seed1 sbatch bash/train_final_experiment.sh

    # Different seeds
    SEED=2 TAG=rnn_diverse_influence_seed2 sbatch bash/train_final_experiment.sh
    ...

Explicit canonical tags above let representation analysis discover seeds
1–5. If TAG is omitted, the launcher uses a timestamped run name instead.
For online-v2 representation fits, use `--eval-dir dev/eval_out/online_v2`
and a separate output path such as `--out-dir analysis/representation_results_online_v2`.

## Evaluation

    python analysis/evaluate_partner_modelling.py \
        --config <path to config json> \
        --params <path to safetensors> \
        --n_episodes_per_capability 20 \
        --seed 12345 \
        --out_prefix dev/eval_out/online_v2/<condition>_seed<seed> \
        --save_hidden               # only meaningful for MODEL_TYPE=rnn

Outputs a per-checkpoint HDF5 rollout record for both the train and test
capability slices, plus a `{prefix}_summary.json` with headline metrics.
The bulk launcher `bash/eval_all_checkpoints.sh` loads saved configs,
uses evaluation seed 12345 and 20 repetitions/profile, and requests
hidden states only for RNN checkpoints. It skips prefixes with an
existing summary; if a rollout file is missing or malformed despite that
summary, rerun the evaluator explicitly for the affected checkpoint
rather than assuming the bulk launcher repaired it.

The current evaluator summarizes success, return and successful completion
for the valid prefix through the first final done. Online allocation metrics
exclude t=0, because its supplied default is not a policy decision:

    n_allocation_decisions / n_allocation_decisions_by_round
    initial_allocation_red_fraction              # initialization diagnostic
    fraction_partner_assigned_faster / _by_round # actual assignment, speed only
    fraction_ego_requested_faster_partner_goal   # requested, hypothetical in no-influence
    assignment_switch_count / _by_round
    assignment_switches_per_round / assignment_switch_rate
    partner_assigned_red_given_red_adv / blue_adv / tie

Speed adherence compares the assigned goal's d_R/d_B, excludes equal-delay
profiles, and weights each valid t>=1 decision equally. It is **not an
optimal-allocation oracle**: geometry, remaining distance, collisions and
switch costs can make the slower task a reasonable assignment. Success
and return remain the realized performance measures. The old t=0
optimal-allocation fraction/regret headline fields are not emitted for
online_v2. Their analytical helpers remain only for historical fixed-role
layout validation. In no-influence, actual assignment statistics and ego
requests are reported separately, rather than attributing causal control
to the ego.

HDF5 adds `assignment_changed` (actual switch) and `ego_request_changed`
(request switch). At t=0 `ego_alloc_action` records the supplied default;
at t>=1 it records the ego's request. Existing hidden-state timing and
first-done masking remain unchanged.

**Hidden-state convention (`--save_hidden`):** the saved `hidden_state`
is the POST-observation state h_t = RNN(h_{t-1}, o_t) — the state used
to produce this step's action. Downstream probes decode from the
representation the policy was actually acting from.

**Completion-time correction (2026-09-29):** the environment supplies
`info["round_time"] = new_time`, the **post-transition** round-local time.
The evaluator saves this as `round_time`, replacing the old pre-step
`time` field, and uses it for successful completion times. This avoids
an off-by-one mismatch with the analytical completion-time convention.
`test_info_round_time_is_post_transition` covers the correction. The
post-observation hidden-state convention remains unchanged.

The current scan schema includes `hidden_state`, `capability`, `dones`,
`round_idx`, `round_done`, `round_time`, `is_t0`, `layout_idx`,
`ego_alloc_action`, `partner_assignment`, `partner_goal`, `rewards`,
`success`, `assignment_changed`, `ego_request_changed`,
`capability_index_per_ep`, and `capability_pool`. RNN scans have
shape `(E, 2000, 128)` for hidden state; `E=480` for train and `E=440`
for test. The actual episode often ends before the 2000-step scan.
Behavioral summaries and representation features use states only through
the **first final done, inclusive**; later scan states are excluded.

Every trainer run also drops `<tag>_config.json` beside `<tag>.safetensors`
so the standalone evaluator can restore MODEL_TYPE / PARTNER_REGIME /
INFLUENCE / SINGLE_PARTNER / ENV_KWARGS exactly. The evaluator
explicitly re-injects `INFLUENCE` into `env_kwargs["influence"]` so a
no-influence checkpoint cannot be accidentally evaluated under an
influence-enabled env.

## Online-v2 implementation and verification

Changed environment resets/transitions, both policy masks, keyed scheduled
resets, PPO replay resets, training diagnostics, evaluator schema/metrics,
and launch/evaluation artifact directories. Representation analysis rejects
mixed fixed_v1/online_v2 HDF5 inputs and requires a separate online_v2 output
directory, preserving historical figures and fits. Capabilities, layout corpus,
reward, horizon and network widths are preserved.

Regression coverage checks random defaults and capability independence,
both-agent t=0 STAY, same-step t>=1 movement/reassignment, goal-specific
switch delays, repeated-assignment cadence, no-influence independence,
round/episode boundaries, JIT/vmap, RNN/MLP action probabilities, legacy
config rejection, one bounded PPO update and evaluation masking.

Verified on CPU: **11 online-allocation regression tests, all 20 existing
environment tests, and 14 representation/protocol tests passed**. The
replay regression reproduces collected log-probabilities and values using
unchanged parameters across an episode reset. The bounded PPO update
produced finite losses and zero initialization entropy. Shell syntax,
Python compilation and `git diff --check` also passed. This validation
does not constitute training or scientific evaluation of the new experiment.

    JAX_PLATFORMS=cpu python -m pytest tests/coordination_grid/test_online_allocation.py -q
    JAX_PLATFORMS=cpu python dev/test_capability_env.py
    python -m pytest tests/analysis/test_representation_analysis.py -q

## Historical fixed-allocation five-seed behavioral evaluation

**Everything in this section and the representation-results section below
was measured before online_v2. These numbers do not evaluate the new design.**

The table below is calculated from the five committed
`dev/eval_out/<condition>_seed*_summary.json` files per condition, using
the **novel test capability slice**. Values are mean ± sample SD over
the five independently trained policies; no policy was excluded.

| Condition | Round success | Episode return | Optimal ego allocation | Allocation regret |
| --- | --- | --- | --- | --- |
| Multi-partner RNN | 0.945 ± 0.058 | 13.388 ± 3.498 | 0.747 ± 0.212 | 0.0645 ± 0.0557 |
| Multi-partner MLP | 0.895 ± 0.058 | 10.556 ± 2.347 | 0.503 ± 0.005 | 0.1297 ± 0.0017 |
| Single-partner RNN | 0.961 ± 0.047 | 13.073 ± 2.017 | 0.497 ± 0.006 | 0.1324 ± 0.0013 |
| No-influence RNN | 0.907 ± 0.108 | 10.634 ± 4.269 | 0.501 ± 0.002 | 0.1290 ± 0.0006 |

For this historical no-influence control, the last two columns scored the
ego's hypothetical t=0 choice; actual partner assignment used round parity. High success alone does not demonstrate
partner-specific task allocation: the single-partner control succeeds
frequently while its chosen allocations remain near 0.5 optimal.

Mean novel-slice optimal-allocation fraction from round 0 to round 19:

| Condition | Round 0 | Round 19 |
| --- | --- | --- |
| Multi-partner RNN | 0.529 | 0.757 |
| Multi-partner MLP | 0.517 | 0.516 |
| Single-partner RNN | 0.502 | 0.496 |
| No-influence RNN | 0.482 | 0.512 |

These descriptive endpoints support mean adaptation in the multi-partner
condition, with substantial policy-seed variability. They are not a
formal significance test or a claim that every main-condition seed
converged successfully. Full by-round values, training-slice results,
allocation regret and assignment diagnostics remain in the source JSONs.

## Historical fixed-allocation representation analysis

### Entry point and analysis unit

```bash
python -m pip install -r requirements/representation.txt
python analysis/representation_analysis.py \
    --eval-dir dev/eval_out \
    --out-dir analysis/representation_results \
    --analysis-seed 0
```

The CLI discovers all five seeds for the three recurrent conditions and
fails with helpful errors for missing, ambiguous or malformed rollout
files. `--logs-dir` defaults to `logs`; `--device` defaults to `cpu` and
`--threads` to 1. The script fits probes in PyTorch and uses UMAP for
embedding; it imports neither the evaluator nor the PPO environment.
These fits train analysis probes only, not the saved policies.

For every network, train and test capability scans are combined into
**46 profiles × 20 repetitions = 920 partner episodes**. The episode,
not layout identity, is the analysis unit. The MLP is a behavioral control
only and is excluded from probes and UMAP. Hidden-state coordinate
systems from independently trained networks are never pooled.

Loading validates the authoritative populations, 920 episodes,
46 distinct pairs, exactly 20 repetitions/profile, hidden dimension 128,
constant capability on valid steps, finite hidden scans, final done for
every episode, and 20 correctly ordered round boundaries. Terminal steps
are included; padding/post-terminal states never enter the features.
All 15 recurrent policies passed, representing **13,800 episodes**.

### Probe features, split and fitting

- **Primary paper-style curve:** average the valid episode prefix through
  `min(t,L)` states for `t = 1,50,100,150,200,250,300,350,400`, where
  `L` includes the first final done. Save the fraction already complete
  at every cutoff.
- **Grid-native curve:** for rounds 0–19, average all valid hidden states
  from episode start through the inclusive end of that round. This is a
  separate analysis, not a replacement for the absolute-timestep curve.
- **Targets:** separate `d_R` and `d_B` probes with 10 ordered classes
  labeled directly 0–9. Inputs contain only the averaged 128-D hidden
  vector; capabilities and other rollout fields supply labels, masks
  and grouping only.
- **Architecture/training:** one affine 128→10 layer with bias,
  softmax cross entropy, full-batch Adam, lr=0.01, 1000 updates, no hidden
  layers, scaling, weight decay, early stopping or condition-specific
  tuning. Initialization seeds are 10000 for RED and 10001 for BLUE.
  Batched fitting has independent weights and Adam moments per probe.

`rollout_rep` is the zero-based occurrence within each capability profile
in source episode order, checked against stored capability indices.
Episodes are then ordered by `(d_R,d_B,rollout_rep)`. One PCG64 permutation
of repetitions 0–19, analysis seed 0, is reused across every condition,
policy seed, target and cutoff:

    train reps: [4,19,6,2,13,16,3,11,10,8,0,12,7,5,18,17]
    test reps:  [14,9,1,15]

This gives exactly 16/4 per profile and 736/184 episodes overall. No
profile is held out of probe fitting. The split and per-episode assignment
are saved in `probe_split.json` and `episode_manifest.csv`.

These are **repetition indices**, not twenty globally shared PRNG seeds.
The grid evaluator splits a root key (`12345`) into `C*20` reset keys;
different profiles therefore receive different layout sequences. The
reference evaluator instead reruns every profile with each seed 0–19.
The grid split preserves the 16/4 counts and is paired across conditions,
but is not an exact replication of that shared-rollout-seed design.

The primary metric is distance-aware accuracy,
`mean(1 - abs(predicted_delay - true_delay)/9)`. Exact 10-class accuracy
and MAE in delay units are secondary diagnostics. Lower cooldown means
faster task execution; a higher distance score does not mean a higher
partner cooldown or skill.

All **870 primary probes** were fitted separately by condition × policy
seed × target × cutoff: 270 timestep fits and 600 round fits. At each
point the robustness summary reports mean, sample SD and a 95% percentile
CI from **10,000 bootstrap resamples of the five policy seeds**, RNG seed
30000. Paired nominal-seed condition differences are also saved. This
uncertainty is across policies, not a rollout-level CI treated as if it
described independent models.

### Behavior-based selection and baselines

The paper-style view independently selects the highest final-training
`ep_return_mean` in each condition from completed logs that saved the
corresponding checkpoint. Selection uses the last PPO-update metric,
rounded to four decimals in stdout; no probe or novel-test score enters.

| Condition | Selected seed | Final-training episode return |
| --- | --- | --- |
| Multi-partner RNN | 5 | 15.8960 |
| Single-partner RNN | 5 | 17.4740 |
| No-influence RNN | 3 | 13.3662 |

Exact training metrics were available for all 15 policies. The implemented
fallback, unused in this run, is highest `train.mean_ep_return` from
committed standalone summaries, using that same proxy for all five
candidates within a condition. `selected_seeds.json` records selections,
candidate metrics, source logs, and whether a proxy was used.

The **random baseline** uses independent Normal(0,1) features of shape
`(920,128)`, real capability labels, the shared split and identical probe
training. Vector seeds 0–4 are saved: the selected-seed figure uses seed 0;
all-seed and round figures use the five-vector-seed mean. It is a
time-independent reference, not shuffled real hidden states.

The **shuffled-label diagnostic** permutes paired RED/BLUE labels over
episodes using seed 20000 and fits real t=400 features with the unchanged
split. All 30 primary diagnostic fits passed the predefined descriptive
screen: absolute distance-score difference from the random mean ≤0.10.
This is a broad leakage screen, not proof of statistical equivalence or
absence of every possible confound.

### Quantitative representation results

Values below combine all 46 profiles and retain all five policies.
Square brackets give 95% policy-seed bootstrap CIs at t=400.

| Condition | RED t=1 | RED t=400 [CI] | BLUE t=1 | BLUE t=400 [CI] | RED / BLUE round 19 |
| --- | --- | --- | --- | --- | --- |
| Multi-partner RNN | 0.637 | 0.838 [0.766, 0.900] | 0.650 | 0.840 [0.760, 0.907] | 0.861 / 0.861 |
| Single-partner RNN | 0.637 | 0.914 [0.862, 0.960] | 0.641 | 0.835 [0.816, 0.853] | 0.943 / 0.838 |
| No-influence RNN | 0.632 | 0.678 [0.665, 0.693] | 0.643 | 0.689 [0.654, 0.720] | 0.704 / 0.701 |
| Random Normal mean | — | 0.621 | — | 0.637 | 0.621 / 0.637 |

For multi-partner and single-partner RNNs, both targets improve from t=1
to t=400 and from round 0 to round 19 in **all five policies**. No-influence
decoding is weaker and nonmonotonic: its round-0→19 mean change is
−0.032 for RED and −0.085 for BLUE. Multi-partner t=400 SD is 0.084 for
RED and 0.095 for BLUE, so selecting a strong policy hides material
variation. No unsuccessful/weak policy was removed.

Multi-partner minus no-influence t=400 differences are +0.160 for RED
and +0.151 for BLUE, with paired bootstrap CIs above zero. Against
single-partner, the RED difference is −0.076 [−0.148,+0.006] at t=400
and −0.082 [−0.129,−0.026] at round 19; BLUE differences are small with
CIs spanning zero. Therefore the predicted selective advantage over
**both** controls on **both** cooldowns is not observed.

At t=400, shuffled-label scores across 15 policies average 0.648 for RED
and 0.638 for BLUE, compared with Normal means 0.621 and 0.637. Scores
must be interpreted against these diagnostics: distance-aware accuracy
is not exact accuracy, and central-class guesses can already score well.

Secondary held-out-rollout decoding partitions the same fitted probe's
test examples into 96 TRAIN-profile and 88 novel-profile episodes:

| Condition | RED familiar / novel | BLUE familiar / novel |
| --- | --- | --- |
| Multi-partner RNN | 0.830 / 0.847 | 0.848 / 0.832 |
| Single-partner RNN | 0.915 / 0.912 | 0.832 / 0.838 |
| No-influence RNN | 0.667 / 0.691 | 0.679 / 0.701 |

Novel profiles are absent from **policy** training but present in probe
training under the standard 16/4 split. This is not a held-out-capability
transfer test for the probe. For the single-partner policy, "familiar"
labels the experiment's 24-profile TRAIN slice; only `(1,4)` was actually
seen during its policy training.

### UMAP and protocol correspondence

Each behavior-selected network has its own 2-D UMAP of 920 vectors:
mean of the final 50 valid hidden states (all valid states if fewer than
50), `min_dist=1.0`, `n_neighbors=919`, `random_state=42`, otherwise
UMAP defaults. Each fit uses one trained policy's representations;
different policies' coordinate systems are never pooled. Episodes within
that fit contain changing layouts from the shared corpus, unlike the
reference's separate policy/UMAP for each fixed kitchen.
Panels share a color scale for `relative_red_speed_advantage = d_B-d_R`:
positive means RED faster, negative means BLUE faster.
`delay_diff = d_R-d_B` is saved too; companion figures color by each
individual cooldown. Selected multi-partner and single-partner panels
both show visibly organized relative-capability coloring; the no-influence
panel has more intermingled colors. UMAP is qualitative only and neither
quantifies representation strength nor demonstrates causal use.

The Overcooked-to-grid mapping preserves task-1/task-2 cooldowns as
`d_R`/`d_B`, 46 profiles (24 TRAIN + 22 novel), 20 rollouts/profile,
920 samples/network, 128-D recurrent state, final-50 and prefix averaging,
single-layer probes, 16/4 rollout split, Adam lr=0.01 for 1000 updates,
distance-aware accuracy, Normal baseline and the stated UMAP settings.
Major episode/analysis differences include **variable-length 20-round episodes** instead
of fixed 400-step episodes and **layouts changing each round from 1000
layouts** instead of five fixed kitchens. By t=400, 24.02% of multi-partner
and 20.91% of single-partner episodes are already complete on average,
versus 0% for no-influence. Cutoffs are clipped to valid length; round
curves provide the complementary view of interaction history.

### Outputs and verification

All results are under `analysis/representation_results/`:

    probe_timestep_per_seed.csv       probe_timestep_summary.csv
    probe_by_round_per_seed.csv       probe_by_round_summary.csv
    probe_by_round.csv                # alias of the per-seed round table
    probe_condition_differences.csv
    random_baseline.csv               shuffled_label_baseline.csv
    probe_split.json                  selected_seeds.json
    analysis_metadata.json            validation.json
    episode_manifest.csv              umap_selected.csv
    probe_models/*.npz                # fitted affine weights/biases/losses
    umap_features_<condition>.npz      # selected final-50 inputs
    umap_qualitative_observation.json  # inspected figure hash + observations
    README.md                         # complete methods/results/deviations

Publication figures are saved in **PNG and PDF**:

    figure_probe_timestep_selected.*
    figure_probe_timestep_all_seeds.*
    figure_probe_by_round.*
    figure_umap.*
    figure_umap_d_R.*
    figure_umap_d_B.*

Metadata records software versions, input paths/sizes/timestamps, source
hash, hyperparameters, RNG seeds, validation and checkpoint selection.
Qualitative UMAP notes are reused by the report only when their saved
figure hash matches the current image.

**13 representation regression tests passed**, covering terminal-inclusive
masking, round aggregation, malformed files, exact per-profile splits,
distance-aware scoring, equivalence of batched and independent linear
fits, behavior-only checkpoint selection/fallback, five-policy bootstrap,
paired contrasts and RNN-only discovery. Final artifact checks verified
15 policies, 30 source HDF5s, 13,800 episode records, 810 timestep rows,
1800 round rows (each table includes all/familiar/novel subsets),
2760 selected UMAP points and all 12 figure files. No PPO/environment
test rerun was needed for this analysis-only addition.

```bash
python -m pytest tests/analysis/test_representation_analysis.py -q
```

## Historical fixed-allocation analytical validation on the corpus

Run once after any capability-population or `max_steps` change:

    python dev/capability_validation.py \
        --layouts_dir dev/grids_capability_selected/layouts/train \
        --max_steps 100 --step_penalty 0.01 \
        --out dev/train_logs/capability_validation_ms100.json

At the currently frozen `max_steps=100`, on the 1000-layout corpus, with
the partner-blind baseline chosen by expected REWARD per layout
(matching `capability_selection.evaluate_layout`; the earlier
success-rate-based blind was misleading at max_steps=100 where nearly
every alloc succeeds):

    train pool  : oracle round_succ = 1.000, blind = 0.999
                  oracle reward/round = +0.8252, blind = +0.6988
                  oracle - blind reward = +0.1264
    test  pool  : oracle round_succ = 1.000, blind = 1.000
                  oracle reward/round = +0.8676, blind = +0.7418
                  oracle - blind reward = +0.1258
    max observed oracle completion = 50 steps  (safety margin ~2x)

Both slices have `flip_fraction = 1.0` — every layout's optimal
allocation depends on the capability profile, which is exactly the task
pressure we want.

## Historical scientific replication criteria and conclusion

These are scientific success criteria, distinct from completion of the
implementation and analysis. High PPO success or decodable hidden states
alone do not establish the full claim.

**Core replication complete** when across multiple training seeds:

    1. Multi-partner RNN,  2. Multi-partner MLP,  3. Single-partner RNN

trained under matched conditions and evaluated on the same novel test
partner population, show behavior that depends on partner relative
capability — with the multi-partner RNN allocating tasks in a way that
tracks `d_B - d_R` more strongly than the two controls do.

Then, for the representational result:

    4. No-influence RNN

trained identically to (1) except allocation is decoupled from ego
action; use linear probes of GRU hidden state → `d_R` and → `d_B` to
compare how strongly capability is decodable across (1) vs (3) vs (4).

**Current conclusion:** the completed analyses show partner-dependent
allocation adaptation in the multi-partner condition on average, and
linearly recoverable cooldown information that strengthens with history
in both influence-enabled RNN conditions. The main condition decodes
more strongly than no-influence, but the diversity-specific
representational prediction fails against the single-partner control.
Strong single-partner decoding coexists with nearly flat allocation
adaptation. Information being recoverable does not establish that the
policy uses it causally. Report this distinction and policy-seed
variability rather than treating the full scientific replication as
established or changing the experiment to obtain the expected pattern.

## Historical replication-fidelity audit (2026-10-01, before redesign)

This audit describes the original fixed-allocation implementation. Online_v2
supersedes its t=0 commitment, immutable round assignment and parity-control
rows, and fixes its PPO replay reset mismatch. Other differences remain,
including ego speed, task/layout structure, population weighting and model
architecture. This redesign has not established any new scientific result.

Audited against the [published NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/0b8e8bfc40184226888e821620b216c9-Paper-Conference.pdf)
and upstream commit
[`35a8430046531bd4b62e923d7507ee9f02fd97ea`](https://github.com/ruaridhmon/emergent_partner_modelling/tree/35a8430046531bd4b62e923d7507ee9f02fd97ea),
which matched remote `main` at audit time. Original source was read with
`git show HEAD:<path>`: the local reference checkout contains additional
working-tree changes that are not part of the release. All 20 saved grid
configs were inspected, rather than inferring runs from current defaults.

**Verdict:** the grid implementation preserves the central Experiment-1
question, four controls, recurrent architecture family and representation
measurements. It is a scientific task adaptation, not yet the closest
possible methodological replication of Overcooked. Differences in task
allocation, executed speeds, control architecture and training populations
can change the pressure to model a partner. They are not explanations
established by the current results.

### What matches the released implementation

- All 20 saved configs share the released recurrent PPO settings:
  256 environments, 256-step buffers, four epochs, 64 minibatches,
  lr=5e-4, 5% warmup and cosine decay, clip=0.2, entropy=0.01,
  gamma=0.99, lambda=0.95, value coefficient=1 and max gradient norm=0.25.
  The 60-million-step budget matches the released RNN YAML default.
- The CNN has the same six convolutions, followed by normalization and
  a 128-D GRU; the recurrent actor/critic heads use the same widths and
  initialization scheme. The grid/allocation encoder is an adaptation:
  CNN output 64 plus an 8-D allocation embedding, then projection to 128;
  the reference CNN projects directly to 128.
- The capability **supports** match released evaluation code exactly:
  24 familiar and 22 novel profiles, with 20 rollouts each. Post-observation
  hidden-state logging, prefix means and final-50 means also match the
  reference representation conventions.
- Five trained seeds per condition and return-based checkpoint selection
  are available. Reporting all five seeds and round-based curves adds
  useful robustness evidence beyond the selected-policy view.

### Differences that matter for fidelity

| Component | Released Overcooked code | Current grid implementation | Consequence |
|---|---|---|---|
| Ego execution | Cooldown parameter 2; training acts every 2 ticks and evaluation every 3 without switches | Moves every tick after the allocation step | Different balance of ego and partner contribution |
| Partner execution | Train counter gives period max(d,1); evaluation gives d+1 | Consistent d+1 in both | Nominally identical profiles have different training speeds |
| Influence | Toggle partner subtask during an episode | Allocation only at t=0 of each round; immutable during navigation | Different exploration and allocation decisions |
| Interaction horizon | Continuous 400-step evaluation episode | 20 rounds, fresh physical state/layout per round, variable total length | Different history, reward opportunities and state distribution |
| Partner | Two frozen learned subtask policies, sampled actions | Deterministic shortest-path BFS with goal-specific delays | Capabilities produce simpler, more regular movement evidence |
| Layout/observation | Separate policies for fixed kitchens; self-centred observations | One policy across 1000 layouts; absolute full-grid observation | Extra layout variation and a different observation encoder |
| Diverse population weights | RED delay uniform over seven values, then BLUE drawn from opposite group | Uniform over all 24 pairs | Reference orientation probabilities 3/7 and 4/7 become 1/2 and 1/2 |
| Single-partner control | Fixed (2,2) | Fixed (1,4) | Balanced reference control becomes an asymmetric specialist |
| Feedforward control | Flattened observations, separate two-layer width-64 actor/critic MLPs | Shared CNN encoder, LayerNorm and width-128 dense replacement for GRU | Valid memory control, but different capacity and inductive bias |
| No-influence schedule | Training mode set by episode time before/after t=200 | Partner assignment alternates by round parity | Different exposure to each subtask |
| Rollout randomization | Seeds 0–19 reused for every profile | One batch root key, distinct per-profile repetition keys | Counts match, shared seed/layout matching does not |
| Learner samples | Buffer/loss includes both agents, despite partner action replacement | Only ego transitions train the policy | Cleaner ego-only learning, not literal source-code equivalence |

The released feedforward YAML also uses different optimizer/rollout
defaults (lr=2.5e-3, 16 environments, 128-step buffers, four minibatches,
15 million steps, tanh). Our feedforward condition deliberately shares
the recurrent settings. This improves matching within the grid experiment
but does not reproduce the released feedforward configuration; neither
YAML alone proves which settings produced the published figures.

The published paper also has a separate red/blue **CoinGame** experiment
(Appendix A), with stochastic skill and an asymmetric single partner.
That supports asymmetric controls as a task-adaptation choice; it does not
make (1,4) a reproduction of the released Overcooked single control.
The current grid task is not that CoinGame protocol either.

Behavioral evaluation also uses grid-specific oracle allocation accuracy
and regret incorporating geometry. These are useful for this task but
are not the reference's soup-throughput and task/relative-speed
correlation measurements. For no-influence, the ego-choice accuracy is
hypothetical, as qualified above.

### Published protocol versus released analysis

The published document is internally inconsistent on training budget:
Table 1 lists 10 million steps while B.1.1 says 15 million. The released
RNN YAML lists 60 million. Its reported scalar cooldown sets also differ
from the released profile lists. Therefore matching code defaults is not
the same as matching every statement in the paper, and actual published
run configurations cannot be established from those defaults alone.

The representation stage follows the **written probe protocol**: separate
fits, Adam lr=0.01 for 1000 steps, 16/4 repetitions per profile, real
capability labels for Normal features, and the stated UMAP settings.
It is not a literal reproduction of
[`analysis/do_ablation_plots.py`](https://github.com/ruaridhmon/emergent_partner_modelling/blob/35a8430046531bd4b62e923d7507ee9f02fd97ea/analysis/do_ablation_plots.py),
which instead:

- uses AdamW with weight decay 1e-3 and 1001 updates;
- reshuffles/splits within scalar label classes, rather than grouping
  by profile and shared rollout seed;
- selects probe weights on the **test set**, evaluating every 20 updates;
- carries selected weights forward between time cutoffs;
- uses random integer labels for its Normal-feature baseline.

The release's [`analysis/utils.py`](https://github.com/ruaridhmon/emergent_partner_modelling/blob/35a8430046531bd4b62e923d7507ee9f02fd97ea/analysis/utils.py)
also forces 5000 UMAP epochs, whereas current UMAP uses library defaults.
Probe initialization differs too (Flax default dense initialization
versus current fixed uniform initialization). These differences should
be disclosed; reproducing test-set selection would compromise the
held-out interpretation and is not recommended for the primary analysis.
The current analysis provenance points to arXiv v1; in the published
version, the corresponding methods are **B.3**, not A.2.

### Correctness issues discovered in source comparison

The grid trainer has a recurrent **PPO replay reset mismatch**, also
present in the released recurrent trainer. During collection it supplies
`last_done` before processing an observation (`ippo_rnn_coordination_grid.py`,
lines 538–541). The buffer stores the terminal flag from the subsequent
environment step (line 607), and the loss reuses that flag as the reset
for the stored observation (line 688). Thus a terminal transition resets
one observation too early in replay; the next episode's first observation
does not receive its collection-time reset. A rollout buffer beginning
immediately after termination can also replay with an incorrect initial
reset. GAE correctly needs post-step terminal flags; GRU replay needs a
separate pre-observation reset field. This affects training correctness,
not the saved evaluation hidden-state convention.

An isolated CPU check executed the actual trainer's `ScannedRNN` class
on five synthetic observations, with identical parameters and initial
carry for collection and replay. Collection reset flags were
`[False, False, False, True, False]`; current replay used
`[False, False, True, False, False]`. Maximum absolute hidden-state
differences per step were approximately `[0, 0, 0.4170, 0.1401, 0.0726]`.
Using the aligned collection reset flags gave exactly zero difference.
This verifies the reset mismatch; it does not quantify its effect on
the trained checkpoints or scientific results. Scalar counter checks
also confirmed the reference train/eval action periods in the table.

The reference no-influence evaluator contains another discrepancy:
`mode = where(t >= 200, 1-mode, mode)` flips mode on every step after
t=200, rather than once. Its trainer uses `mode = time >= 200`.
Grid parity assignment consistently removes causal influence, and this
reference evaluation behavior should not be copied as an intended control.

### Recommended order of follow-up work

1. In a separately versioned training change, fix the replay reset field
   and verify that unchanged parameters reproduce collection log-probabilities
   across episode boundaries. Existing checkpoints/results must retain
   their provenance; this requires new training to assess consequences.
2. Define the intended replication target explicitly: Overcooked
   Experiment 1, CoinGame Appendix A, or the present grid adaptation.
   Record choices where paper and code disagree, especially cooldowns,
   training budget and single-partner profile.
3. For a closer Overcooked task transfer, add an explicitly defined ego
   cooldown and within-episode partner switching, then match the single
   control and population weighting. Revalidate layout feasibility and
   analytical allocation rewards before retraining: slowing the ego changes
   those calculations. Preserve current results as the original experiment.
4. For an evaluation-only improvement, use explicit shared seeds 0–19
   across profiles and save seed IDs. Keep the current primary probe
   protocol; disclose released-script differences rather than silently
   replacing it with test-selected fits.

No environment, trainer, analysis implementation, rollouts or checkpoints
were changed by this audit. Only this project log was updated. The
partner-switching and blind-agent studies from the paper remain outside
the implemented Experiment-1 scope.

## Deferred / not-in-this-experiment

Partner **capability/profile** switching mid-episode (distinct from the
implemented within-round goal reassignment), causal capability-shuffle interventions
(distinct from the completed shuffled-label probe diagnostic),
observation blindness / local visibility, communication conditions,
larger capability grids, unseen-layout generalization, layout
augmentation, additional procedural environments, more elaborate causal
representation interventions — these are follow-ups. They must not
block or complicate the core replication.
