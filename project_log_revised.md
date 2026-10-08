# CoordinationGrid — Counterbalanced v1/v2 experiments

Last design change: **2026-10-01** (random initialization and within-round allocation).
Last documentation update: **2026-10-07** (obsolete experiments deleted,
counterbalanced reruns verified, and completed ten-seed results documented).

## Current status

Both categorical counterbalanced batches completed: seeds 1–5 in
`counterbalanced1096_20261002_235609` and seeds 6–10 in
`counterbalanced1096_20261007_001529_seeds6to10`. Each contains four conditions
and both allocation protocols. Their manifests, matching frozen sources,
shared 1096-layout corpus, schedules, weights, and evaluations are retained.

Random-sampling runs, greedy-action archives, and their derived analyses were
deleted on 2026-10-07. Only CoordinationGrid remains in the active JAX package.
Frozen counterbalanced sources retain inherited modules for exact imports and
integrity checks. Common-seed figures now use counterbalanced-only selection.

## What we are testing

> When an ego agent interacts repeatedly with partners that differ in
> hidden task-specific capabilities, does recurrent training with diverse
> partners cause the ego to infer partner capability and use that
> representation to allocate tasks appropriately?

Everything in this repo — env, trainer, config, tests, evaluation —
targets that question and only that question. This is the grid analogue
of Mon-Williams et al.'s Overcooked partner-modelling experiment.

## Design summary

The transition rules below describe v2. V1 chooses its allocation at round
start and retains it for the round; its exact rules are preserved in the frozen
counterbalanced source. Both protocols use the paired counterbalanced scheduler.

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
the pinned external Overcooked reference. It does **not** exactly
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

Current corpus: `data_prep/grids_capability_selected_balanced_1096/` (1096 layouts).

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
  is recorded by `data_prep/capability_validation.py`,
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

Both allocation protocols have **10 independent training seeds per condition**:
seeds 1–5 in the base batch and seeds 6–10 in the exact-method extension.
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

Counterbalanced training uses `train/counterbalanced_scheduler.py`. Paired
profile packets feed a global queue shared by all workers. The single-partner
condition shares layout prefixes with the diverse conditions. The same layout
corpus and schedule seed (2026) apply to v1 and v2. Allocated episodes are
balanced at every queue prefix; completed exposure is audited separately.

The root trainer and episode scheduler remain source templates for preparation.
Prepared copies receive the counterbalanced scheduler patch before training.
Direct CLI training rejects the unprepared root template.

## Active files

- `jaxmarl/environments/coordination_grid/`: task and authoritative capabilities.
- `data_prep/`: source-corpus preparation, balancing, and validation.
- `bash/submit_counterbalanced_training.py`: paired snapshot preparation/submission.
- `bash/extend_counterbalanced_training.py`: exact-method seeds 6–10 extension.
- `train/`: shared trainer template, counterbalanced scheduler, validation and audits.
- `eval/`: behavior, representation and partner-dynamics analyses.

## Checkpoints, configs and data provenance

### Counterbalanced v1/v2 training — submitted 2026-10-02

The sampler lives in `train/counterbalanced_scheduler.py`.
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
`train/manifests/sbatch_counterbalanced1096_20261002_235609.json`; frozen source,
corpus, schedules, and preflight results are beneath
`train/run_snapshots/counterbalanced1096_20261002_235609/`.

| Condition | v1 jobs (seeds 1–5) | v2 jobs (seeds 1–5) |
|-----------|---------------------|---------------------|
| Diverse RNN + influence | 17697345–17697349 | 17697365–17697369 |
| Diverse MLP + influence | 17697350–17697354 | 17697370–17697374 |
| Single RNN + influence | 17697355–17697359 | 17697375–17697379 |
| Diverse RNN, no influence | 17697360–17697364 | 17697380–17697384 |

Evaluation jobs **17697385** (v1) and **17697386** (v2) were queued after
successful completion of their version's 20 training jobs. They used
20 episodes per familiar/novel capability, 20 rounds per episode, and
evaluation seed 12345. Checkpoints/configs/audits and evaluation outputs
are separated under `train/train_logs/{v1,v2}_balanced_training/` and
`eval/eval_out/{v1,v2}_balanced_training/`, respectively.

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

Independent reconstruction of each schedule prefix, subtracting the exact
256 active episodes at the cutoff, matched every saved profile-layout
matrix of rounds started and completed in all 40 runs. All trained profiles
saw all 1096 layouts. Allocated episodes differ by at most one per profile;
actual started counts for any one layout differ by at most two across
profiles. Environment-step exposure is not balanced across profiles because
round durations differ. This verification is reproducible with
`train/audit_counterbalanced_results.py counterbalanced1096_20261002_235609`.

Plots and per-seed/aggregate JSON/CSV results are in
`eval/protocol_comparison/counterbalanced1096_20261002_235609/`.
Generate the three-panel comparison, episode-step plot, and round-success
plot with `eval/compare_allocation_protocols.py --counterbalanced-batch
counterbalanced1096_20261002_235609`. The launch manifest now records the
completed scheduler states and the sampling verification report.
The verifier also exports `actual_exposure_by_run.csv` (40 runs) and
`actual_exposure_by_profile.csv` (730 run/profile rows) beneath
`train/sampling_audits/counterbalanced1096_20261002_235609/`.
These distinguish allocated episodes from the empirical
started/completed episodes, rounds, and environment steps. The original
audit NPZs retain all profile-by-layout counts, including the unfinished
tail; full training observation/action trajectories were not saved.

### Seeds 6–10 extension — completed 2026-10-07

`counterbalanced1096_20261007_001529_seeds6to10` adds 40 policies using the
base batch's exact frozen trainer, environment, evaluator, corpus, schedules,
and hyperparameters. All additional training jobs, both evaluations, and the
combined comparison completed successfully (43 jobs). Each added policy used
its recorded learner seed and completed 915 updates / 59,965,440 transitions.

The manifest is
`train/manifests/sbatch_counterbalanced1096_20261007_001529_seeds6to10.json`.
Combined familiar/novel metrics and per-profile figures are beneath
`eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/`;
`RESULTS_SUMMARY.md` there reports all four conditions. All ten learner seeds
contribute equally, retaining categorical action sampling and sample SD across
seeds. Novel-partner diverse-RNN success is 97.77 ± 1.56% for v1 and
95.96 ± 4.56% for v2, with episode returns 15.07 ± 1.14 and 13.07 ± 2.48.
Base five-seed results remain available separately.

## How to launch each condition

Prepare both protocols and all four conditions together, validate the new
manifest, then submit it:

```bash
python bash/submit_counterbalanced_training.py --prepare
python train/validate_counterbalanced_training.py --manifest train/manifests/<manifest>.json
python bash/submit_counterbalanced_training.py --submit-manifest train/manifests/<manifest>.json
```

The `--prepare` stage submits no jobs. Seeds 6–10 can be reproduced using the
extension workflow documented in `README.md`.

## Evaluation

Each frozen evaluator loads its protocol's matching network and resolved config.
The same 1096 familiar layouts are reused. Familiar evaluation covers the 24
training capability profiles; novel evaluation covers 22 held-out profiles.
Each policy contributes 20 episodes per profile, with 20 rounds per episode.
Terminal-inclusive masks exclude padded rollout ticks from all summaries.

Evaluation HDF5s and summaries live under
`eval/eval_out/{v1,v2}_balanced_training/<batch>/`. Use the frozen evaluator
selected by each batch manifest; current v2 environment code cannot evaluate
v1 weights with different transition rules.

```bash
python eval/compare_allocation_protocols.py --counterbalanced-batch counterbalanced1096_20261002_235609
python eval/compare_allocation_protocols.py --extension-manifest train/manifests/sbatch_counterbalanced1096_20261007_001529_seeds6to10.json --population test
python eval/plot_common_training_seed.py
```

The common seed is chosen using only retained counterbalanced training returns.
All-ten-seed performance includes every learner seed with equal weight.

## Historical online-v2 implementation checks

Changed environment resets/transitions, both policy masks, keyed scheduled
resets, PPO replay resets, training diagnostics, evaluator schema/metrics,
and launch/evaluation artifact directories. Representation analysis rejects
mixed fixed_v1/online_v2 HDF5 inputs and requires separate protocol-specific
output directories. Capabilities, layout corpus, reward, horizon and network
widths are preserved.

Regression coverage checks random defaults and capability independence,
both-agent t=0 STAY, same-step t>=1 movement/reassignment, goal-specific
switch delays, repeated-assignment cadence, no-influence independence,
round/episode boundaries, JIT/vmap, RNN/MLP action probabilities, legacy
config rejection, one bounded PPO update and evaluation masking.

Before the counterbalanced launches, CPU checks passed **11 online-allocation
regression tests, all 20 existing environment tests, and 14
representation/protocol tests**. The
replay regression reproduces collected log-probabilities and values using
unchanged parameters across an episode reset. The bounded PPO update
produced finite losses and zero initialization entropy. Shell syntax,
Python compilation and `git diff --check` also passed. These were implementation
checks; the completed training and scientific evaluations are recorded above.

    JAX_PLATFORMS=cpu python -m pytest tests/coordination_grid/test_online_allocation.py -q
    JAX_PLATFORMS=cpu python tests/coordination_grid/test_capability_env.py
    python -m pytest tests/analysis/test_representation_analysis.py -q

## Cleanup and verification — 2026-10-07

Deleted the original random-sampling v1/v2 runs, archived greedy experiments,
their checkpoints/configs/analyses, obsolete smoke outputs, unrelated active
JAX environments and utilities, and upstream Dockerfiles/documentation.
The active package registers only CoordinationGrid. Shared preparation
templates, layout provenance, all counterbalanced artifacts, and exact frozen
snapshots remain. Inherited modules and dependencies inside the snapshots are
required for their imports and recorded integrity checks.

Analysis entry points now use counterbalanced artifacts. Common-seed selection
uses only the base counterbalanced policies and still selects seed 5; all-ten
aggregation includes every learner seed. No retained scientific settings,
checkpoints, or frozen source bytes were changed.

Verification passed 119 tests and the standalone capability-environment checks.
A fresh batch prepared from the cleaned sources passed eight short CPU
training/evaluation checks covering both protocols and all four conditions.
The 101 recorded checkpoint/manifest hashes were unchanged, and full frozen
source, corpus, schedule, and extension-config integrity checks passed. No
training jobs were submitted during cleanup.

## Deferred / not-in-this-experiment

Partner **capability/profile** switching mid-episode (distinct from the
implemented within-round goal reassignment), causal capability-shuffle interventions
(distinct from the completed shuffled-label probe diagnostic),
observation blindness / local visibility, communication conditions,
larger capability grids, unseen-layout generalization, layout
augmentation, additional procedural environments, more elaborate causal
representation interventions — these are follow-ups. They must not
block or complicate the core replication.
