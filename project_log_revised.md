# CoordinationGrid — Final Experiment (Overcooked replication in a gridworld)

Last refactor: **2026-09-29**.

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
- Each round: ego commits at t=0 to a task allocation, then both agents
  navigate; round ends on success or `max_steps`.
- **20 rounds per partner episode.** Hidden capability `(d_R, d_B)` is
  **fixed** for all 20 rounds. RNN hidden state persists across rounds;
  physical round state resets; layout may change round-to-round from the
  same 1000-layout corpus.
- `done['__all__']` fires **only** at the end of the 20th round — that's
  where the GRU hidden state resets and a fresh capability is drawn.

### Capability semantics (reference-style delays)

Partner capability is a pair `(d_R, d_B)` where each entry is the number
of wait steps between successive partner moves while pursuing the
corresponding goal:

    d = 0  -> moves every step   (fastest)
    d = k  -> moves once every (k + 1) steps

This matches the reference `slow_delay_0 / slow_delay_1 / wait_buffer`
mechanic in `REF_emergent_partner_modelling/baselines/IPPO/ippo_rnn_overcooked_v2.py`.

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

- Ego action ∈ {UP, DOWN, LEFT, RIGHT, STAY} × {NONE, ALLOC_RED, ALLOC_BLUE},
  flat-encoded in [0, 15).
- **t=0:** only `STAY+ALLOC_RED` (13) and `STAY+ALLOC_BLUE` (14) legal.
- **t≥1:** only the 5 moves + NONE (ids 0, 3, 6, 9, 12).
- `ALLOC_RED` ⇒ partner takes BLUE; `ALLOC_BLUE` ⇒ partner takes RED.
- **INFLUENCE=true:** partner assignment is determined by the ego's t=0
  alloc (the "main" mechanism).
- **INFLUENCE=false:** partner assignment is determined by round-index
  parity (round_idx & 1), independent of ego's action. This is the
  no-influence control.

The env keeps two separate fields to prevent an observation confound:

    state.last_ego_allocation  -> ego action channel; drives obs["last_allocation"]
    state.partner_assignment   -> internal; drives the partner's goal commit

`obs["last_allocation"]` has **identical semantics** under influence=true
and influence=false (the ego always sees its own alloc at t=1 and NONE
thereafter). Only `partner_assignment` differs across the two
conditions. Verified in `test_observation_schema_matches_across_influence`.

Capability is **never** in the ego's observation.

### Layouts

Sole corpus: `dev/grids_capability_selected/` (1000 layouts).

Same 1000 layouts are used for training AND evaluation. There is no
layout-generalization experiment. There are no val/test layout splits.

### Reward + horizon

- `success_reward = 1.0`, `step_penalty = 0.01` (validated setting).
- `max_steps = 100` per round. Analytical worst-case oracle completion
  under the training+test capability populations on the selected corpus
  is 50 steps (see `dev/train_logs/capability_validation_ms100.json`),
  so this horizon is comfortably above infeasibility.

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

Target: 5 independent training seeds per condition.

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
1000-layout corpus. The materialized schedule is long enough
(`N_EPS_TOTAL = 131072`) that a training run does not wrap in practice.

There is **no** exhaustive capability × layout balancing. The old
`sanity_check_schedule` is kept as an alias to `summarize_schedule` which
just reports coverage stats.

## Active files

The following are the ONLY files wired into the current experiment.
Everything else in `dev/` and `analysis/` is retired and has been marked
with a `SUPERSEDED` banner at the top.

    Environment:
      jaxmarl/environments/coordination_grid/coordination_grid.py
      jaxmarl/environments/coordination_grid/capability_populations.py
      jaxmarl/environments/coordination_grid/__init__.py

    Trainer + scheduler:
      baselines/IPPO/ippo_rnn_coordination_grid.py     # unified trainer
      baselines/IPPO/sweep_scheduler.py                # simplified sampler
      baselines/IPPO/config/ippo_coordination_grid.yaml

    Launcher:
      bash/train_final_experiment.sh                    # CONDITION=... SEED=... sbatch

    Analytical validation + selection:
      dev/capability_validation.py                      # oracle vs blind gap
      dev/capability_selection.py                       # completion_time primitive
      dev/build_final_corpus.py, dev/env_generator.py   # layout generator

    Tests:
      dev/test_capability_env.py

    Evaluation:
      analysis/evaluate_partner_modelling.py

## How to launch each condition

    # Main multi-partner RNN, seed 1
    sbatch bash/train_final_experiment.sh

    # Other conditions (same launcher, one env var):
    CONDITION=mlp_diverse_influence     sbatch bash/train_final_experiment.sh
    CONDITION=rnn_single_influence      sbatch bash/train_final_experiment.sh
    CONDITION=rnn_diverse_noinfluence   sbatch bash/train_final_experiment.sh

    # Different seeds
    SEED=2 sbatch bash/train_final_experiment.sh
    ...

## Evaluation

    python analysis/evaluate_partner_modelling.py \
        --config <path to config json> \
        --params <path to safetensors> \
        --n_episodes_per_capability 20 \
        --seed 12345 \
        --out_prefix dev/train_logs/<tag> \
        --save_hidden               # only meaningful for MODEL_TYPE=rnn

Outputs a per-checkpoint HDF5 rollout record for both the train and test
capability slices, plus a `{prefix}_summary.json` with headline metrics.

The primary behavioral metric is **fraction_optimal_allocation** —
computed per t=0 sample using the same analytical
`completion_time`/`_reward_from_time` primitive as
`capability_validation.py`. For each (layout_idx, capability) the
optimal alloc is `argmax(reward_A, reward_B)`; ties (equal analytical
rewards) are excluded from the optimal-fraction accuracy but count with
regret = 0 in `allocation_regret_mean`. Metrics reported per slice:

    round_success_rate
    mean_reward_per_round
    mean_successful_completion_time
    fraction_optimal_allocation_overall
    fraction_optimal_allocation_by_round        # main adaptation curve
    allocation_regret_mean
    allocation_regret_by_round
    partner_assigned_red_given_relative_cap     # secondary/qualitative

`fraction_optimal_allocation_by_round` is the main adaptation signal:
the diverse-partner RNN should improve with round index; the MLP should
not; the single-partner RNN may learn a fixed prior with no adaptation.

**Hidden-state convention (`--save_hidden`):** the saved `hidden_state`
is the POST-observation state h_t = RNN(h_{t-1}, o_t) — the state used
to produce this step's action. Downstream probes decode from the
representation the policy was actually acting from.

Every trainer run also drops `<tag>_config.json` beside `<tag>.safetensors`
so the standalone evaluator can restore MODEL_TYPE / PARTNER_REGIME /
INFLUENCE / SINGLE_PARTNER / ENV_KWARGS exactly. The evaluator
explicitly re-injects `INFLUENCE` into `env_kwargs["influence"]` so a
no-influence checkpoint cannot be accidentally evaluated under an
influence-enabled env.

## Analytical validation on the current corpus

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

## Definition of "core replication complete"

Not "PPO reaches a high success rate". Not merely "the RNN trains".

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

## Deferred / not-in-this-experiment

Partner switching mid-episode, capability-shuffle interventions,
observation blindness / local visibility, communication conditions,
larger capability grids, unseen-layout generalization, layout
augmentation, additional procedural environments, more elaborate causal
representation interventions — these are follow-ups. They must not
block or complicate the core replication.
