# Project log — CoordinationGrid & emergent partner modeling

**Status:** 2026-09-25
**Purpose:** minimal but complete record of the task, data, training setup, major design iterations, final experiment, and what results are currently trustworthy.

**Scope change (2026-09-25):** the project has been pared back to the
**action_only** condition only. The prior three-condition design
(action_only / universal / partner_specific) has been retired from the
codebase and from this log. All references below to a three-condition
experiment describe the earlier scope of the project; the current experiment,
training pipeline, and evaluation cover **action_only** exclusively.

---

## 1. Research question (revised scope)

The project asks whether a recurrent learning agent, coordinating with a
scripted partner over multiple rounds of a small gridworld task, develops
useful **cross-round memory** about that partner when the *only* signal it
can rely on is the partner's realized behavior — i.e. no symbolic
communication channel is available.

There are two agents:

- **ego:** the only learned agent;
- **partner:** scripted and controlled by a latent parameter `z`.

The single supported communication condition is:

1. **`action_only`** — no symbolic communication. The partner commits to
   a goal uniformly at random at t=1 and then navigates greedily toward
   it. `z` is still carried in state (for schedule parity with the
   historical multi-condition experiment) but the partner's commit
   probability does not depend on `z`.

The retired conditions (`universal`, `partner_specific`) previously
produced z-independent and z-conditional message decoders; they have been
removed from the environment, trainer, config, shell scripts, and tests.

---

## 2. Task: two agents, two goals, one maze

Each round is a 7×7 gridworld containing:

- walls;
- ego start;
- partner start;
- RED goal;
- BLUE goal.

A round succeeds when the agents occupy **different goals simultaneously**:

```text
ego=RED  + partner=BLUE
or
ego=BLUE + partner=RED
```

They are therefore coordinating on a **complementary assignment**, not choosing the same color.

### Round timing

At round-local `t=0`:

- both agents must stay still;
- the ego's only legal action is `STAY+NONE` (no symbolic message channel
  is available in `action_only`).

At `t=1`:

- the partner commits once to RED or BLUE uniformly at random (P=0.5
  each), independent of any prior ego message and independent of `z`;
- it starts navigating toward that goal.

For `t>=1`:

- ego moves;
- the partner follows a deterministic shortest-path policy toward its
  committed goal;
- the partner's goal is fixed for the rest of that round.

### Movement and collisions

Movement actions are:

```text
UP, DOWN, RIGHT, LEFT, STAY
```

Walls and boundaries block movement. If both agents would occupy the same cell, or swap cells in one step, the move is blocked.

Partner navigation uses precomputed BFS next-action tables. Navigation itself never depends on `z`.

### Reward and termination

Per step:

```text
success       -> +1.0
otherwise     -> -0.01
```

A round ends on:

- coordination success, or
- `max_steps=15`.

The primary outcome throughout the final experiment is **round-level success**.

---

## 3. Observation and action representation

### Observation

Each agent receives the same dict:

```python
{
    "grid": (H, W, 5),
    "last_message": (3,),
    "is_t0": scalar,
}
```

Grid channels are:

1. walls;
2. RED goal;
3. BLUE goal;
4. ego position;
5. partner position.

`last_message` is a one-hot encoding of **ego's own message on the
previous step**. Under `action_only` the only legal ego message is
`NONE`, so this field is `[1, 0, 0]` on every step; it remains in the
observation for parity with earlier stages of the project.

`z` is **never included in the observation**.

### Ego action space

The network keeps the flat 15-way action head (5 moves × 3 messages) for
API stability, but under `action_only` the legal set is:

```text
t=0    :  {STAY+NONE}                                  (1 legal action)
t>=1   :  {UP,DOWN,RIGHT,LEFT,STAY} × NONE             (5 legal actions)
```

Illegal logits are set to `-inf`, so the same mask is respected during action sampling, PPO log-probability recomputation, and entropy calculation.

At t=0 the policy has entropy 0 by construction (single legal action);
this is a deliberate consequence of removing the message channel.

---

## 4. Partner type `z`

The final partner pool is:

```text
z ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
```

`z` is fixed for one 20-round partner episode.

Under `action_only`, `z` has no causal effect on the partner's goal
commit or navigation. It is still sampled and stored on state to keep
the balanced (z × layout) schedule structurally identical to the
retired three-condition experiment.

---

## 5. Partner episodes and recurrent memory

A **partner episode** contains:

```text
20 rounds
```

Within one partner episode:

- `z` stays fixed;
- each round uses a new layout;
- partner goal resets each round;
- round-local time resets to 0;
- ego's GRU hidden state **does not reset**.

Only after round 19 does:

```python
done["__all__"] = True
```

and only this terminal signal resets recurrent state.

This was an intentional departure from treating every maze round as an independent RL episode. The whole point is to give the recurrent policy a place to carry information across different task instances with the same partner — but in `action_only` any such information must come from *behavioral* observation, not messages.

GAE and value bootstrapping also use this partner-episode `done`, so intermediate round boundaries are not treated as RL terminals.

---

## 6. Network architecture

Only ego is learned. The partner remains scripted.

The recurrent actor-critic is adapted from the JaxMARL / Overcooked recurrent PPO code.

### Encoder

Grid:

```text
(H,W,5)
 -> CNN
 -> 64-d grid embedding
```

The CNN uses the existing Overcooked-style convolution stack:

```text
128@1×1
128@1×1
8@1×1
16@3×3
32@3×3
32@3×3
```

followed by flattening and a dense projection.

Message:

```text
last_message (3)
 -> Dense
 -> 8-d message embedding
```

Grid and message embeddings are concatenated and projected to:

```text
GRU_HIDDEN_DIM = 128
```

The resulting embedding is layer-normalized before the GRU.

### Recurrent core

```text
GRU hidden size = 128
```

The carry persists across rounds and resets only at the end of a 20-round partner episode.

### Heads

Actor:

```text
GRU output
 -> Dense(128), ReLU
 -> Dense(15 logits)
 -> legality mask (t=0: {12}; t>=1: {0,3,6,9,12})
 -> Categorical
```

Critic:

```text
GRU output
 -> Dense(128), ReLU
 -> scalar value
```

---

## 7. PPO learning rule

Training uses recurrent PPO with GAE.

Final matched hyperparameters:

```text
NUM_ENVS            = 256
NUM_STEPS           = 128
UPDATE_EPOCHS       = 4
NUM_MINIBATCHES     = 8

TOTAL_TIMESTEPS     = 60,000,000
LR                  = 5e-4
LR_WARMUP           = 0.05
ANNEAL_LR           = true

CLIP_EPS            = 0.2
GAMMA               = 0.99
GAE_LAMBDA          = 0.95
VF_COEF             = 1.0
ENT_COEF            = 0.01
MAX_GRAD_NORM       = 0.25
```

The learning-rate schedule is:

1. linear warm-up;
2. cosine decay.

Training is JAX/JIT-based and vectorized across 256 parallel environments.

The final experiment uses **one training seed** (`SEED=1`). This is enough for the current pilot-level result but is not a multi-seed robustness estimate.

---

## 8. Layout generation

### Generator

Base generator:

```text
dev/env_generator.py
```

The 7×7 layouts are rejection-sampled with:

```text
wall-density parameter        ~ Uniform[0.15, 0.60]
min agent-start separation    = 3 shortest-path steps
min goal separation           = 3 shortest-path steps
max assignment-cost gap       = 5
min mean switching cost       = 1.0
switching-cost prefix k       = 2
geometric preference threshold= 2
```

Every accepted layout must have all four agent→goal paths reachable.

The generator also rejects layouts where geometry makes the complementary assignment too obvious: if the two agents strongly prefer opposite goals by geometry alone, with both advantages at least the threshold, the layout is rejected.

Stored metadata includes:

- four agent→goal shortest-path lengths;
- number of shortest paths;
- junctions and dead ends;
- canonical path overlap;
- two assignment costs and their difference;
- switching-cost summaries;
- sampled and realized wall density.

The goal was to make coordination matter while avoiding layouts whose answer is visually trivial.

### Coordinate footgun

`env_generator.py` stores coordinates as:

```text
(row, col)
```

while `CoordinationGrid` uses:

```text
(x, y)
```

D4 helpers use `(x,y)`, so corpus construction explicitly converts at this boundary.

---

## 9. D4 symmetry: two different uses during development

D4 consists of the 8 square symmetries:

```text
identity
90° rotation
180° rotation
270° rotation
left-right reflection
up-down reflection
main-diagonal reflection
anti-diagonal reflection
```

Walls, starts, and both goals are transformed together. RED and BLUE labels are preserved.

BFS tables are recomputed after transformation instead of trying to remap action IDs.

### Pilot use: 8× online augmentation

During the first multi-layout pilots, every training layout was expanded to all 8 D4 versions inside the environment.

For 210 base training layouts:

```text
210 × 8 = 1680 effective training geometries
```

Validation and test were not augmented.

### Final use: one symmetry per base layout

For the final corpus we changed the design.

We generated exactly 2000 base layouts and applied **one** D4 transform to each:

```text
8 transforms × 250 base layouts each = 2000 transformed layouts
```

Then we deterministically shuffled and split them.

Therefore the final experiment uses:

```text
augment_symmetries = false
```

The environment must not expand them another 8×.

---

## 10. Final layout corpus

Builder:

```text
dev/build_final_corpus.py
```

Actual build used:

```text
master_seed  = 2026
shuffle_seed = 2026
```

The build log confirmed:

```text
2000 / 2000 unique base layouts
2000 / 2000 unique transformed layouts
```

and disjoint splits:

```text
train = 1600
val   = 200
test  = 200
```

Corpus:

```text
dev/grids_final/layouts/train
dev/grids_final/layouts/val
dev/grids_final/layouts/test
```

Each JSON records the applied symmetry for traceability.

The 250-per-symmetry balance is exact over the full 2000-layout corpus. Because the corpus was shuffled before splitting, each individual split is only approximately balanced over D4 transforms.

One implementation note: the builder currently **warns** rather than resamples if a future seed produces duplicate *base* layouts. The actual final seed produced 2000 unique bases, so this did not affect the reported experiment.

---

## 11. Balanced `z × layout` training schedule

We explicitly decided **not** to partition layouts by partner type.

The bad design would have been:

```text
z=.1 -> 320 layouts
z=.3 -> another 320 layouts
...
```

because maze geometry would then be predictive of partner identity.

Instead, every `z` sees the complete 1600-layout training set.

Scheduler:

```text
baselines/IPPO/sweep_scheduler.py
```

One balanced sweep contains:

```text
5 z values × 1600 layouts = 8000 (z, layout) pairings
```

For each `z`:

1. independently shuffle all 1600 layout indices;
2. chunk them into groups of 20;
3. each group becomes one 20-round partner episode.

Therefore:

```text
1600 / 20 = 80 partner episodes per z
80 × 5    = 400 partner episodes per sweep
```

Within one complete sweep:

- each `z` appears in exactly 80 episodes;
- each `z` sees each training layout exactly once;
- no `(z, layout)` pair is missing or duplicated;
- episode order is shuffled across z values.

Final scheduler settings:

```text
N_SWEEPS      = 256
SCHEDULE_SEED = 2026
```

The 256 vectorized workers start at different sweep boundaries. At the end of a partner episode, the trainer advances that worker's schedule cursor and injects the next scheduled `(z, 20-layout sequence)`.

The schedule wraps if training runs past its end.

Under the current single-condition scope, `z` is only a scheduling coordinate — the partner is `z`-independent — so the schedule's balance across `z` is preserved solely for symmetry with the historical multi-condition setup.

---

## 12. Manual scheduled reset

The original JaxMARL wrappers automatically reset an environment when it terminates.

That was incompatible with a precomputed balanced `(z, layout-sequence)` schedule.

For the final trainer:

- `LogWrapper` / automatic reset behavior is not used for training resets;
- `_env_step` calls `env.step_env` directly;
- when a partner episode ends, the trainer advances that slot's schedule cursor;
- `env.reset_from_schedule(z, layout_seq)` injects the next planned episode.

Episode return and length bookkeeping were reimplemented in the trainer.

This was one of the more invasive changes relative to the original Overcooked pipeline.

---

## 13. Development chronology and pilot lessons

### Stage A — prove the basic RL loop works

Setup:

```text
1 fixed layout
z = 1
1 round per episode
```

The recurrent PPO agent learned the task rapidly. In the logged run, success reached about:

```text
0.98
```

by roughly 180k environment transitions.

This established that reward, movement, partner scripting, recurrent PPO, and message-conditioned goal selection could be learned at all.

---

### Stage B — many layouts exposed severe memorization

First multi-layout corpus:

```text
210 train
45 val
45 test
```

Single-round training with `z=1`.

The model fit training extremely well but generalized poorly:

```text
held-out val  ≈ 0.16
held-out test ≈ 0.14
```

Diagnostics showed that the dominant held-out failure was navigation, not simply communication.

This was the first major surprise: the CNN+GRU could memorize a few hundred layouts instead of learning a transferable navigation/coordination rule.

---

### Stage B + D4 — geometry augmentation fixed much of the overfit

We then expanded each training layout to all 8 D4 symmetries.

Diagnostic held-out success rose to approximately:

```text
val  = 0.639
test = 0.622
```

The error breakdown changed substantially:

```text
val:
  success          .639
  navigation fail  .178
  assignment error .180

test:
  success          .622
  navigation fail  .246
  assignment error .116
```

This showed that orientation/location memorization had been a major part of the problem.

---

### Phase-specific action masking

We then enforced the task timing directly in the policy:

```text
t=0   -> message only (retired; now: STAY+NONE only)
t>=1  -> movement only
```

With masking, a later Stage-B diagnostic reached roughly:

```text
val  success = .645
test success = .677
```

and assignment errors fell to about 1%.

The remaining failures were mostly navigation:

```text
val  nav fail ≈ .342
test nav fail ≈ .309
```

This is why the final implementation keeps the 15-way head but masks illegal phase combinations rather than learning to ignore them. Under the current `action_only` scope, that mask degenerates to a single legal action at t=0.

---

### Stage C+D — introduce partner episodes and latent z

Next we moved from single rounds to:

```text
20 rounds per partner episode
z fixed across rounds
layout changes every round
GRU persists across rounds
```

The pilot z pool was:

```text
{0.2, 0.4, 0.6, 0.8}
```

with D4-augmented training layouts.

Partner visible throughout (`hide_partner_until_time=0`):

```text
val  overall ≈ .515
test overall ≈ .485
```

The striking result was the round trajectory:

```text
val r0 ≈ .066
val r1 ≈ .492
```

with later rounds around the .5-.6 range.

Something useful was clearly being carried across rounds.

(These pilots ran under a partner-specific decoder that has since been
removed from the codebase; the numbers are recorded here only as
development history.)

---

### Causal memory pilot: memory mattered, but not in the expected partner-specific way

We compared:

```text
normal      -> carry hidden state normally
round_reset -> zero hidden state at each round boundary
shuffle     -> transplant hidden state across parallel partner episodes
```

Results:

```text
VAL:
normal      .511
round_reset .095
shuffle     .504

TEST:
normal      .488
round_reset .069
shuffle     .481
```

So recurrence was essential, but shuffling the recurrent state between episodes barely hurt.

That means:

> "memory helps" did **not** imply "the hidden state contains a unique belief about this partner's z."

This was an important change in interpretation, and one of the motivations for eventually paring the study back to the single `action_only` regime while the causal-memory question is separated from the message-semantics question.

---

### The behavioral bypass

The Stage C+D policy (partner_specific decoder, retired) also learned to
avoid the intended partner-specific communication problem.

Its t=0 message distribution was dominated by `NONE`:

```text
val  NONE ≈ .758
test NONE ≈ .736
```

The partner was visible, so ego could often:

1. send no useful message;
2. watch which way the partner began moving;
3. infer its realized goal;
4. navigate toward the other goal.

Thus the task did not strictly force the agent to infer `z`.

Under the current `action_only` scope this bypass path *is* the only
available strategy: there is no message channel at all, so any
round-level success beyond chance must come from either navigation
skill, cross-round layout familiarity, or observing partner motion.

---

### K=3 partner-hiding intervention

To suppress the movement-observation bypass, we added:

```text
hide_partner_until_time = 3
```

which hides the partner-position channel at round-local t=0,1,2.

Unexpectedly, performance **improved** on the retired partner_specific
condition:

```text
val  ≈ .704
test ≈ .629
```

and the agent became even more likely to use `NONE`:

```text
NONE ≈ .988
```

The causal pattern remained similar:

```text
VAL:
normal      .700
round_reset .299
shuffle     .690

TEST:
normal      .638
round_reset .267
shuffle     .614
```

So K=3 did not produce the clean "must infer z from the message convention" regime we expected.

We therefore did **not** adopt K=3 in the final `action_only` experiment.

Final setting:

```text
hide_partner_until_time = 0
```

---

## 14. Why the scope collapsed to `action_only`

The earlier three-condition design tried to compare, in one experiment:

- how much explicit communication helps;
- how much easier a universal convention is;
- what additional burden comes from partner-dependent semantics.

Two things pushed us back to a single, narrower question:

1. In the previous three-condition run, the universal / partner-specific
   comparison was confounded with channel reliability (universal is
   deterministic; partner-specific caps at min(z, 1-z)-adjusted reliability),
   so the gap between them was not a clean estimate of the cost of
   inferring `z`.
2. The causal-memory diagnostic on the same models suggested that even
   under partner-specific communication, the recurrent state did not
   look partner-specific in the way we needed to make a strong claim.

Rather than layer more analysis on a confounded design, we pared the
codebase back to a single, well-defined condition (`action_only`) and
plan to re-introduce message-based conditions later, with a design that
avoids the reliability confound.

The current final experiment therefore runs one condition:

```text
action_only
```

on the same balanced (z × layout) schedule as before, with the same
architecture, PPO settings, layouts, z pool, and episode structure.

---

## 15. Final experiment launcher

Launcher:

```text
bash/train_final_experiment.sh
```

Single-job (no Slurm array):

```text
COMM_CONDITION = action_only
```

Final overrides:

```text
partner_z_values        = [0.1, 0.3, 0.5, 0.7, 0.9]
rounds_per_episode      = 20
hide_partner_until_time = 0
augment_symmetries      = false

train layouts = dev/grids_final/layouts/train
val layouts   = dev/grids_final/layouts/val
test layouts  = dev/grids_final/layouts/test
```

The generic YAML

```text
baselines/IPPO/config/ippo_rnn_coordination_grid.yaml
```

now targets `action_only` directly (previously it carried older
Stage-C+D defaults referencing the retired conditions and the wrong z
pool). The launcher still overrides all environment kwargs for
clarity.

---

## 16. Final evaluation procedure

For each split:

```text
5 z values
× 64 partner episodes per z
× 20 rounds per episode
= 6400 evaluated rounds
```

Each z therefore contributes exactly:

```text
1280 rounds
```

The held-out pool contains:

```text
200 validation layouts
200 test layouts
```

During evaluation, z is forced per parallel slot so each z gets exactly 64 partner episodes.

Layouts are sampled from the held-out pool within the episode; evaluation is **not** an exhaustive 5×200 z-layout Cartesian sweep.

All five final z values were used during training.

Therefore the final result tests:

> **generalization to held-out layouts, not generalization to unseen z values.**

Note again that under `action_only` the partner is z-independent, so
per-z eval numbers are expected to differ only through the layout /
schedule randomness in each z-conditional slot, not through any real
sensitivity of the policy to `z`.

---

## 17. Trusted final held-out results (action_only)

These values come from the current final evaluation file:

```text
final_action_only_seed1_20260925_142734_eval.json
```

Produced by rerunning the full training + eval pipeline (`sbatch
bash/train_final_experiment.sh`, Slurm job 17607287) under the pared-down
single-condition codebase, with `SEED=1`, `SCHEDULE_SEED=2026`, and all
other hyperparameters at the launcher defaults documented in §7 and §15.
Wall-clock training + eval was ~20 minutes on one 80G GPU.

### Overall round success

| split | round success | mean episode return |
|---|---:|---:|
| val  | **0.165** |  0.51 |
| test | **0.135** | -0.12 |

### Per-z round success

Because `action_only`'s partner does not depend on `z`, per-z differences
are effectively schedule / layout noise.

| z | val | test |
|---:|---:|---:|
| .10 | .166 | .147 |
| .30 | .173 | .127 |
| .50 | .170 | .143 |
| .70 | .151 | .144 |
| .90 | .168 | .117 |

### Selected per-round-index success (val / test)

| round | val | test |
|---:|---:|---:|
| r0  | .134 | .088 |
| r1  | .166 | .134 |
| r2  | .106 | .094 |
| r5  | .134 | .147 |
| r10 | .178 | .156 |
| r15 | .153 | .094 |
| r19 | .194 | .150 |

Round-level success rises from roughly r0 ≈ .09–.13 to r19 ≈ .15–.19.
Since there is no message channel, this modest upward drift is not
driven by learning any message convention; it reflects some
combination of navigation improvement across the episode,
layout-agnostic coordination heuristics, and observation of the
partner's realized motion after t=1.

### Training vs held-out gap

At the end of training, on-policy training round-success sat around
**0.55** while held-out val / test dropped to **0.17 / 0.14**. That
train/eval gap is large — clear evidence of overfitting to the
training layouts under this single seed. An earlier `action_only`
checkpoint on the previous (multi-condition-branched) codebase, run
with the same nominal seed, generalized much better (val ≈ .36, test ≈
.34). Because JIT-level trace changes shift the RNG stream even at the
same numeric seed, the two runs sampled different training trajectories
and landed on different solutions; the new run is what the current
codebase actually produces and is what we now report. A multi-seed
sweep is needed before treating either number as a stable characteristic
of the task.

---

## 18. What the final behavioral result supports (action_only)

### Above-chance coordination without communication

Round-level success of ~0.17 on val / ~0.14 on test is above the
"never coordinate" floor for a stationary ego and clearly above the
purely-random baseline, but far below the training-time round-success
of ~0.55 — the current single-seed model does not generalize well
without messages.

### Cross-round improvement is present but small

The r0 → r19 gain (~0.13 → ~0.19 on val) is real but modest: the
recurrent policy uses some cross-round experience to coordinate better
later in a partner episode, even with the partner scripted,
message-less, and z-independent. Under the earlier multi-condition
codebase a different action_only seed showed a much larger within-
episode ramp (r0 ≈ .17 → r19 ≈ .43), so this quantity is highly seed-
dependent and should be re-measured across multiple seeds before being
used as a headline result.

### Cross-round improvement is not partner-specific here

Because the partner's commit does not depend on `z`, any cross-round
memory that helps must be about *layout / behavioral* structure, not
about "who this partner is." So the action_only result serves as a
useful **floor for the emergent-partner-modeling story**: whatever
benefit we later attribute to modeling `z` needs to exceed this
baseline.

---

## 19. Retired analysis scripts

Post-hoc analyses that only make sense in the retired multi-condition
setup have been removed from the "trusted" set of outputs:

- `final_universal_*_eval.json`, `final_universal_*_causal.json`,
  `final_universal_*_bhz.json`
- `final_partner_specific_*_eval.json`,
  `final_partner_specific_*_causal.json`,
  `final_partner_specific_*_bhz.json`
- `final_action_only_causal.json`, `final_action_only_bhz.json`
  (these were the previously-flagged mis-configured runs)

The corresponding safetensors checkpoints for the retired conditions
have also been removed from `dev/train_logs/`.

The following historical analysis scripts remain in the tree:

```text
dev/behavior_by_z_by_round.py
dev/causal_memory_control.py
```

They can still be pointed at the current `action_only` checkpoint if we
later want:

- a hidden-state → `z` probe (chance = 1/5 = .20 for the five-z pool);
- normal vs round_reset vs same-z / cross-z hidden-state shuffle;
- per-`z`-per-round behavioral summaries.

Under `action_only` these serve as **floor** measurements: any
predictive information about `z` in the hidden state would be
suspicious (the training partner is `z`-independent), and any drop from
round-reset would be a pure recurrent-adaptation signal, not
partner-specific memory.

---

## 20. Relationship to the reference Overcooked project

Reference repository:

```text
ruaridhmon/emergent_partner_modelling
```

The conceptual inspiration is the same:

- recurrent ego agent;
- latent partner traits;
- traits fixed within an episode;
- PPO;
- 256 parallel environments;
- ask whether useful partner representations emerge from task pressure.

But several implementation details differ substantially.

### Reference Overcooked

- partner traits are task-specific action cooldowns / competence;
- partner type is sampled at episode boundaries;
- ego can influence which subtask the partner performs;
- partner traits alter observable behavior over the episode;
- each training run uses a fixed named Overcooked layout;
- the paper evaluates across several separate named layouts;
- training and evaluation partner configurations are disjoint in the main partner-generalization analysis.

### CoordinationGrid (current scope)

- layouts are procedural and change every round;
- there is no symbolic message channel; the only ego action at t=0 is
  `STAY+NONE`;
- the partner commits to a single goal uniformly at random at t=1
  and does not use `z`;
- final z values are the same at train and eval; held-out
  generalization is over layouts.

The balanced 8000-pair sweep is our addition. The reference code stochastically samples partner properties rather than building an exact partner×layout Cartesian schedule.

---

## 21. Evaluation / implementation bugs and fixes encountered

Key issues that changed the implementation or interpretation:

1. **Task generalization failure.**
   210 layouts were easy to memorize. D4 augmentation was needed.

2. **Phase/action ambiguity.**
   A 15-way unmasked head allowed redundant actions. Phase masks made the policy/action semantics explicit. Under the current `action_only` scope the t=0 mask reduces to a single legal action.

3. **Round boundary vs partner-episode boundary.**
   GRU reset must happen only after the final round, not every maze.

4. **Auto-reset wrappers.**
   Standard reset behavior conflicted with the balanced schedule, so reset and return bookkeeping moved into the trainer.

5. **Evaluation after terminal.**
   Multi-round evaluation uses `step_env` without autoreset and an alive mask so repeated post-terminal `round_done` states are not counted.

6. **D4 strategy changed.**
   Pilot: all 8 variants per training layout.
   Final: one balanced transform per independently generated base layout.

7. **Random z/layout sampling changed.**
   Pilot: independent uniform samples.
   Final: exact without-replacement z×layout sweep.

8. **K=3 intervention failed to do what was intended.**
   It improved task performance while messages collapsed toward NONE (partner_specific pilot), so it was excluded from the final design.

9. **"Memory helps" was overinterpreted initially.**
   Pilot hidden-state shuffling showed that useful recurrence need not mean z-specific memory.

10. **Retired analyses.** The previous `final_*_causal` / `final_*_bhz`
    files were generated with mismatched configuration and, for the
    universal / partner_specific conditions, with a decoder that no
    longer exists in the codebase. They have been removed rather than
    rerun.

11. **Scope reduction (2026-09-25).** The `universal` and
    `partner_specific` decoders, together with their trainer/network
    branches, config knobs, tests, and shell-script arms, were removed
    from the codebase. `communication_condition` still exists as an env
    kwarg but only accepts `action_only`; anything else raises.

---

## 22. Current trusted conclusions (action_only)

1. **The task is learnable at training time without any symbolic
   communication.** On-policy training round-success reaches ~0.55.

2. **The current single-seed policy generalizes poorly to held-out
   layouts.** Held-out round success is ~0.17 (val) / ~0.14 (test) —
   well below training performance. A multi-seed sweep is required
   before treating this as a stable estimate.

3. **The recurrent policy uses some cross-round experience.**
   Round-level success climbs modestly from ~.09–.13 at r0 to
   ~.15–.19 at r19 on held-out layouts.

4. **This improvement is not evidence of a partner-specific belief.**
   The partner ignores `z`, so any within-episode gain must come from
   layout/behavioral generalization, not from a model of the partner.

5. **`z` is a scheduling coordinate, not a task signal, under
   action_only.** Per-z eval differences are within schedule / seed
   noise.

6. **The current `action_only` numbers set the floor for any future
   partner-modeling experiment.** Additional benefit from
   partner-specific communication or partner-conditioned behavior
   needs to clear this baseline — but the baseline itself should be
   re-established across seeds before it can be used quantitatively.

---

## 23. Files that matter now

### Environment

```text
jaxmarl/environments/coordination_grid/coordination_grid.py
jaxmarl/environments/coordination_grid/__init__.py
```

### Layout generation

```text
dev/env_generator.py
dev/build_final_corpus.py
dev/grids_final/layouts/{train,val,test}
```

### Training

```text
baselines/IPPO/ippo_rnn_coordination_grid.py
baselines/IPPO/sweep_scheduler.py
baselines/IPPO/config/ippo_rnn_coordination_grid.yaml
bash/train_final_experiment.sh
```

### Tests

```text
dev/test_coordination_grid.py
```

These tests cover, among other things:

- t=0 legality (single legal action under action_only);
- p_red is 0.5 for every (msg, z) combination;
- D4 transforms;
- z persistence across rounds;
- layout changes;
- GRU reset semantics;
- JIT/vmap behavior;
- partner hiding.

### Final trained checkpoint

```text
dev/train_logs/final_action_only_seed1_20260925_142734.safetensors
```

### Trusted final evaluation

```text
dev/train_logs/final_action_only_seed1_20260925_142734_eval.json
```

### Historical / diagnostic analysis scripts

Retained but not currently pointed at fresh runs:

```text
dev/behavior_by_z_by_round.py
dev/causal_memory_control.py
```

### Pilot records

Useful historical files include:

```text
dev/train_logs/stageA1_20260924_122632.log
dev/train_logs/stage_b_diagnostic.json
dev/train_logs/stage_b_aug_diagnostic.json
dev/train_logs/stage_b_masked_diagnostic.json
dev/train_logs/stage_cd_R20_seed1_20260924_165732_eval_fixed.json
dev/train_logs/stage_cd_causal_memory.json
dev/train_logs/stage_cd_R20_hideK3_seed1_20260924_202349_eval.json
dev/train_logs/stage_cd_hideK3_causal.json
```

---

## 24. Reproduction checklist for a new coding agent

Before rerunning the final experiment, verify:

```text
[ ] final corpus has 1600/200/200 layouts
[ ] augment_symmetries=false
[ ] z pool = [.1,.3,.5,.7,.9]
[ ] rounds_per_episode=20
[ ] hide_partner_until_time=0
[ ] communication_condition=action_only (the only supported value)
[ ] schedule sanity check says 8000 pairings/sweep
[ ] every z sees all 1600 train layouts once/sweep
[ ] GRU resets only after round 19
[ ] z is absent from observation
[ ] final held-out evaluation uses dev/grids_final
```

For stronger scientific conclusions, rerun training with multiple
independent seeds and — when re-introducing partner-specific
communication — do so with a design that decouples channel reliability
from the "must infer z" pressure.
