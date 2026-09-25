# Project log — CoordinationGrid & emergent partner modeling

**Status:** 2026-09-25  
**Purpose:** minimal but complete record of the task, data, training setup, major design iterations, final experiment, and what results are currently trustworthy.

---

## 1. Research question

The project asks when a recurrent learning agent develops and uses a **partner-specific representation** during coordination.

There are two agents:

- **ego:** the only learned agent;
- **partner:** scripted and controlled by a latent parameter `z`.

The central manipulation is the communication system:

1. **`action_only`** — no symbolic communication;
2. **`universal`** — a token has the same meaning for every partner;
3. **`partner_specific`** — the meaning of a token depends on `z`.

The main question is not just whether memory helps. It is whether the ego uses cross-round experience in a genuinely **partner-specific** way when communication semantics depend on the partner.

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
- ego may send its one allowed message for the round.

At `t=1`:

- the partner reads ego's previous message;
- it commits once to RED or BLUE according to the communication condition;
- it starts navigating toward that goal.

For `t>=1`:

- ego moves;
- the partner follows a deterministic shortest-path policy toward its committed goal;
- later messages do not change the partner goal.

The partner's goal is fixed for the rest of that round.

### Movement and collisions

Movement actions are:

```text
UP, DOWN, RIGHT, LEFT, STAY
```

Walls and boundaries block movement. If both agents would occupy the same cell, or swap cells in one step, the move is blocked.

Partner navigation uses precomputed BFS next-action tables. Navigation itself never depends on `z`; only the partner's goal choice can depend on `z`.

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

`last_message` is a one-hot encoding of **ego's own message on the previous step**:

```text
NONE, M0, M1
```

The partner does not send a symbolic message back to ego.

`z` is **never included in the observation**.

### Ego action space

The network always has one flat 15-way action head:

```text
5 moves × 3 messages = 15 actions
```

encoded as:

```python
action = 3 * move + message
```

We ultimately used a **single categorical head plus legality masks**, rather than separate move/message heads.

At `t>=1`, all conditions allow only:

```text
{UP, DOWN, RIGHT, LEFT, STAY} × NONE
```

At `t=0`:

```text
action_only:
    STAY × NONE

universal:
    STAY × {NONE, M0, M1}

partner_specific:
    STAY × {NONE, M0, M1}
```

Illegal logits are set to `-inf`, so the same mask is respected during action sampling, PPO log-probability recomputation, and entropy calculation.

This masking change mattered in the pilots: before strict phase masking, the policy could waste capacity on behaviorally redundant move/message combinations and assignment errors were much larger.

---

## 4. Partner type `z` and the three communication conditions

The final partner pool is:

```text
z ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
```

`z` is fixed for one 20-round partner episode.

### `action_only`

There is no useful symbolic channel.

```text
P(RED) = 0.5
P(BLUE) = 0.5
```

This ignores both the t=0 action and `z`.

Only `STAY+NONE` is legal at t=0.

### `universal`

Token meanings are fixed for every partner:

```text
P(RED | NONE) = 0.5
P(RED | M0)   = 1.0
P(RED | M1)   = 0.0
```

`z` is still carried in environment state so the episode structure stays matched across conditions, but it has **no causal effect on partner behavior** in this condition.

### `partner_specific`

Token meaning depends on the partner:

```text
P(RED | NONE)  = 0.5
P(RED | M0, z) = z
P(RED | M1, z) = 1 - z
```

For example:

- `z=.9`: M0 strongly means RED and M1 strongly means BLUE;
- `z=.1`: the mapping is nearly reversed;
- `z=.5`: M0 and M1 are both completely uninformative about the realized goal.

### Important interpretation caveat

`universal` and `partner_specific` differ in **two** ways:

1. whether the mapping must be partner-specific;
2. channel reliability.

Universal communication is deterministic. In the partner-specific condition, even an agent that knows `z` perfectly can only obtain best-token goal reliabilities:

```text
z=.1 -> .9
z=.3 -> .7
z=.5 -> .5
z=.7 -> .7
z=.9 -> .9
```

Therefore:

> `universal - partner_specific` is **not** a pure estimate of the cost of inferring `z`.

It mixes partner-specific inference with the fact that the universal channel is more reliable.

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

This was an intentional departure from treating every maze round as an independent RL episode. The whole point is to give the recurrent policy a place to carry information about the same partner across different task instances.

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
 -> legality mask
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

The final experiment used **one training seed per communication condition** (`SEED=1`). This is enough for the current pilot-level comparison but is not a multi-seed robustness estimate.

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

All three communication conditions use the same corpus, z pool, scheduler construction, and schedule seed.

### Why this changed

Earlier Stage C+D code sampled:

```text
z ~ uniform pool once per episode
layout ~ uniform pool independently each round
```

That kept `z` and layout independent, but did not mathematically guarantee that every z saw every layout.

For the final experiment we strengthened this to the balanced without-replacement sweep above.

### Small exposure-count caveat

Training is stopped by a fixed number of **environment transitions** (`60M`), not a fixed number of completed partner episodes.

Because successful rounds terminate early, a condition that solves rounds faster can advance farther through the schedule and therefore complete more partner episodes within the same transition budget.

All conditions see the full crossed support many times, but their exact number of completed schedule pairings need not be identical.

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
t=0   -> message only
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

This is why the final implementation keeps the 15-way head but masks illegal phase combinations rather than learning to ignore them.

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

This was an important change in interpretation.

---

### The behavioral bypass

The Stage C+D policy also learned to avoid the intended partner-specific communication problem.

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

This is a major difference from the intended "learn the partner-specific convention" story.

---

### K=3 partner-hiding intervention

To suppress the movement-observation bypass, we added:

```text
hide_partner_until_time = 3
```

which hides the partner-position channel at round-local t=0,1,2.

Unexpectedly, performance **improved**:

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

We therefore did **not** use K=3 in the final three-condition experiment.

Final setting:

```text
hide_partner_until_time = 0
```

We also discussed stronger task changes, such as making the t=0 assignment irreversible, but deliberately did not adopt them before the full three-condition comparison.

---

## 14. Why we moved to the three communication conditions

The earlier Stage C+D setup had only the partner-specific decoder, so it could not cleanly answer:

- how much does any explicit communication help?
- how much easier is a universal convention?
- what additional burden comes from partner-dependent semantics?

This motivated the final matched comparison:

```text
action_only
universal
partner_specific
```

The three conditions share architecture, PPO, layouts, z schedule, rewards, navigation, and episode structure.

The intended substantive manipulation is the availability / semantics of the t=0 token.

---

## 15. Final experiment launcher

Launcher:

```text
bash/train_final_experiment.sh
```

Slurm array:

```text
0 -> action_only
1 -> universal
2 -> partner_specific
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

### Important config footgun

The generic file:

```text
baselines/IPPO/config/ippo_rnn_coordination_grid.yaml
```

still contains older Stage-C+D defaults, including:

```text
partner_z_values        = [0.2, 0.4, 0.6, 0.8]
augment_symmetries      = true
hide_partner_until_time = 3
old dev/grids paths
communication_condition = partner_specific
```

The **final experiment is defined by the overrides in `train_final_experiment.sh`, not by the YAML file alone**.

A future coding agent should not run the YAML directly and assume it reproduces the final experiment.

---

## 16. Final evaluation procedure

For each communication condition and each split:

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

Earlier plans / pilots used different train and evaluation z sets, but that is not the final experiment.

---

## 17. Trusted final held-out results

These values come from the condition-specific final evaluation files:

```text
final_action_only_seed1_20260925_002403_eval.json
final_universal_seed1_20260925_002403_eval.json
final_partner_specific_seed1_20260925_002403_eval.json
```

### Overall round success

| condition | val | test |
|---|---:|---:|
| `action_only` | 0.362 | 0.338 |
| `partner_specific` | 0.594 | 0.585 |
| `universal` | **0.790** | **0.731** |

So the trusted final ordering is:

```text
universal > partner_specific > action_only
```

### Per-z round success

| z | action_only val | universal val | partner_specific val | action_only test | universal test | partner_specific test |
|---:|---:|---:|---:|---:|---:|---:|
| .10 | .377 | .773 | .606 | .337 | .732 | .591 |
| .30 | .359 | .807 | .604 | .320 | .737 | .599 |
| .50 | .375 | .779 | .588 | .338 | .717 | .570 |
| .70 | .347 | .800 | .562 | .352 | .752 | .586 |
| .90 | .352 | .791 | .609 | .342 | .716 | .577 |

Within each condition, held-out performance is fairly flat across z.

This does **not** by itself show that z is or is not represented internally. Equal performance could arise either because z does not matter to the learned strategy or because the policy successfully compensates for it.

### Selected validation round indices

| round | action_only | universal | partner_specific |
|---:|---:|---:|---:|
| r0  | .166 | .741 | .403 |
| r1  | .284 | .791 | .613 |
| r2  | .341 | .775 | .613 |
| r5  | .369 | .784 | .609 |
| r10 | .363 | .806 | .628 |
| r15 | .381 | .813 | .578 |
| r19 | .431 | .759 | .659 |

---

## 18. What the final behavioral result currently supports

### Universal communication

Universal is strong immediately:

```text
r0 val ≈ .74
```

This is expected because no partner inference is needed to interpret M0/M1.

Its round trajectory is comparatively flat.

### Partner-specific communication

Partner-specific improves sharply:

```text
r0 .403 -> r1 .613
```

This is consistent with useful cross-round adaptation.

But the trajectory alone does **not** prove that the agent inferred or represented `z`.

### Action-only

Action-only is the no-symbolic-communication floor:

```text
val  .362
test .338
```

It also improves substantially across rounds:

```text
r0 .166 -> r19 .431
```

Therefore cross-round improvement is not unique to the partner-specific communication problem. Recurrent state can help generic behavioral coordination / navigation as well.

### Communication gains

Approximate overall differences:

```text
universal - action_only:
  +43 pp val
  +39 pp test

partner_specific - action_only:
  +23 pp val
  +25 pp test

universal - partner_specific:
  +20 pp val
  +15 pp test
```

The last gap must not be labeled simply "the cost of z inference" because of the reliability confound described above.

---

## 19. Critical audit finding: the current `final_*_causal.json` and `final_*_bhz.json` files are not trustworthy final-condition analyses

The repository currently contains files named:

```text
final_action_only_causal.json
final_universal_causal.json
final_partner_specific_causal.json

final_action_only_bhz.json
final_universal_bhz.json
final_partner_specific_bhz.json
```

The previous project log interpreted these as post-hoc analyses of the final experiment.

That interpretation is not supported by the current scripts.

### Why

Both:

```text
dev/causal_memory_control.py
dev/behavior_by_z_by_round.py
```

default to loading:

```text
baselines/IPPO/config/ippo_rnn_coordination_grid.yaml
```

and the old:

```text
dev/grids/layouts/val
dev/grids/layouts/test
```

The YAML still specifies the old Stage-C+D environment:

```text
communication_condition = partner_specific
partner_z_values        = [0.2, 0.4, 0.6, 0.8]
hide_partner_until_time = 3
old layout pools
```

The analysis scripts currently do not infer the communication condition from the checkpoint.

For `action_only` and `universal` checkpoints this means the evaluation environment / action mask can be wrong.

There is direct evidence this happened:

- the saved causal outputs use a **4-way z pool** with chance probe accuracy `.25`, not the final five-way pool with chance `.20`;
- `final_universal_causal.json` reports normal success around `.425`, while the correct final universal evaluation is `.790` on val;
- `final_action_only_causal.json` similarly differs from the trusted final eval.

Therefore:

> **Do not use the current `final_*_causal.json` or `final_*_bhz.json` files to make claims about the final three-condition experiment.**

The old Stage-C+D causal analyses remain useful as records of those pilots because their configuration matches that pilot.

### Required rerun before making representation claims

The post-hoc scripts should be changed so each run explicitly uses:

```text
the checkpoint's true communication condition
the final five-z pool
hide_partner_until_time = 0
dev/grids_final/layouts/{val,test}
augment_symmetries = false
```

and the network config must receive the same condition-specific t=0 mask.

Then rerun:

1. normal hidden state;
2. round-reset;
3. same-z hidden-state shuffle;
4. strict cross-z hidden-state shuffle;
5. hidden-state → z probe;
6. message policy by z × round.

For the final five-z probe:

```text
chance classification accuracy = 1/5 = .20
```

Until that rerun is complete, the final experiment supports a **behavioral communication result**, but not yet a clean causal claim about z-specific hidden representations.

---

## 20. What the correct causal tests are intended to distinguish

### `normal`

Carry GRU hidden state normally across rounds.

### `round_reset`

Zero hidden state at every round boundary.

Tests:

> does any cross-round memory help?

This can hurt even if the memory is generic rather than partner-specific.

### same-z shuffle

At round boundaries, replace a slot's hidden state with one from another episode having the same z.

Tests whether useful memory is specific to the exact episode/history rather than merely the partner class.

### cross-z shuffle

At round boundaries, replace hidden state with one accumulated under a different z.

This is the stronger partner-model test.

If a hidden state contains a causally used z-specific belief, cross-z transplantation should be especially disruptive in the partner-specific condition.

A simple drop under `round_reset` is not enough to establish this.

---

## 21. Relationship to the reference Overcooked project

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

### CoordinationGrid

- layouts are procedural and change every round;
- partner type `z` affects the t=0 message→goal decoder only in `partner_specific`;
- the partner commits to a single goal once per round;
- ego has one small symbolic message opportunity at t=0;
- the final trainer explicitly crosses every z with every training layout;
- final z values are the same at train and eval; held-out generalization is over layouts.

The balanced 8000-pair sweep is our addition. The reference code stochastically samples partner properties rather than building an exact partner×layout Cartesian schedule.

### Important task-pressure difference

The Overcooked partner traits directly affect ongoing partner behavior and task efficiency.

In CoordinationGrid with the partner visible, ego can sometimes coordinate by observing the partner's realized movement after t=1, even without knowing z.

That bypass was demonstrated directly in the Stage-C+D pilots and is why hidden-state evidence is necessary before claiming a partner-specific internal model.

---

## 22. Evaluation / implementation bugs and fixes encountered

Key issues that changed the implementation or interpretation:

1. **Task generalization failure.**  
   210 layouts were easy to memorize. D4 augmentation was needed.

2. **Phase/action ambiguity.**  
   A 15-way unmasked head allowed redundant actions. Phase masks made the policy/action semantics explicit and greatly reduced assignment errors.

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
   It improved task performance while messages collapsed toward NONE, so it was excluded from the final design.

9. **"Memory helps" was overinterpreted initially.**  
   Pilot hidden-state shuffling showed that useful recurrence need not mean z-specific memory.

10. **Final post-hoc config mismatch discovered during project-log audit.**  
    Current files named `final_*_causal` / `final_*_bhz` were generated with old/default analysis settings and must be rerun before use.

---

## 23. Current trusted conclusions

At this point the strongest claims supported by the correctly configured final evaluation are:

1. **Explicit communication helps substantially.**
2. **A universal fixed convention is easiest to exploit and works immediately.**
3. **Partner-specific communication also gives a large benefit over action-only.**
4. **Partner-specific performance improves strongly after the first round, consistent with online adaptation.**
5. **Action-only also improves across rounds, so recurrent adaptation is not automatically partner-specific.**
6. **Performance is roughly flat across the five trained z values.**
7. **The final experiment does not test unseen-z generalization.**
8. **The universal-vs-partner-specific gap is confounded with channel reliability.**
9. **A z-specific hidden representation has not yet been established for the final models.**

---

## 24. Files that matter now

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

- message decoder probabilities;
- condition-specific t=0 masks;
- D4 transforms;
- z persistence across rounds;
- layout changes;
- GRU reset semantics;
- JIT/vmap behavior;
- partner hiding.

### Final trained checkpoints

```text
dev/train_logs/final_action_only_seed1_20260925_002403.safetensors
dev/train_logs/final_universal_seed1_20260925_002403.safetensors
dev/train_logs/final_partner_specific_seed1_20260925_002403.safetensors
```

### Trusted final evaluation

```text
dev/train_logs/final_action_only_seed1_20260925_002403_eval.json
dev/train_logs/final_universal_seed1_20260925_002403_eval.json
dev/train_logs/final_partner_specific_seed1_20260925_002403_eval.json
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

### Post-hoc scripts requiring a corrected final rerun

```text
dev/behavior_by_z_by_round.py
dev/causal_memory_control.py
```

Do not treat the existing `final_*_bhz.json` and `final_*_causal.json` outputs as final-condition evidence until the configuration mismatch is fixed and they are regenerated.

---

## 25. Reproduction checklist for a new coding agent

Before rerunning the final experiment, verify:

```text
[ ] final corpus has 1600/200/200 layouts
[ ] augment_symmetries=false
[ ] z pool = [.1,.3,.5,.7,.9]
[ ] rounds_per_episode=20
[ ] hide_partner_until_time=0
[ ] condition is set explicitly for each job
[ ] schedule sanity check says 8000 pairings/sweep
[ ] every z sees all 1600 train layouts once/sweep
[ ] GRU resets only after round 19
[ ] z is absent from observation
[ ] same schedule seed/corpus used across conditions
[ ] final held-out evaluation uses dev/grids_final
[ ] post-hoc analysis explicitly receives the checkpoint's condition
[ ] five-z probe chance is .20, not .25
```

For stronger scientific conclusions, rerun training with multiple independent seeds per condition and rerun the corrected causal-memory / z-probe analyses.
