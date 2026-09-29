# Project log — CoordinationGrid & emergent partner modeling

**Status:** 2026-09-25
**Purpose:** minimal but complete record of the task, data, training setup, major design iterations, and what results are currently trustworthy.

**Scope note (2026-09-25):** the project pivoted in three steps this
session:

1. Retired the earlier three-condition design (action_only / universal /
   partner_specific).
2. Ran a single-condition `action_only` experiment with a scalar latent
   `z` that had no effect on partner behavior. Held-out success was ~0.17
   / 0.14 (val / test) — clear evidence that with a `z`-independent
   partner and no communication, there is no reason for the ego to model
   the partner. This served as a null-control.
3. **Current pivot:** replaced the scalar `z` with a graded **2-D
   capability vector** `(c_R, c_B)` that controls per-goal partner
   movement cooldowns, and introduced an explicit role-allocation action
   at t=0. This creates real pressure to infer the partner's capability
   from motion timing. This log documents the current design and the
   pre-PPO validation showing that inferring the capability profile is
   genuinely useful.

---

## 1. Research question (current scope)

Does a recurrent ego agent, coordinating over 20 rounds with a scripted
partner whose latent **capability profile** `(c_R, c_B)` is fixed but
never observed, learn to infer that profile from the partner's movement
timing and use it to choose the better of two possible role
allocations?

There are two agents:

- **ego:** the only learned agent;
- **partner:** scripted, deterministic BFS navigation, but with a
  per-goal movement cooldown controlled by the capability vector.

The capability vector is fixed for one 20-round partner episode, never
appears in the observation, and only affects the partner's move
cadence.

---

## 2. Task: two agents, two goals, one maze

Each round is a 7×7 gridworld containing:

- walls;
- ego start;
- partner start;
- RED goal;
- BLUE goal.

A round succeeds when the agents occupy **different goals
simultaneously**:

```text
ego=RED  + partner=BLUE
or
ego=BLUE + partner=RED
```

### Round timing

At round-local `t=0`:

- both agents STAY;
- the ego MUST commit to a role via its t=0 action: **ALLOC_RED** or
  **ALLOC_BLUE** (see §3);
- the partner deterministically takes the **complementary** goal at
  t=1.

For `t>=1`:

- ego moves as its policy chooses;
- the partner follows the precomputed shortest-path table toward its
  committed goal, subject to a movement cooldown (§4);
- the partner's goal is fixed for the rest of the round.

### Reward and termination

Per step:

```text
success       -> +1.0
otherwise     -> -0.01
```

A round ends on:

- coordination success, or
- `max_steps=15`.

Twenty rounds per partner episode; `done['__all__']` only fires on the
last round's terminal step; the GRU hidden state carries across
intermediate rounds and only resets between partner episodes.

---

## 3. Observation and action representation

### Observation

Each agent receives the same dict:

```python
{
    "grid": (H, W, 5),        # walls, RED, BLUE, ego pos, partner pos
    "last_allocation": (3,),  # one-hot over {NONE, ALLOC_RED, ALLOC_BLUE}
    "is_t0": scalar,
}
```

The 2-D capability vector `(c_R, c_B)` is **deliberately absent** from
the observation. The only channel through which capability can influence
the ego's behavior is the *timing* of the partner's realized movements.

### Ego action space

The network keeps the flat 15-way action head:

```text
a = 3 * move + alloc
move  ∈ {UP, DOWN, RIGHT, LEFT, STAY}
alloc ∈ {NONE, ALLOC_RED, ALLOC_BLUE}
```

with legal masks:

```text
t=0    :  {STAY+ALLOC_RED, STAY+ALLOC_BLUE}     (ids 13, 14)
t>=1   :  {UP,DOWN,RIGHT,LEFT,STAY} + NONE      (ids 0, 3, 6, 9, 12)
```

At t=0 the ego is **forced** to commit to RED or BLUE (a 2-way
allocation, not 3-way). This is not a symbolic message with
partner-dependent meaning; it is a fixed-semantics role commitment. The
partner takes the complementary goal deterministically:

```text
ego ALLOC_RED  -> partner_goal = BLUE
ego ALLOC_BLUE -> partner_goal = RED
```

At `t>=1` no allocation is legal.

---

## 4. Partner capability vector `(c_R, c_B)` and cooldown mechanic

### Capability pool

Individual per-goal cooldown values:

```text
CAPABILITY_VALUES = {1, 2, 3, 4, 7, 9}
```

`(c_R, c_B)` is drawn from a discrete pool of 30 training pairs and is
**fixed for one 20-round partner episode**. Lower cooldown = faster
partner on that goal.

### Movement cooldown

At each round-local step `t>=1`, if the partner has a committed goal:

- if `partner_move_ctr == 0`, the partner takes one BFS step and
  `partner_move_ctr` resets to `c - 1`, where `c = c_R` if
  `partner_goal == RED` else `c_B`;
- otherwise the partner STAYs and the counter decrements (floored at
  0).

So `c=1` → partner moves every step (fastest); `c=9` → partner moves
at round-local steps `t=1, 10, 19, ...` (slowest given a 15-step
horizon). `partner_move_ctr` resets to 0 at every round boundary.

BFS navigation itself is unchanged — the shortest-path table depends
only on layout, not on capability.

### Training / held-out capability split

The full grid is `{1,2,3,4,7,9} × {1,2,3,4,7,9}` = 36 pairs. Six are
held out and the remaining 30 are used at training:

```text
HELDOUT_CAPABILITY_PAIRS =
    (1, 4), (2, 7), (3, 9), (4, 1), (7, 3), (9, 2)
```

Each of the six individual capability values appears once as `c_R` and
once as `c_B` in the held-out set, and every value also occurs in the
training set at both positions. So the held-out pool tests **unseen
combinations of already-seen individual capability values**, not new
values.

Capability sampling is independent of layout identity by construction —
the scheduler builds balanced `(capability_pair, layout)` sweeps
(§10). This is verified by
`dev/test_capability_env.py::test_capability_independent_of_layout`.

---

## 5. Partner episodes and recurrent memory

A **partner episode** contains:

```text
20 rounds
```

Within one partner episode:

- capability `(c_R, c_B)` stays fixed;
- each round uses a new layout;
- partner goal resets each round;
- round-local time resets to 0;
- ego's GRU hidden state **does not reset**;
- `partner_move_ctr` resets to 0 at each round boundary (it is
  round-local state, not episode-scoped).

Only after round 19 does `done['__all__'] = True` fire and the
recurrent state reset.

GAE and value bootstrapping use this partner-episode `done`, so
intermediate round boundaries are not RL terminals.

---

## 6. Network architecture

Only ego is learned. The partner remains scripted.

The recurrent actor-critic is adapted from the JaxMARL / Overcooked
recurrent PPO code and is unchanged from the previous build except
that the message-embedding path now consumes `last_allocation`
(same 3-slot one-hot).

### Encoder

Grid:

```text
(H,W,5) -> CNN -> 64-d
```

CNN stack:

```text
128@1×1, 128@1×1, 8@1×1, 16@3×3, 32@3×3, 32@3×3
-> flatten -> Dense(64) -> ReLU
```

Allocation:

```text
last_allocation (3) -> Dense(8) -> ReLU
```

Concatenate and project to `GRU_HIDDEN_DIM = 128`, layer-norm, then
GRU.

### Heads

Actor:

```text
GRU -> Dense(128), ReLU -> Dense(15 logits) -> legality mask -> Categorical
```

- t=0 legal ids: `{13, 14}` (STAY+ALLOC_RED, STAY+ALLOC_BLUE).
- t>=1 legal ids: `{0, 3, 6, 9, 12}` (movement, NONE).

Critic:

```text
GRU -> Dense(128), ReLU -> scalar value
```

---

## 7. PPO learning rule

Recurrent PPO with GAE. Final hyperparameters (unchanged from previous
build):

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

Training is JAX/JIT-based and vectorized across 256 parallel
environments. First PPO run under the new design has now been launched
and its held-out numbers are reported in §14b.

---

## 8. Layout generation (unchanged)

Layouts come from the balanced `dev/grids_final` corpus built earlier:

```text
2000 base layouts × 1 D4 transform each
= 2000 transformed layouts
train = 1600, val = 200, test = 200
augment_symmetries = false
```

Details of the generator, D4 handling, and coordinate footguns are as
before and unchanged by this pivot.

---

## 9. Held-out layout corpus (unchanged)

```text
dev/grids_final/layouts/train  (1600)
dev/grids_final/layouts/val    (200)
dev/grids_final/layouts/test   (200)
```

Each split is disjoint. Splits are approximately balanced across D4
transforms (exact balance only holds over the full 2000).

---

## 10. Balanced `(capability_pair, layout)` training schedule

Every capability pair sees every training layout exactly once per sweep.

Scheduler:

```text
baselines/IPPO/sweep_scheduler.py
```

One balanced sweep contains:

```text
30 capability pairs × 1600 layouts = 48000 (cap, layout) pairings
```

For each capability pair:

1. independently shuffle all 1600 layout indices;
2. chunk them into groups of 20 (one partner episode);
3. aggregate across capability pairs, then shuffle episode order.

Therefore:

```text
1600 / 20 = 80 partner episodes per cap
80 × 30    = 2400 partner episodes per sweep
```

Final scheduler settings:

```text
N_SWEEPS      = 256
SCHEDULE_SEED = 2026
```

The 256 vectorized workers start at different sweep boundaries. At the
end of a partner episode, the trainer advances that worker's cursor and
`env.reset_from_schedule(capability, layout_seq)` injects the next
planned episode.

Capability sampling is independent of layout identity by construction.
This is checked by `dev/test_capability_env.py`.

---

## 11. Manual scheduled reset (unchanged)

`LogWrapper` autoreset is bypassed; the trainer calls `env.step_env`
directly and, on `done['__all__']`, advances the per-slot schedule
cursor and calls `reset_from_schedule`. Episode return / length
bookkeeping lives in the trainer.

---

## 12. Chronology (very compressed)

The lengthy Stage A / B / C+D / three-condition / K=3 history from the
previous log is superseded by the current design. The one takeaway
worth carrying forward:

> The previous `action_only` result (val 0.17, test 0.14) with a
> `z`-independent partner and no communication is what motivated the
> capability-vector pivot. It is a null control: nothing to infer,
> nothing to allocate, and no communication → the ego cannot benefit
> from being recurrent.

The full pilot history is preserved in git (see `git log
project_log_revised.md`) but not repeated here.

---

## 13. Current experiment specification

```text
communication_condition : "action_only" (task-level allocation only; no
                          partner-dependent symbolic channel)
rounds_per_episode      : 20
max_steps (per round)   : 64      # widened from 15 so (9,9), (7,7) etc.
                                  # are feasible on ~91% of layouts.
partner_capability_pairs: 30 training pairs (§4)
augment_symmetries      : false
hide_partner_until_time : 0
layouts (train/val/test): dev/grids_final/layouts/{train,val,test}
step_penalty            : 0.01
success_reward          : 1.0
```

PPO also bumped: `NUM_STEPS = 256` (from 128) so a single rollout still
spans a few full 64-step rounds under the new horizon.

Held-out capability pool for evaluation:

```text
HELDOUT_CAPABILITY_PAIRS = (1,4), (2,7), (3,9), (4,1), (7,3), (9,2)
```

Every evaluation is run twice per layout split:

1. **familiar capabilities**: the 30 training pairs;
2. **held-out capabilities**: the 6 unseen combinations above.

---

## 14. Pre-PPO validation (headline results)

Analytical script: `dev/capability_validation.py`.

For each `(layout, capability_pair)` we compute the completion time of
each of the two allocations (ego→RED or ego→BLUE), assuming
shortest-path ego navigation and BFS + cooldown partner navigation:

```text
call_success(alloc) = max(
    1 + BFS(ego,     ego_goal),
    2 + (BFS(partner, partner_goal) - 1) * partner_cooldown
)
success(alloc)      = (call_success <= max_steps)
```

### At the original max_steps = 15 (retired)

The initial run at the smaller horizon showed a large **success-rate**
gap between oracle and partner-blind:

| metric | val (familiar) | val (held-out) | test (familiar) | test (held-out) |
|---|---:|---:|---:|---:|
| flip fraction | 0.925 | 0.870 | 0.930 | 0.845 |
| oracle success | 0.782 | 0.843 | 0.793 | 0.865 |
| partner-blind success | 0.705 | 0.705 | 0.710 | 0.710 |
| oracle − blind (success) | +0.077 | +0.138 | +0.083 | +0.155 |

The downside: (7,7), (9,9), (7,9), (9,7) etc. were near-infeasible in
15 steps (oracle rate 0.10–0.13). Those extreme profiles dragged the
familiar-pool mean down, and made the pool composition confound
success comparisons across slices.

### At the current max_steps = 64

Widening the horizon to 64 makes every capability pair feasible on
≥ 91% of layouts, so **success saturates near 1.0** for both oracle
and partner-blind. The discriminative signal shifts from *whether*
the round succeeds to *how fast* it succeeds — i.e. from success rate
to completion time and per-round reward.

| metric | val (familiar) | val (held-out) | test (familiar) | test (held-out) |
|---|---:|---:|---:|---:|
| flip fraction | 0.925 | 0.870 | 0.930 | 0.850 |
| oracle success | 0.997 | 1.000 | 0.998 | 1.000 |
| partner-blind success | 0.992 | 0.992 | 0.993 | 0.993 |
| oracle avg completion time | 12.04 | 10.34 | 11.80 | 10.22 |
| partner-blind avg completion time | 17.23 | 17.23 | 16.65 | 16.65 |
| **oracle expected reward / round** | 0.875 | 0.897 | 0.878 | 0.898 |
| **partner-blind expected reward / round** | 0.815 | 0.815 | 0.824 | 0.824 |
| **oracle − blind reward / round** | **+0.060** | **+0.081** | **+0.055** | **+0.074** |

Reward here uses the training reward model
(`success_reward=1.0`, `step_penalty=0.01`).

Per-capability oracle timing on the held-out pool (val):

| cap (c_R, c_B) | oracle success | oracle avg time |
|---|---:|---:|
| (1, 4) | 1.000 | 7.28 |
| (2, 7) | 1.000 | 10.20 |
| (3, 9) | 1.000 | 13.26 |
| (4, 1) | 1.000 | 7.41 |
| (7, 3) | 1.000 | 13.33 |
| (9, 2) | 1.000 | 10.54 |

### Interpretation

1. **Optimal allocation remains capability-dependent** on ~87–93% of
   layouts. The median regret for the wrong allocation is now 9 steps
   (familiar) / 13 steps (held-out) — larger than at max_steps=15
   because slow-partner profiles that previously *failed* under the
   wrong allocation now merely *take much longer*.
2. **Success is no longer the right discriminator** at this horizon
   (both oracle and blind ≥ 0.99). Expected per-round reward is the
   right one to track. On held-out capabilities the reward gap is
   **+0.07–0.08 per round**, i.e. **~+1.5 per 20-round partner
   episode**. That's smaller than the max_steps=15 gap (~+3/episode
   under the retired horizon) but still a real gradient.
3. The oracle-vs-blind reward gap is smaller on the training pool
   (+0.055 to +0.060) than on held-out (+0.074 to +0.081) for the
   same pool-composition reason as before: the training pool contains
   symmetric diagonals like `(1,1), (2,2), (3,3), (7,7), (9,9)` where
   allocation is a no-op.
4. **No more infeasibility cliff**: every held-out pair now has
   oracle success = 1.0, and even `(3, 9)` and `(7, 3)` complete in
   ~13 steps under the correct allocation — well within the 64-step
   budget.

Outputs:

```text
dev/train_logs/capability_validation_{train,val,test}_ms64.json
```

**Conclusion:** the design still passes the pre-PPO gate, but the
benchmark to compare a trained policy against has moved from
success-rate ceiling to expected-reward ceiling:
- Oracle expected reward per round: ~0.88 (familiar), ~0.90 (held-out).
- Blind expected reward per round: ~0.82.
- A well-trained policy should approach ~0.88–0.90 per round.
- A "partner-blind" learned policy will top out near ~0.82.

---

## 14b. First PPO run under the new design (max_steps = 15, retired)

Slurm job 17608947 (tag `capability_seed1_20260925_163928`), single
seed, 60M timesteps, wall-clock ~15 min on one 80G GPU. Config exactly
as §13 **except** max_steps=15 and NUM_STEPS=128 (the retired settings).

Held-out numbers (round-level success on 320 rounds per capability
pair; 16 partner episodes × 20 rounds):

| split | slice | round success | mean ep return |
|---|---|---:|---:|
| val  | familiar (30 caps) | 0.372 | 4.90 |
| val  | held-out (6 caps)  | **0.406** | 5.61 |
| test | familiar (30 caps) | 0.367 | 4.81 |
| test | held-out (6 caps)  | **0.407** | 5.65 |

For reference, the analytical ceilings from §14:

| split | slice | oracle | partner-blind | PPO |
|---|---|---:|---:|---:|
| val  | familiar | 0.782 | 0.705 | 0.372 |
| val  | held-out | 0.843 | 0.705 | 0.406 |
| test | familiar | 0.793 | 0.710 | 0.367 |
| test | held-out | 0.865 | 0.710 | 0.407 |

Selected per-capability round-success on the held-out pool (test
split):

| cap (c_R, c_B) | oracle | PPO |
|---|---:|---:|
| (1, 4) | 1.000 | 0.572 |
| (2, 7) | 0.900 | 0.356 |
| (3, 9) | 0.700 | 0.262 |
| (4, 1) | 1.000 | 0.575 |
| (7, 3) | 0.695 | 0.266 |
| (9, 2) | 0.895 | 0.412 |

Per-round-index success (test/held-out):

```text
r0=0.29 r1=0.42 r2=0.35 r3=0.42 r4=0.33 ... r19=0.51
```

Cross-round improvement is real but modest.

### Interpretation of the first PPO run

1. **PPO is well below both the analytical oracle and the analytical
   partner-blind ceiling** on all four (split, slice) cells (roughly
   35–45 pp below oracle, 30–35 pp below blind). Those ceilings
   assume perfect shortest-path ego navigation on the held-out
   layouts; PPO must learn both allocation *and* navigation, so
   direct comparison to the ceilings is loose. Still, the fact that
   PPO underperforms even the blind ceiling means the current policy
   is leaving substantial value on the table — likely mostly in
   navigation, not allocation.
2. **Held-out ≥ familiar** on both splits (0.406 vs 0.372 on val;
   0.407 vs 0.367 on test). Not a mistake: the analytical held-out
   oracle is also higher than the training oracle (0.84 vs 0.78 on
   val) because the held-out pool by construction contains more
   asymmetric profiles like `(1,4)` and `(4,1)`, on which the correct
   allocation is easier to identify from partner motion.
3. **Per-capability spread matches the physics.** Fast-partner-both-
   ways profiles like `(1,1)` reach 0.62; extreme-slow profiles like
   `(9,9)`, `(9,7)`, `(7,7)` are 0.10–0.13 because even the correct
   allocation often can't complete within 15 steps.
4. **Not yet a real capability-inference result.** With PPO well below
   the partner-blind ceiling, we cannot yet distinguish "the policy
   uses capability information" from "the policy has learned a
   partner-agnostic reasonable-navigation prior". A hidden-state →
   capability probe or a capability-shuffle intervention (§18) is
   required to make claims about internal partner modeling.
5. **Concrete next steps before drawing conclusions:** (a) rerun with
   several seeds to see the noise band on these numbers; (b) run the
   validation-style ORACLE and PARTNER-BLIND baselines against the
   trained ego (i.e. force the ego to pick the analytical optimum vs
   a fixed alloc, and measure end-to-end success under the SAME
   navigation) to isolate allocation quality from navigation quality;
   (c) add the hidden-state probe.

Files:

```text
dev/train_logs/capability_seed1_20260925_163928.safetensors
dev/train_logs/capability_seed1_20260925_163928_eval.json
```

## 14c. Second PPO run under max_steps = 64 (in flight)

Slurm job 17611364 (tag `capability_ms64_seed1_20260925_214638`),
single seed, 60M timesteps, config as §13 (max_steps=64, NUM_STEPS=256).
Results to be filled in when the run completes.

---

## 15. Tests

Focused test file: `dev/test_capability_env.py`.

It verifies specifically:

- capability split constants: 30 training / 6 held-out, disjoint, and
  every individual value in `{1,2,3,4,7,9}` appears in both;
- t=0 legal-mask = `{STAY+ALLOC_RED, STAY+ALLOC_BLUE}`;
- t>=1 legal-mask = `{move+NONE}`;
- `state.capability` is `(2,)` int32 and matches a pool member;
- `obs['agent_0']` has no `capability` key, and get_obs output is
  bit-identical for two envs differing only in capability;
- capability stays fixed across all 20 rounds of an episode;
- partner cadence when pursuing RED matches `c_R` exactly (verified
  via `partner_move_ctr` sequence, layout-independent);
- symmetric check for `c_B` on BLUE;
- both cadences co-exist within one capability profile;
- capability sampling is roughly independent of layout (marginals
  balanced within ~15%);
- `ALLOC_RED → partner_goal = BLUE`, `ALLOC_BLUE → partner_goal = RED`;
- jit + vmap don't crash, and info dict carries capability;
- capability with value 0 raises.

All checks currently pass.

---

## 16. Files that matter now

### Environment

```text
jaxmarl/environments/coordination_grid/coordination_grid.py
jaxmarl/environments/coordination_grid/__init__.py
```

Exports include `CAPABILITY_VALUES`, `TRAINING_CAPABILITY_PAIRS`,
`HELDOUT_CAPABILITY_PAIRS`, `ALL_CAPABILITY_PAIRS`, `Allocations`,
`bfs_distance_map`.

### Layout generation (unchanged)

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
dev/test_capability_env.py
```

### Pre-PPO validation

```text
dev/capability_validation.py
dev/train_logs/capability_validation_{train,val,test}.json
```

### Trained checkpoint under the new design

```text
dev/train_logs/capability_seed1_20260925_163928.safetensors
dev/train_logs/capability_seed1_20260925_163928_eval.json
```

### Previous-build artifacts (retained but stale)

```text
dev/train_logs/final_action_only_seed1_20260925_142734.safetensors
dev/train_logs/final_action_only_seed1_20260925_142734_eval.json
```

These come from the retired scalar-`z` `action_only` build; they are
NOT compatible with the current env (`state` schema changed).

---

## 17. Reproduction checklist for a new coding agent

Before rerunning:

```text
[ ] final corpus has 1600/200/200 layouts
[ ] augment_symmetries=false
[ ] partner_capability_pairs is the 30 training pairs
[ ] EVAL_HELDOUT_CAPABILITY_PAIRS is the 6 held-out pairs
[ ] rounds_per_episode=20
[ ] hide_partner_until_time=0
[ ] communication_condition="action_only"
[ ] scheduler sanity check passes with 48000 pairings/sweep
[ ] every capability pair sees all 1600 train layouts once/sweep
[ ] GRU resets only after round 19
[ ] capability is absent from observation
[ ] final held-out evaluation uses dev/grids_final
[ ] pre-PPO validation report shows oracle ≫ partner-blind on held-out cap pool
```

For stronger scientific conclusions, run multiple independent seeds
per condition and add a hidden-state-to-capability probe once a
trained policy exists.

---

## 18. Open questions / next steps

1. **PPO run.** Now that the validation gate is passed, launch a full
   60M-timestep PPO run under the new capability env. Compare the
   learned policy's held-out success to the oracle ceiling on both
   familiar and held-out capability pools.
2. **Movement inference test.** After training, hold out the
   partner-position observation channel after the first few steps and
   check whether the policy can still commit correctly at t=0 based on
   *later* rounds' observations (via GRU memory).
3. **Hidden-state probe.** Train a small MLP from GRU hidden state
   (post round 5, say) to predict the capability pair. Compare to
   chance (1/30 for the training pool, 1/6 for held-out).
4. **Larger capability grid.** If the current pool is too easy, extend
   `CAPABILITY_VALUES` and re-generate the split.
