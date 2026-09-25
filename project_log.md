# Project log — CoordinationGrid & emergent partner modeling

A minimal record of every dataset, algorithm, and design choice behind the
final experiment.

---

## 1. What the project is

Two agents live on a small grid: a **learning agent** ("ego") and a
**scripted partner**. Each *round*, both agents pick a color (RED or
BLUE); if they pick the same one the round scores a point. The partner's
color choice depends on a hidden parameter `z ∈ [0, 1]` (a Bernoulli
bias) that the ego cannot observe. The ego has to work out who its
partner is from behavior.

We stack rounds into **partner episodes** of 20 rounds. `z` is sampled
once at the start of the episode and held fixed; the layout swaps every
round. So over 20 rounds the ego can accumulate evidence about *this*
partner, and (in principle) specialize its behavior.

We ran the same recurrent PPO agent under three **communication
conditions** that vary what the ego and partner can say to each other,
and asked: does the policy build a real partner model in its hidden
state?

---

## 2. Environment (`jaxmarl/environments/coordination_grid/`)

One JAX-friendly Gymnax-style env file. Highlights:

**Action space (ego).** 15 flat actions = `movement (5) × message (3:
NONE, M0, M1)`. We tried a *factored* two-headed actor first (as
Overcooked does), then replaced it with a **single categorical head +
legality mask**: at t=0 of each round only "stand still + send message"
actions are legal; at t≥1 only "movement + send NONE". The mask sets
illegal logits to `-inf`, distrax handles sampling/entropy. Simpler
mask logic, cleaner entropy metrics, no drift between env and policy
about what "message step" means.

**Observation.** grid tensor + `last_message` + `is_t0`. **`z` is *not*
in the obs.** The ego has to infer it from behavior/messages.

**State machine.**
- `reset` samples a random `z` from `partner_z_values` and a random
  20-layout sequence.
- Each round: t=0 → ego picks a message; partner samples goal per the
  condition (see below); ego then has `max_steps=15` moves to reach it.
- Round ends on success or timeout. The env swaps in the next layout in
  `episode_layout_seq` *without* resampling `z`.
- Episode ends after `rounds_per_episode=20`.

**Three communication conditions.** (This semantics was our third
iteration; earlier versions were either trivial or unlearnable.)

| condition | partner behavior | what the ego needs `z` for |
|---|---|---|
| `action_only` | 50/50 goal, ignores ego message | nothing — control |
| `universal` | `M0 → RED, M1 → BLUE` deterministically; goal = Bernoulli(z) | nothing for action; but message *stream* still Bernoulli(z) |
| `partner_specific` | `P(RED \| M0, z) = z` | to interpret every message |

Only `partner_specific` requires a partner-specific model to act well.

**D4 augmentation** (identity + 3 rotations + 4 reflections) is
implemented in the env; **off** in the final experiment because the
corpus is already balanced-symmetrized.

**`hide_partner_until_time`** blanks the partner-position channel for K
early steps. Wired but left at 0 in the final run.

---

## 3. Layout corpus (`dev/build_final_corpus.py`)

We iterated corpus size several times.

| stage | corpus | purpose |
|---|---|---|
| Pilot A | 1 fixed layout, z=1 | prove PPO can learn at all |
| Pilot B | 210 layouts + online D4 | multi-layout generalization |
| Pilot C+D | same, multi-round + z pool | rounds/persistent-GRU sanity |
| **Final** | **2000 unique layouts × balanced D4 → 1600/200/200** | this experiment |

**Final corpus.** 2000 base layouts from
`env_generator.generate_envs(n=2000, master_seed=2026, wall_density=[0.15, 0.6])`;
each layout gets one D4 transform (250 layouts per group element);
sha1-fingerprinted for uniqueness; split into 1600 train / 200 val /
200 test.

**Coord-convention footgun.** `env_generator` uses (row, col); the env
uses (x, y). D4 helpers `_sym_position` / `_sym_wall` convert at the
boundary. Fingerprint uniqueness passed once this was right.

---

## 4. Training algorithm
(`baselines/IPPO/ippo_rnn_coordination_grid.py`)

Backbone: recurrent PPO adapted from JaxMARL's
`ippo_rnn_overcooked_v2.py`. Same optimizer, GAE, PPO objective,
minibatching. What we changed:

### 4a. GRU persists across rounds within an episode

The Overcooked baseline resets the RNN carry at every env reset. We
don't. The `done` signal that zeros the carry is `done["__all__"]`,
which fires only when a *20-round episode* ends. Between rounds `r` and
`r+1` the carry flows through. This is the mechanism that could carry
partner information from one round into the next.

### 4b. Manual auto-reset (LogWrapper / AutoResetWrapper removed)

Both wrappers auto-reset a slot with a fresh random `(z, layout)` at
terminals. We needed to inject a *pre-computed* `(z, layout_seq)` from a
balanced schedule instead. Rather than fight the wrappers we removed
them and re-implemented their bookkeeping directly in `_env_step`:

- per-slot `episode_return_acc`, `episode_length_acc` (replaces LogWrapper);
- on `done_all` in a slot, advance a per-slot cursor into the schedule
  and call `env.reset_from_schedule(z, layout_seq)` instead of
  `env.reset(key)`;
- downstream metrics look identical to what LogWrapper produced.

Time-consuming to unpick (wrappers do many small useful things), but
the result is a trainer where every worker's episode content is
deterministic given the schedule seed.

### 4c. Balanced (z × layout) sweep scheduler (`sweep_scheduler.py`)

- **One sweep** = 5 z × 1600 layouts = 8000 pairings.
- Grouped into 20-round episodes: 80 episodes per z, 400 per sweep.
- In a sweep, every `(layout, z)` appears exactly once. Episode order
  shuffled so consecutive episodes don't cluster by z.
- `n_sweeps=256` sweeps concatenated to form the full schedule
  (~102400 total episodes).
- Worker `w` starts at sweep `w % n_sweeps`, so the 256 vmap slots in a
  batch span 256 different sweeps at once. This defuses "everyone
  doing z=0.1 this update" batch-level correlation.
- `sanity_check_schedule` asserts the invariants at startup.

The scheduler is what makes per-z eval numbers apples-to-apples: every
condition sees every `(layout, z)` pair the same number of times in the
same order.

### 4d. Condition-specific t=0 action mask

Two module-level constants (`ACTION_MASK_T0_ACTION_ONLY` — a
single-action mask; `ACTION_MASK_T0_VERBAL` — three legal messages) live
in the env. The actor picks the right one by condition. Env and policy
stay in sync by construction.

### 4e. Per-z evaluation

`evaluate_policy` runs partner episodes on the val/test corpora with
per-slot `z` overrides so we can report success broken out by `z`,
which matters because the three conditions have very different
z-sensitivity profiles.

---

## 5. Hyperparameters (final run, same across all three conditions)

- `NUM_ENVS=256`, `NUM_STEPS=128`, `UPDATE_EPOCHS=4`, `NUM_MINIBATCHES=8`
- `LR=5e-4`, `MAX_STEPS=15`/round, `ROUNDS_PER_EPISODE=20`
- `TOTAL_TIMESTEPS=60_000_000`
- `partner_z_values=[0.1, 0.3, 0.5, 0.7, 0.9]`
- `augment_symmetries=false`, `hide_partner_until_time=0`
- `N_SWEEPS=256`, `SCHEDULE_SEED=2026`
- SLURM job array, one seed per condition, one 80 GB GPU each

---

## 6. Headline results

Overall round success on val / test:

| condition | val | test |
|---|---|---|
| `partner_specific` | **0.594** | **0.577** |
| `universal` | 0.425 | 0.424 |
| `action_only` | 0.355 | 0.329 |

`partner_specific` beats the other two by ~17 pp; `universal` beats
`action_only` by ~7-9 pp.

---

## 7. Post-hoc analyses

### 7a. Behavior × z × round (`dev/behavior_by_z_by_round.py`)

Per (z, round) we log the ego's t=0 message distribution.
- `action_only`: message distribution is essentially z-invariant
  (correct — no signal to learn from).
- `universal`: ego collapses to `M0 = 1.000` for every z and every
  round. Fully deterministic.
- `partner_specific`: ego varies its message with `z` and with round.
  Max-minus-min-across-z is 10-25 pp on most rounds. Policy is doing
  z-conditional behavior.

### 7b. Causal memory interventions (`dev/causal_memory_control.py`)

Four rollout modes:
- **`normal`** — carry flows normally.
- **`round_reset`** — zero the GRU carry at every round boundary. Tests
  whether *any* cross-round memory helps.
- **`shuffle`** — at each round boundary, swap this slot's carry with
  another slot of the *same* z. Tests whether memory is
  episode-specific or just z-specific.
- **`cross_z_shuffle`** — same as shuffle, but the target slot is
  guaranteed to have a *different* z
  (`target_z = (own_z + offset) % Z`, `offset ∈ [1, Z)`). The strong
  test for a "z-specific partner model": if partner_specific stores a
  z-belief in its carry, transplanting a z=0.2 carry into a z=0.8
  episode should actively mislead it.

### 7c. Ridge z-probe

Per round, fit ridge regression `hidden_state → z_true` (standardized,
alpha=1.0, bias unregularized). Report snap-to-pool test accuracy vs
chance = `1/|z_pool_eval|` = 0.25 for the 4-way eval pool.

We used ridge, not OLS: hidden dim (128) is on the same order as sample
size (n≈90 train), and OLS gave test R² < -40 routinely. Ridge gave
usable numbers.

---

## 8. Analysis results

### 8a. Memory helps two of three conditions

Δ vs `normal` under `round_reset`:

| condition | val | test |
|---|---|---|
| `action_only` | **-0.176** | **-0.159** |
| `partner_specific` | **-0.153** | **-0.105** |
| `universal` | -0.016 | -0.035 |

Expected: `universal` doesn't need memory (the message unambiguously
determines the goal *within* the current round). Both other conditions
need memory.

### 8b. Shuffle interventions are null, in both flavors

Δ vs `normal`:

| condition | Δ shuffle | Δ cross_z_shuffle | Δ(cross_z − shuffle) |
|---|---|---|---|
| `action_only` (val / test) | -0.007 / -0.007 | -0.014 / +0.011 | -0.007 / +0.017 |
| `universal` (val / test) | +0.001 / -0.006 | -0.002 / -0.004 | -0.003 / +0.003 |
| `partner_specific` (val / test) | -0.008 / -0.004 | -0.003 / +0.015 | **+0.005 / +0.019** |

**Surprising:** `partner_specific` is *not* especially harmed by
cross-z shuffle. If anything, cross-z is marginally *better* than
same-z shuffle on both splits. The strong "z-specific partner model in
hidden state" prediction is not supported.

### 8c. Ridge z-probe (VAL, mean over rounds; chance = 0.25)

| condition | normal | round_reset | shuffle | cross_z_shuffle |
|---|---|---|---|---|
| `action_only` | 0.277 | 0.245 | 0.260 | 0.253 |
| `partner_specific` | 0.263 | 0.227 | 0.287 | 0.263 |
| **`universal`** | **0.342** | 0.238 | **0.349** | 0.329 |

**Also surprising:** `universal`'s hidden state contains *more*
z-information than `partner_specific`'s. Explainable — under
`universal` the partner's message stream is Bernoulli(z), so watching
messages leaks z into the carry even though `universal` doesn't need z
for action. `partner_specific` does not concentrate a clean z
representation.

### 8d. Interpretation

- Ranking (`partner_specific` > `universal` > `action_only`) is stable
  and real.
- Memory *helps* under `action_only` and `partner_specific`, but the
  memory is not a persistent per-episode belief about `z`. Both shuffle
  modes leave performance intact, which means the useful content of
  the carry can be replaced with *any* other partner's carry without
  harm.
- Best guess: the useful memory is episode-generic scaffolding (layout
  history, running message-token counts, general navigation state)
  that survives being swapped. `partner_specific` re-derives its
  message-interpretation per round from within-round evidence.
- `round_reset` still hurts because zero-carry breaks the
  scale/distributional properties the downstream network was trained
  under, not because it erases a specific belief.

---

## 9. Places we iterated (short list, for future me)

1. **Success metric.** Original was per-step; had to switch to
   per-episode to match paper convention.
2. **Factored actor head → masked single head.** Cleaner masking,
   cleaner entropy split.
3. **LogWrapper / AutoResetWrapper removed.** Needed deterministic
   per-slot `(z, layout_seq)` injection. Rebuilt tracking by hand.
4. **210 → 2000 layouts with balanced D4.** Fingerprint check caught
   duplicates from the D4 sampler.
5. **Communication-condition semantics rewritten twice.** Third
   design (documented above) is what shipped.
6. **Balanced sweep scheduler.** Per-z eval numbers were noisy under
   random sampling.
7. **Ridge probe (not OLS).** OLS on `n≈90` in `d=128` is meaningless.
8. **`cross_z_shuffle` added late** as the strict diagnostic mode.
   Its null result *is* the informative finding.

---

## 10. Files that matter

- Env: `jaxmarl/environments/coordination_grid/coordination_grid.py`
- Trainer: `baselines/IPPO/ippo_rnn_coordination_grid.py`
- Config: `baselines/IPPO/config/ippo_rnn_coordination_grid.yaml`
- Scheduler: `baselines/IPPO/sweep_scheduler.py`
- Corpus builder: `dev/build_final_corpus.py`
  → `dev/grids_final/layouts/{train,val,test}`
- SLURM launcher: `bash/train_final_experiment.sh`
- Analyses:
  - `dev/behavior_by_z_by_round.py`
  - `dev/causal_memory_control.py`
- Trained params:
  `dev/train_logs/final_{action_only,universal,partner_specific}_seed1_20260925_002403.safetensors`
- Analysis outputs:
  `dev/train_logs/final_{...}_{causal,bhz}.json`
