# CoordinationGrid — Final Experiment (Overcooked replication in a gridworld)

Last refactor: **2026-09-29**.
Last documentation update: **2026-10-01** (completed five-seed experiment
and representation analysis; published-paper/reference-code fidelity audit).

## Current status

- All **20 policies** have been trained: four conditions × seeds 1–5.
  Resolved configs, in-trainer evaluation JSONs, and standalone evaluation
  summaries were committed in `120f4b2b` (2026-09-30).
- Standalone evaluation provides 24 training profiles × 20 rollouts and
  22 novel profiles × 20 rollouts per policy. Existing local HDF5s were
  reused for the representation analysis; no rollouts were regenerated.
- Representation analysis is complete for **all 15 recurrent policies**:
  two linear cooldown probes, absolute-timestep and round curves,
  random and shuffled-label diagnostics, five-policy-seed uncertainty,
  behavior-selected checkpoints, and final-50-state UMAPs.
- The observed result is mixed. Multi-partner RNNs show better mean
  allocation behavior than the MLP and single-partner controls and
  stronger decoding than no-influence RNNs. However, single-partner RNNs
  also encode capabilities strongly: RED decoding is stronger on average
  than in multi-partner RNNs, and BLUE decoding is comparable. The full
  predicted diversity-specific representation advantage is **not observed**.

Detailed methods, validation, numerical results and figures are in
[the representation report](analysis/representation_results/README.md).
Completing the implementation and analyses does not imply that every
scientific replication criterion has been met.

The replication-fidelity audit below distinguishes the **published NeurIPS
paper**, the **released Overcooked implementation**, and the **current grid
experiment**. They are not interchangeable specifications. The current
experiment is an adaptation of Experiment 1, with material protocol
differences beyond changing the task.

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

Completed: **5 independent training seeds per condition**, seeds 1–5.
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
1000-layout corpus. The materialized schedule is long enough
(`N_EPS_TOTAL = 131072`) that a training run does not wrap in practice.

There is **no** exhaustive capability × layout balancing. The old
`sanity_check_schedule` is kept as an alias to `summarize_schedule` which
just reports coverage stats.

## Active files

The following entry points and supporting files are wired into the
current experiment. Earlier stage/sweep and Overcooked plotting scripts
marked `SUPERSEDED` remain retired; the new representation entry point
does not reactivate them. Generated data and generic supporting modules
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
      tests/analysis/test_representation_analysis.py    # 13 representation regression tests

    Evaluation:
      analysis/evaluate_partner_modelling.py

    Representation analysis:
      analysis/representation_analysis.py               # current grid-specific entry point
      requirements/representation.txt                  # standalone analysis dependencies
      analysis/representation_results/README.md         # methods, results, interpretation
      analysis/representation_results/                  # numerical tables, metadata, figures, NPZs

## Checkpoints, configs and data provenance

The committed full experiment includes, for each of the 20 policies:

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
        --out_prefix dev/eval_out/<condition>_seed<seed> \
        --save_hidden               # only meaningful for MODEL_TYPE=rnn

Outputs a per-checkpoint HDF5 rollout record for both the train and test
capability slices, plus a `{prefix}_summary.json` with headline metrics.
The bulk launcher `bash/eval_all_checkpoints.sh` loads saved configs,
uses evaluation seed 12345 and 20 repetitions/profile, and requests
hidden states only for RNN checkpoints. It skips prefixes with an
existing summary; if a rollout file is missing or malformed despite that
summary, rerun the evaluator explicitly for the affected checkpoint
rather than assuming the bulk launcher repaired it.

The primary behavioral metric is **fraction_optimal_allocation** —
computed per t=0 sample using the same analytical
`completion_time`/`_reward_from_time` primitive as
`capability_validation.py`. For each (layout_idx, capability) the
optimal alloc is `argmax(reward_A, reward_B)`; ties (equal analytical
rewards) are excluded from the optimal-fraction accuracy but count with
regret = 0 in `allocation_regret_mean`. Metrics reported per slice:

    round_success_rate
    mean_ep_return
    mean_reward_per_round
    mean_successful_completion_time
    fraction_optimal_allocation_overall
    fraction_optimal_allocation_by_round        # main adaptation curve
    allocation_regret_mean
    allocation_regret_by_round
    partner_assigned_red_given_relative_cap     # secondary/qualitative

`fraction_optimal_allocation_by_round` is the main adaptation signal:
the design predicted improvement for the diverse-partner RNN and little
adaptation for the MLP and single-partner controls. Actual results are
reported below; this prediction is not enforced by the analysis.

**No-influence metric semantics:** optimal-allocation fraction and regret
are calculated from the ego's chosen t=0 allocation, while the actual
partner assignment is controlled by round parity. These are diagnostics
of the ego's hypothetical allocation choice in that condition, not
measures of an allocation it causally imposed. Round success and episode
return still describe realized behavior. The relative-capability
assignment diagnostic uses the actual `partner_assignment` field.

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
`success`, `capability_index_per_ep`, and `capability_pool`. RNN scans have
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

## Completed five-seed behavioral evaluation

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

The last two columns for no-influence have the hypothetical-choice
semantics described above. High success alone does not demonstrate
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

## Completed representation analysis

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

## Scientific replication criteria and current conclusion

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

## Replication-fidelity audit (2026-10-01)

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

Partner switching mid-episode, causal capability-shuffle interventions
(distinct from the completed shuffled-label probe diagnostic),
observation blindness / local visibility, communication conditions,
larger capability grids, unseen-layout generalization, layout
augmentation, additional procedural environments, more elaborate causal
representation interventions — these are follow-ups. They must not
block or complicate the core replication.
