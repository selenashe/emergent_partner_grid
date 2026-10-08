# Partner representations and recurrent dynamics

An additive, executed analysis pipeline for the CoordinationGrid RNN policies.
The supplied [specification](rnn_partner_dynamics_codex_prompt.md) is retained
verbatim. Historical checkpoints, frozen sources and rollout files are read-only.

## Run in this checkout

Use the existing experiment environment; the base Python lacks the dependencies.
From the repository root:

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
PYTHON=/nlp/scr/jshe/miniconda3/envs/emergent_partner_model/bin/python
$PYTHON -m eval.partner_dynamics --dry-run
$PYTHON -m eval.partner_dynamics --phase geometry --smoke
$PYTHON -m eval.partner_dynamics --phase collect --smoke
$PYTHON -m eval.partner_dynamics --phase dynamics --smoke
$PYTHON -m eval.partner_dynamics --phase replay --smoke --protocol v2
$PYTHON -m eval.partner_dynamics --phase all
```

`--phase` accepts `audit`, `geometry`, `collect`, `replay`, `validate`, `controls`, `local`, `dynamics`, `report`,
or `all`. `--protocol v1|v2`, `--condition NAME`, and `--seed N` restrict a run.
`--config PATH` selects a JSON configuration. Completed geometry and captures
are reused unless `--force` is supplied; replay/dynamics/report can be regenerated
from the compact local captures. Smoke outputs live in a separate `smoke/` tree.
If changing the experiment or sampling configuration, use a fresh output root
or `--force`; caches are not interchangeable between configurations.

## Implemented phases and evidence

- `data.py`: inventory, frozen-source/checkpoint hashes, streaming terminal-inclusive
  validation, 20-round checks, deterministic 12/4/4 repetition splits and raw
  round-start/midpoint/end samples. No averaging replaces temporal states.
- `geometry.py`: train-only PCA, disjoint-repetition noise-normalized RDMs,
  competing capability models, familiar-only scalar regression, stronger
  target-layout tests, cross-temporal transfer, and a regression alternative
  to sparse crossed dPCA. Each network has separate coordinates and models.
- `collect.py`, `adapters.py` and `local.py`: import isolation for the original frozen v1/v2
  source, small diagnostic rollouts, full observation/state/key capture,
  normalized GRU inputs, original Flax replay, exact Jacobians, finite-difference
  checks and finite-horizon derivative products.
- `interventions.py` and `replay.py`: discovery-only candidate subspaces,
  whole-carry/projected/attenuation/sham/random/complement controls, magnitude
  sweeps, identical-input replay and paired physical recipient branches.
- `input_dsa.py` and `dynamics.py`: pinned official DMDc and controllability
  components, corrected trajectory-boundary pairing, validation-selected
  rank/delay fits, predictive baselines, short-horizon diagnostics, within-network
  regime comparisons and common-raw-input comparisons across policies.
- `events.py`: observable movement evidence, actual step-aligned update windows,
  request/assignment distinctions and equal-episode/exposure-weighted behavior.
- `report.py`: individual-policy results, grouped uncertainty and limitations.

The primary batch is `counterbalanced1096_20261002_235609`: 30 recurrent policies,
60 historical rollout files, 27,600 full partner episodes and 46 profiles. Both
protocols retain all five learner seeds in each of the three recurrent conditions.
MLP placeholder arrays are excluded from recurrent-state analyses.

The targeted dataset uses six prespecified profiles, four independent histories,
eight layout strata and the first four rounds of each history: 720 histories in
all. It supports early-experience input dynamics and matched new-round tests;
it does not establish late-20-round input dynamics. Historical geometry covers
all 20 rounds. Optional fixed/slow-point optimization was not run.

## Outputs and validation

Outputs are under `eval/partner_dynamics/results/<batch>/<protocol>/<policy>/`.
The batch-level `report.md`, `methods.md`, `data_dictionary.md`, inventory and
coverage files explain the findings and numerical tables. Rich `capture.h5`
files remain local and ignored by Git. Compact fitted operators, projections,
tables and PNG/PDF figures are kept visible.

All analysis scripts, vendored numerical dependencies, documentation, and
generated outputs now live within this `eval/partner_dynamics/` directory.
`eval/partner_dynamics_results` is a compatibility symlink to `results/`, so
previous figure links and the original run's recorded paths still resolve.
The run's original configuration and provenance are preserved; new runs use
the canonical output root from `config.json`.

```bash
$PYTHON -m pytest tests/analysis/test_partner_dynamics.py -q
$PYTHON -m pytest tests/analysis/test_representation_analysis.py \
  tests/analysis/test_representation_analysis_strict.py \
  tests/coordination_grid/test_online_allocation.py \
  tests/coordination_grid/test_evaluation_layouts.py \
  tests/data_prep/test_geometry_goal_dependence.py -q
```

Candidate-subspace validation failures are retained and flagged. A decoder axis
is not automatically a belief axis. Whole-carry effects establish sensitivity
to history-dependent state but can change other remembered variables. Selective
partner-memory claims require effects exceeding the matched controls.

## Interpretable summary plots

The standalone plotting command reads completed numerical summaries without
rerunning or refitting the analyses. It creates four plain-language PNG/PDF
figures and CSVs of their plotted values under the batch's `findings/` folder:

```bash
$PYTHON -m eval.partner_dynamics.plot_findings \
  --results eval/partner_dynamics/results/counterbalanced1096_20261002_235609
```

The figures cover novel-partner delay prediction, faster-goal information over
experience, input-matched allocation effects, and next-state prediction with
observation input. Protocols remain separate; all five seeds are retained.

The completed [figure gallery and numerical exports](results/counterbalanced1096_20261002_235609/findings/README.md)
and [full report](results/counterbalanced1096_20261002_235609/report.md)
are inside this directory. `plot_findings.py` generates the four summary plots;
`geometry.py`, `events.py`, `dynamics.py` and `report.py` generate the detailed
per-policy figures and supporting summaries. `collect.py`, `interventions.py`,
`replay.py`, `rank_controls.py` and `local.py` generate the mechanistic evidence.

## What this adds to the existing linear probes

Comparison sources: the [original written-methods report](../representation_results/counterbalanced1096_20261002_235609/README.md),
its [v1 methods and results](../representation_results/counterbalanced1096_20261002_235609/v1_balanced_training/README.md),
and the [released-code probe comparison](../representation_results/counterbalanced1096_20261002_235609_strict_replication_of_covercooked/protocol_comparison/README.md).
These use the same frozen policies and historical batch. The new targeted
rollouts supplement that batch; no policy training changed.

| Question | Existing linear-probe analysis | Added evidence here |
| --- | --- | --- |
| Is partner capability readable, and does readability grow with experience? | Yes: separate red/blue delay classification from prefix-averaged hidden states, with within-profile episode holdouts. | Raw instantaneous states and faster-goal readouts broadly confirm this. The overall rise in readability is mostly a replication, not a new discovery. |
| Does the readout transfer to partner profiles absent from its fitting data? | Probe fitting includes all 46 profiles. Its familiar/novel subset refers to policy training, not probe training. | Fit and select the delay regressor exclusively on the 24 familiar profiles, then test on the 22 novel profiles. This is a stronger generalization diagnostic. |
| Is there useful speed information beyond knowing which goal is faster? | Random-state and shuffled-label controls test decodability, without this task-specific oracle baseline. | Compare delay prediction against training-derived typical delays conditional on the **true** faster-goal label. Diverse + influence v1 improves MAE from 1.318 to 1.071: improvement 0.247, 95% seed interval [0.180, 0.317]. V2 improves to 1.232, but its improvement interval [-0.039, 0.166] crosses zero. |
| Does one readout remain informative across different phases? | Separate probes at each prefix cutoff/round; accumulated averages blend past and current state. | Fit at one phase and evaluate instantaneous state at another in disjoint episodes. Round-20 start-to-end faster-goal accuracy is 98.0% in v1 and 91.4% in v2 for diverse + influence. This supports cross-phase readability, without isolating memory from current task information. |
| Does recurrent history affect the actor's next allocation? | Probing measures recoverability, without altering policy state. | Swap natural recurrent states under identical recipient inputs. In diverse + influence, whole-state swaps shift assignment probability toward the opposite-orientation donor by 78.1 percentage points in v1 and 22.3 in v2. These swaps can also transplant nuisance memories. |
| Is the identified partner subspace selectively used? | Not tested by the readouts. | Candidate-direction edits are small: 0.95 percentage points in v1 and 0.57 in v2, with similar random-direction effects. Validation uses only six histories per seed; selective partner-memory use remains inconclusive. This is an informative limit, not evidence that partner memory is absent. |
| How do observations drive recurrent-state updates? | Static readouts do not model the state-update mapping. | Input-aware linear dynamical fits beat input-omitting fits in all 30 policies. Median error is 62.3%/65.6% below persistence in v1/v2, within the early four-round capture. Exact local derivatives add sensitivity diagnostics. Predictive success does not identify a partner-belief algorithm or establish an attractor. |

The most substantive additions are novel-profile readout transfer against the
faster-goal prior and controlled interventions on history-dependent state.
They sharpen the original interpretation, but do not establish that the agent
estimates unobserved delays or selectively uses a disentangled partner belief.
The two probe families use different features, targets, splits and fitting
procedures; their raw scores and condition rankings are not directly comparable.
