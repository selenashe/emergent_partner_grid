# Methods and reproduction

The analysis fits each trained network independently. It uses the completed
categorical counterbalanced batch and its hash-verified frozen implementations.
No training, task changes or historical artifact replacement is performed.
The retained analyses use categorical counterbalanced checkpoints.

Primary estimands are context transfer, information beyond the true binary
faster-goal label, cross-temporal persistence, and controlled memory effects.
Historical tests use entire episodes, with validation-selected ridge penalties
and training-only centering/scaling/PCA/noise estimates. Capability regressors
for novel-profile transfer see only familiar profiles during training/validation.
A conditional faster-goal population mean supplies a strong binary-rule baseline.
The target-layout test evaluates analysis-model transfer; the policy was trained
on the shared layout corpus. Histories may overlap target layouts.

Geometry uses one raw state per start/midpoint/end per round, equalizing episode
and profile contributions. RDMs use independent repetition halves and a training
residual diagonal variance estimate with 20% median-variance shrinkage. Full
noise-whitened distances are not claimed. Euclidean/correlation sensitivities
are retained. RDM regression fits training distances and evaluates disjoint test
repetition distances. Dependent profile-pair correlations are descriptive.
Sparse profile-by-layout cells preclude full crossed dPCA, so multivariate
regression compares geometry/phase, binary, capability and interaction predictors.
Correlated capability predictors are not uniquely disentangled.

The collector uses six prespecified familiar/novel profiles, eight coordinate
geometry strata, four histories per profile, and the first four rounds. Full
20-round memory geometry is available from historical files; dynamics and causal
tests are compact early-history diagnostics. Discovery data defines a rank-three
candidate subspace. Held-out validation failures remain visible; perturbation
outcomes do not select a subspace. Paired branches keep recipient truth and keys
fixed. Magnitude zero must reproduce the original outcomes. Whole-carry swaps
can change nuisance memories; they are coarse interventions.

Exact `jax.jacfwd` derivatives apply to the original Flax GRU, with directional
finite-difference checks for hidden/input perturbations, singular-value gains,
local eigenvalues and four-step Jacobian products along real trajectories.
Episode reset is a discrete operation and is excluded from derivative claims.
A fixed-point optimization was not run; it is optional and would concern clamped
GRU inputs rather than the policy–environment loop.

InputDSA uses unmodified official DMDc/controllability files from
[DSA](https://github.com/mitchellostrow/DSA/tree/c437ce93ddab5a578b369ef9f1cbbbfa0f6ff7a2),
MIT, revision `c437ce93ddab5a578b369ef9f1cbbbfa0f6ff7a2`.
The upstream list fitting path concatenates trajectories before forming pairs;
a project subclass fixes only pair assembly so no boundary transition is added.
The synthetic regression test recovers known A/B changes and verifies output
ordering. Native distance tuple ordering is **joint, state, control**; exported
components are explicitly named. Full GRU states are observed, so DMDc is used
rather than a partial-observation SubspaceDMDc estimator. Rank/delay grids use
validation error, not partner separation. Report persistence, input-only and
input-omitting linear baselines, four-step errors and numerical conditioning
before interpretation. State and normalized encoder inputs are projected onto training-only 40- and 20-component PCA bases, respectively; fitted operators approximate the full GRU and do not replace exact derivatives. Fitted regime matrices describe visited states of the
same fixed weights. Common raw-input coordinates are a seeded lossy projection
of every observation leaf; their predictive errors accompany cross-policy distances.

Episode-group bootstrap intervals are supplied for prediction improvements and
paired effects. Condition summaries resample independent learner seeds; only
five seeds are available, so these intervals are descriptive and should not be
read as precise population estimates. Individual-seed results and all weak
policies are retained. No timesteps are counted as independent replicates.

## References inspected

- [InputDSA paper](https://arxiv.org/abs/2510.25943) and
  [original DSA paper](https://arxiv.org/abs/2306.10168).
- [MPRNNPublic](https://github.com/mitchellostrow/MPRNNPublic/tree/36aa4ed23103163fbd3ccc7695642e22592f7565),
  revision `36aa4ed23103163fbd3ccc7695642e22592f7565`: inspected
  `scripts/rsa.py`, `pca.py`, `density.py`, `perturb.py`, and the actual filename
  `scatter_distances.py`. No license file was found, so no code was copied.
  This competitive matching-pennies study motivates diagnostics; it is not task-equivalent.
- [dPCA](https://github.com/machenslab/dPCA/tree/1def5b15854638811a25257dc0b68074ab5a1be0),
  revision `1def5b15854638811a25257dc0b68074ab5a1be0`: inspected Python core and
  `License.md` (MIT; Python source also carries a BSD header). No code copied.
  The README distinguishes the Python core from the paper's MATLAB functionality.
- [JacobianODE](https://github.com/adamjeisen/JacobianODE) motivates local
  propagation; the known GRU permits exact discrete-time JAX derivatives without
  training an estimator or transferring weights.
- [Fixed-point finder](https://github.com/mattgolub/fixed-point-finder), related
  [toolbox paper](https://doi.org/10.21105/joss.01003), and the
  [original fixed-point paper](https://doi.org/10.1162/NECO_a_00409).
  These are optional methodological references; that toolbox was not imported.

## Exact successful commands

All commands ran from `/juice6/u/jshe/emergent_partner_grid`. The shell prefix
was `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2`; collection/replay also set
`JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1`. Runtime:
`/nlp/scr/jshe/miniconda3/envs/emergent_partner_model/bin/python`.

```bash
python -m eval.partner_dynamics --dry-run
python -m eval.partner_dynamics --phase geometry --smoke
python -m eval.partner_dynamics --phase collect --smoke
python -m eval.partner_dynamics --phase dynamics --smoke
python -m eval.partner_dynamics --phase collect --smoke --protocol v2
python -m eval.partner_dynamics --phase replay --smoke --protocol v2
python -m eval.partner_dynamics --phase geometry
python -m eval.partner_dynamics.collect --version v1 --config eval/partner_dynamics/config.json --output eval/partner_dynamics/results/counterbalanced1096_20261002_235609/fixed_v1
python -m eval.partner_dynamics.collect --version v2 --config eval/partner_dynamics/config.json --output eval/partner_dynamics/results/counterbalanced1096_20261002_235609/online_v2
python -m eval.partner_dynamics --phase dynamics
python -m eval.partner_dynamics --phase replay
python -m eval.partner_dynamics --phase validate
python -m eval.partner_dynamics --phase controls --smoke
python -m eval.partner_dynamics --phase controls
python -m eval.partner_dynamics --phase local
python -m eval.partner_dynamics --phase report
python -m pytest tests/analysis/test_partner_dynamics.py -q
python -m pytest tests/analysis/test_representation_analysis.py tests/analysis/test_representation_analysis_strict.py tests/coordination_grid/test_online_allocation.py tests/coordination_grid/test_evaluation_layouts.py tests/data_prep/test_geometry_goal_dependence.py -q
```

The displayed `python` in this command log denotes the absolute runtime above.
The source-package README provides directly executable commands. Provenance,
configuration, per-policy source identifiers and machine-readable coverage are
saved in the output tree. Initial smoke failures exposed parameter-naming/API
and profile-balancing issues; they were corrected before scaling. Synthetic
boundary and exact replay/derivative tests passed before interpretation.
