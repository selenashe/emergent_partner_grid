# Partner representations and recurrent dynamics

Primary batch: `counterbalanced1096_20261002_235609`. Protocols, conditions and learner seeds are analyzed independently; no hidden coordinates are pooled.

## Raw-state geometry and transfer

Primary capability regression uses the raw post-observation state at round 20 start. MAE is compared with a training-derived capability prior conditional on the true faster-goal identity. Positive improvement means information beyond that binary label helps prediction. This is a representational diagnostic, not evidence of causal use.

| Protocol | Condition | Evaluation | Delay MAE | Binary prior MAE | Improvement (95% seed bootstrap) |
| --- | --- | --- | ---: | ---: | ---: |
| fixed_v1 | rnn_diverse_influence | familiar_to_novel | 0.874 | 0.875 | 0.001 [0.001, 0.001] |
| fixed_v1 | rnn_diverse_influence | target_layout | 1.051 | 0.636 | -0.414 [-0.414, -0.414] |
| fixed_v1 | rnn_diverse_influence | within_profile | 0.690 | 0.625 | -0.065 [-0.065, -0.065] |

## Collection, dynamics and interventions

2 policies have completed targeted capture and replay checks.

### fixed_v1

{
  "status": "completed",
  "policies": 1,
  "median_test_improvement_over_persistence": 0.3543145514959567,
  "limitations": [
    "InputDSA uses pinned official DMDc/controllability components with explicit safe transition pairing.",
    "Unmodified upstream list concatenation would create cross-trajectory transitions; adapter corrects this boundary assembly.",
    "Native distance ordering is joint/state/control; exported fields are named explicitly.",
    "Regime matrices reflect the same nonlinear weights visited in different states.",
    "Common raw input projection is a fixed lossy basis; inspect its held-out prediction error before cross-policy comparisons.",
    "Only the prespecified early four-round diagnostic capture is fitted; full late-episode input dynamics require additional collection.",
    "Bootstrap of paired recipients is shown per policy; effects must also be compared across learner seeds."
  ]
}


## Interpretation and limits

Decodability, representational geometry, fitted dynamics and behavioral interventions answer different questions. A positive decoder improvement does not establish that the policy estimated an unobserved delay; correlated population priors and task state remain alternatives.

All 20 episode rounds are audited with terminal-inclusive masks. Geometry uses equal numbers of raw start/midpoint/end samples per episode. Recorded raw trajectories remain intact in the historical HDF5 files. Timesteps are never independent statistical replicates.

Target-layout tests hold out the sampled target layouts from analysis fitting; earlier histories can overlap them. These tests do not measure policy generalization to layouts absent from policy training.

Sparse profile-by-layout crossing is handled with train/validation/test multivariate regression instead of filling missing dPCA cells. Layout conditioning uses recorded geometry and phase, not unrecorded full current observations.

RDM scores use training-only diagonal shrinkage and disjoint repetition halves. Their pairwise correlations are descriptive because profile pairs are dependent. Capability/relative/overall models are correlated and are not uniquely disentangled.

The static corpus audit is a baseline: all authoritative profiles favor giving the partner its faster goal. That formula does not simulate collisions or online switches.

Each policy directory contains numeric tables, event trajectories, PCA/novel geometry, representational models, cross-temporal transfer and variance figures. Consult `data_audit.json`, split manifests and per-file validation for coverage and provenance.

## Reproduction

See the source package README and `methods.md` for exact commands and runtime settings.
