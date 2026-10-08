# Coding-agent prompt: partner representations and recurrent dynamics in CoordinationGrid

You have the local checkout of https://github.com/selenashe/emergent_partner_grid, the trained checkpoints, and the large evaluation HDF5 files containing hidden states. Implement and run a reproducible analysis pipeline that explains how an RNN trained for joint reward with a scripted partner acquires, maintains, updates, and uses partner information. Go beyond capability decoding and UMAP plots. Produce quantitative evidence about representational organization, recurrent dynamics, and behavioral effects, including negative results.

This is an analysis and targeted evaluation task. Reuse the existing policies, experiment definitions, and recorded trajectories. Do not retrain policies, launch new training batches, change the task or reward, or overwrite historical artifacts. When observations needed for a particular analysis were not recorded, collect a small, separate, instrumented evaluation dataset from existing checkpoints.

## 1. Ground the implementation in the actual local experiment

The remote repository was inspected on 2026-10-06 at commit `25a883c5718e6a62ee5de094bb06cabb23a4644d`. Your local checkout may be newer; inspect it, follow applicable `AGENTS.md` instructions, and treat its checkpoint configs and source snapshots as authoritative. Record the local commit and any working-tree changes. Do not revert unrelated changes.

Read these active files before implementing:

- `README.md` and `project_log_revised.md` for experiment provenance.
- `eval/representation_analysis.py` and `tests/analysis/test_representation_analysis.py` for existing data discovery, valid-episode masking, probes, and reporting.
- `eval/evaluate_partner_modelling.py` for rollout generation and the exact hidden-state convention.
- `train/ippo_rnn_coordination_grid.py` for `CommObsEncoder`, `ScannedRNN`, `ActorCriticCommRNN`, and checkpoint loading.
- `jaxmarl/environments/coordination_grid/coordination_grid.py` and `capability_populations.py` for task timing, action semantics, capability populations, and observations.
- `repo_paths.py`, `archive/relocations.json`, the selected batch's manifests, saved configs, and frozen sources for resolving relocated experiments.
- `data_prep/check_geometry_goal_dependence.py` and the existing corpus audit for the task's actual allocation demands.

Important facts to verify locally:

1. Active implementation directories are `train/`, `eval/`, and `data_prep/`; do not rebuild an old `dev/` or root `analysis/` pipeline.
2. Use the completed categorical batch `counterbalanced1096_20261002_235609`, with separate `v1_balanced_training` and `v2_balanced_training` evaluations. Preserve each protocol and its recorded frozen source.
3. Four conditions exist: `rnn_diverse_influence`, `rnn_single_influence`, `rnn_diverse_noinfluence`, and `mlp_diverse_influence`. The first three have recurrent states; do not analyze the MLP's placeholder hidden arrays as meaningful memory. Optional MLP comparisons require real encoder or policy activations.
4. v1 fixes allocation for a round; v2 allows allocation requests during the round. In current v2, t=0 is a supplied random assignment with forced action semantics, not a learned allocation. Read each protocol's frozen implementation, rather than evaluating v1 with a trainer that rejects or changes it.
5. A partner episode contains 20 rounds with fixed partner capability and changing layouts. GRU memory is retained across round boundaries and resets at partner-episode boundaries.
6. Capabilities are delays `(d_R, d_B)`: `d=k` means a partner moves once every `k+1` eligible steps. Training has 24 profiles; held-out capability evaluation has 22 profiles involving previously unseen scalar delays `{0,5,6}`. Import the authoritative constants; do not use the older cooldown definition from earlier project notes.
7. The common 1096-layout corpus is used for policy training and evaluation. A held-out layout split for an analysis model tests representational transfer across contexts, not a policy's generalization to layouts never encountered in training.
8. Existing capability probes fit on episodes from both familiar and novel capability profiles, with a 16/4 repetition split per profile. Their novel-profile scores are not tests of an analysis model trained exclusively on familiar profiles. Keep this baseline unchanged, and label stronger new generalization tests explicitly.
9. The corpus audit reports that the analytically preferred static independent-path assignment always gives the partner its faster goal across the authoritative profiles. Verify this, and distinguish that formula from actual online switching and collision dynamics. This makes a binary relative-speed rule a serious alternative to a full capability representation.

## 2. Reference papers and verified repositories

Use these as methodological references and reuse suitable implementations after checking their current APIs, dependencies, licenses, and numerical assumptions. Pin revisions in provenance. Do not copy whole legacy training stacks into this project.

| Reference | Paper | Code and intended use |
| --- | --- | --- |
| InputDSA | https://arxiv.org/abs/2510.25943 | https://github.com/mitchellostrow/DSA — input-aware dynamics fitting and comparisons. Inspect `InputDSA`, `GeneralizedDSA`, DMDc/SubspaceDMDc, and examples. |
| Original DSA / Beyond Geometry | https://arxiv.org/abs/2306.10168 | https://github.com/mitchellostrow/DSA — dynamical comparison rather than state-cloud geometry. Use ordinary DSA as an input-omitting comparison, not the default explanation of an input-driven policy. |
| Social inference and generalization | https://social-intelligence-human-ai.github.io/docs/camready_8.pdf | https://github.com/mitchellostrow/MPRNNPublic — ideas from `rsa.py`, `pca.py`, `density.py`, `scatter_distance.py`, and `perturb.py`; locate actual paths. This is competitive matching pennies, so adapt the analyses to cooperation rather than claiming task equivalence. |
| Jacobian-based control | https://arxiv.org/abs/2507.01946 | https://github.com/adamjeisen/JacobianODE — local sensitivity and propagation of perturbations. Because our trained GRU is available, compute exact discrete-time derivatives in JAX rather than training an ODE estimator by default. |
| Demixed PCA | https://arxiv.org/abs/1410.6031 | https://github.com/machenslab/dPCA — factor-conditioned population trajectories. The Python implementation provides core functionality; check differences from the paper's MATLAB implementation. |
| Fixed/slow-point analysis | https://doi.org/10.1162/NECO_a_00409 | https://github.com/mattgolub/fixed-point-finder — a related later toolbox, not the original paper's repository. Its companion toolbox paper is https://doi.org/10.21105/joss.01003. Adapt the mathematical objective to JAX; do not port our GRU weights into another framework just for this analysis. |

InputDSA separates input effects from recurrent evolution, but does not assign meanings to latent dimensions. Geometric organization is not itself evidence of causal use. Jacobian dynamics show local sensitivity, and interventions test behavioral influence. Treat these as complementary levels of evidence.

## 3. Scientific questions and competing explanations

Organize the analysis around these questions:

- Is remembered partner information stable across different layouts, round starts, and action choices?
- Does the representation contain absolute delays, relative speed, overall speed, or only a task-sufficient assignment rule?
- When and after what observations does partner-related state change? What persists into the next round?
- Do unfamiliar capabilities occupy predictable locations in a familiar representational structure?
- Does remembered partner information change how the same observation is processed?
- Does modifying that information change subsequent allocation, navigation, value predictions, or joint reward?

Compare representational hypotheses using actual capability values and evidence exposure:

- Full profile `(d_R,d_B)`.
- Relative delay `d_R-d_B`, its magnitude, and faster-goal identity.
- Overall delay `(d_R+d_B)/2`; optionally rates `1/(d_R+1)` and `1/(d_B+1)` as a prespecified sensitivity analysis.
- The delay of the currently pursued partner goal.
- Accumulated observable evidence about each goal's speed.
- Allocation preference, action tendency, and value/completion-time predictions.

The two goal capabilities are not necessarily both identifiable from a particular trajectory. Stratify by which goals the partner has pursued and what informative movement intervals the ego could observe. Do not interpret successful prediction of an unobserved delay as direct estimation without comparing a population-prior explanation. Do not infer epistemic uncertainty from ground-truth labels alone; label any explicit behavioral observer or posterior model as an analysis model, not the agent's demonstrated belief.

## 4. Phase A — audit data, preserve temporal structure, and establish splits

Implement a streaming trajectory loader and write a machine-readable inventory before analysis. Include resolved source paths, batch, protocol, action selection, condition, learner seed, checkpoint/config/source identifiers, file shapes, available fields, and missing fields. Use existing manifest checksums when available; record file size/mtime and reproducible fingerprints without repeatedly hashing enormous arrays.

Existing HDF5 fields include `hidden_state`, `capability`, `layout_idx`, `round_idx`, `round_done`, `success`, `rewards`, `dones`, `ego_alloc_action`, `partner_assignment`, `partner_goal`, `round_time`, `is_t0`, `assignment_changed`, and `ego_request_changed`. Inspect actual files rather than trusting this list. The schema uses `rewards`/`dones` despite singular names in parts of its prose documentation.

Validate:

- Include the first terminal transition and exclude all subsequent scan padding. Validate 20 correctly ordered round endings on complete episodes.
- Verify capability constancy within an episode, shape/dtype/finite values, protocol provenance, and profile-to-episode ordering.
- Keep raw per-step trajectories. Existing prefix averages and round-end cumulative averages are baselines, not substitutes for temporal dynamics.
- `hidden_state[t]` is post-observation state: it produced action t after consuming observation t. `round_time[t]` is post-transition time, while layout/round info identifies the pre-transition round. Document the exact timing of every derived field.
- Preserve real episode and round transitions; never create transitions by concatenating unrelated episodes, padding, or sampled event windows. Analyze actual round-boundary transitions separately when studying persistence.

Create deterministic group-level train/validation/test manifests before fitting. Split entire partner episodes, never individual timesteps. For tests across contexts, additionally separate target layout groups and explain whether preceding histories used those layouts. Stronger history-disjoint tests may need the targeted collector below. Fit scalers, PCA, residual models, dPCA, and dynamical-model hyperparameters using training/validation data only.

Maintain distinct evaluations for within-profile episode generalization, familiar-to-novel capability generalization, and cross-layout/context transfer. For scalar delay extrapolation, use regression or geometric tests; a classifier trained only on familiar scalar classes cannot recognize an unseen delay class by construction.

Start with one policy and a small episode subset to validate the pipeline. Then analyze all available learner seeds in each RNN condition for the primary batch, separately for v1/v2. Existing common-seed selection records may define illustrative plots; do not select the strongest seed by the new representation or held-out outcomes.

## 5. Phase B — representational geometry and event-aligned trajectories

Implement analyses within each trained network separately. Do not pool raw hidden coordinates across independently trained networks.

### B1. Controlled sampling and event alignment

Build a tidy index linking raw hidden-state samples to episodes, capabilities, layouts, round number, within-round phase, and available behavioral variables. Define windows around round start, informative partner movement, allocation requests and actual goal changes, and round completion. Distinguish requests from effective assignments and distinguish t=0 defaults from real policy choices. Require observed movements or explicit simulator state for movement evidence; do not reconstruct trajectories from capability labels alone.

Balance or weight profile/phase/layout exposure so long, slow, or unsuccessful trajectories do not dominate merely by providing more timesteps. Present equal-profile/equal-episode summaries alongside exposure-weighted results. Time normalization may support descriptive plots, but dynamics fitting must use actual environment-step intervals.

### B2. Quantitative geometry

- Fit PCA on training trajectories and apply it to validation/test data. Plot event-aligned trajectories in the same coordinates, with held-out explained variance and uncertainty.
- Construct cross-validated representational dissimilarity matrices at selected phases. Use disjoint episode repetitions for estimates; prefer a justified cross-validated distance with regularized noise estimation, plus a transparent Euclidean/correlation sensitivity analysis.
- Compare the representational matrices with competing capability, relative-speed, overall-speed, observed-task-speed, layout, phase, and behavioral models. Report held-out prediction or variance explained, uncertainty, and identifiable contributions when models are correlated. Do not label correlated predictors as uniquely disentangled.
- Quantify geometry within each faster-goal group: can it distinguish absolute or relative magnitude after removing the binary assignment distinction? Compare profiles with the same faster-goal identity but different absolute delays.
- Measure whether novel profiles occupy the familiar PCA subspace, whether capability relationships predict their locations, and whether they fall outside the sampled familiar support. Use distances in the validated high-dimensional space; do not diagnose extrapolation from the visual position of a UMAP point.
- Compute cross-temporal transfer: fit a partner-related subspace or decoder at one phase/round and evaluate at other phases/rounds and disjoint episodes. Report representation persistence, remapping, and emergence rather than only same-time decodability. Keep decoding a supporting diagnostic, not the main result.

### B3. Mixing with task state

Use balanced factor-conditioned data for dPCA when the required crossed cells have enough repetitions. Separate time/phase, faster-goal or capability, layout/context, and partner-by-context interactions. Full 46-profile × 1096-layout crossing is not supported by the existing random evaluations; use a prespecified subset or the collector. Do not silently fill missing cells with zeros or treat imputed cells as observations.

If dPCA is unsupported or numerically unsuitable, implement an explicit cross-validated multivariate regression/variance-partitioning alternative and document the difference. Examine whether partner directions remain aligned or rotate across task contexts. Report both total representation and representation conditional on current observable state. Conditioning on the chosen action or other downstream outcomes can remove the effect of memory, so keep such conditional tests separate from causal claims.

Deliver figures for event trajectories, representational model comparisons, cross-temporal transfer, novel-profile geometry, and factor-conditioned variance. Preserve numerical tables behind each figure.

## 6. Phase C — small instrumented evaluation dataset where required

The current evaluator does not save raw observations, GRU input embeddings, sampled flat actions, logits, or full simulator states. Inspect local files for richer captures first. If needed, add a separate collector or backward-compatible opt-in capture mode. Keep existing HDF5s and summaries unchanged.

Use the checkpoint's exact saved architecture, source snapshot, environment protocol, action-selection setting, and normalization. Capture:

- Raw observation leaves actually presented to the policy.
- Encoder output after the normalization actually supplied to the GRU.
- GRU carry before observation, reset flag, and post-observation hidden state.
- Policy probabilities/logits on valid actions, legal-action mask, value output, sampled flat action, decoded move/request, and actual partner assignment.
- Explicit pre/post simulator state and PRNG keys sufficient for deterministic replay, including positions, layout, capability, partner move counter, round time, and round boundary. Store ground truth as evaluator metadata only, never feed it into the policy.
- IDs connecting the capture to checkpoint, episode, event, and intervention branch.

Implement a pure one-step adapter using the original Flax/JAX computation and parameters. Verify that it reproduces original hidden states, policy probabilities, value outputs, and sampled actions with the same keys. Reconstruct recorded trajectories only if full replay equivalence can be established; otherwise record a new evaluation dataset and label it clearly.

Prespecify a compact diagnostic layout subset from geometric strata, excluding outcome-based selection, crossed with familiar and novel profiles and repeated independent episode histories. Support `reset_from_schedule` or the protocol-equivalent frozen API for matched layout sequences. Common random numbers help pair evaluations, but paired free-running policies can still reach different observations; they are not observation-matched merely because their seeds agree.

Use two complementary branches:

1. Free-running, in-environment evaluation with real partner behavior and joint reward.
2. Open-loop replay of an identical observation/encoded-input sequence from different history-derived recurrent states to isolate the effect of memory. Open-loop replay supports input-matched network analysis, not claims about achievable environment reward.

For the decisive memory test, obtain different history-derived states, then present the same physically valid new-round observation and random initial assignment. Differences at that point can be attributed to earlier histories more cleanly than differences between freely diverging navigation trajectories. Match exposure duration/phase and nuisance history as far as practical.

Collect only what the analysis needs, write new output paths, and do not regenerate every saved rollout simply because a richer schema is useful.

## 7. Phase D — input-aware dynamics using InputDSA

Wrap the official DSA package behind a project adapter. Inspect the current implementation rather than copying README snippets verbatim. Its README currently exposes DMDc/SubspaceDMDc and separately interpretable intrinsic, input, and joint controllability comparisons. Verify the returned distance-component ordering against the pinned revision.

Define time alignment explicitly. With recorded post-observation states `h_t=G(h_{t-1},e_t)`, the one-step transition pairs are `(h_t,e_{t+1}) -> h_{t+1}`. Fit a validated lifted approximation, schematically `z_{t+1} ≈ A z_t + B v_{t+1}`, where z lifts hidden states and v represents the actual input. Do not shift inputs by one step accidentally.

Use the normalized encoder outputs for faithful within-network GRU analysis. For comparisons across independent policies, use a common raw-observation feature basis or a validated input alignment: learned encoder coordinates differ across networks. Include all policy-driving variables in their appropriate roles, with reset/round-start transitions treated explicitly. Ground-truth partner capability is an analysis label, not an input to B.

Fit and compare:

- Early versus late partner experience, using enough contiguous data rather than one sample per phase.
- Same network across faster-goal groups and, where identifiable, same-orientation/different-magnitude profiles.
- RNN experimental conditions and learner seeds under matched task/input sampling.
- v1/v2 and other batches only as explicitly separate comparisons with matched sampling and preserved provenance.

Start with small rank/delay grids, such as ranks `{10,20,40}` and delay counts `{1,2,4,8}`, adapting to segment lengths and package conventions. Use validation prediction error and stability to select settings, not the strength of partner separation. Delay windows must not cross fabricated boundaries or episode resets. Preserve genuine round transitions only in the analysis intended to model them.

Report held-out one-step and short-horizon predictions against persistence and input-only baselines, conditioning/numerical diagnostics, rank/delay sensitivity, and bootstrap uncertainty. Compare against ordinary DSA to quantify the effect of omitting inputs. Do not interpret unreliable fits. If a global additive input approximation fits poorly, use phase-conditioned fits and exact local derivatives; do not force a mechanistic story from a poor global model.

Crucial interpretation: a given trained network has fixed weights across partner conditions. Differences between fitted A/B matrices across those conditions may reflect different visited regimes of the same nonlinear system, not separate learned partner-specific parameters. Support a partner-memory claim with input-matched histories, local sensitivity, and interventions.

Cache compact fitted operators and compare them independently rather than loading every policy's raw trajectory into one all-to-all object. Keep comparisons within validation-supported ranges. Address closed-loop coupling between observations and actions through the controlled replay branch; observational fitting alone is not causal identification.

Deliver intrinsic/input/joint distance matrices, their relationships to capability hypotheses, local/global fitting diagnostics, and interpretable memory/response summaries. Avoid claiming a fitted eigenmode is a partner belief without separate evidence.

## 8. Phase E — exact local dynamics and targeted interventions

### E1. Derivatives of the actual GRU

For the one-step update `h_next=G(h,e)` with resets disabled at valid nonterminal samples, compute `J_h=∂G/∂h` and `J_e=∂G/∂e` using JAX. Treat episode reset as a separate discrete operation. If examining sensitivity to raw observations, chain through the encoder and state which observation components are differentiable; discrete grid perturbations need actual counterfactual observations for interpretation.

Verify derivatives using directional finite differences at sampled states. Analyze gain, contraction, and rotation of candidate partner directions, and how identical inputs are processed from different history-derived states. Because the system is time-varying and can be non-normal, supplement pointwise eigenvalues with singular-value gain and finite-horizon Jacobian products along real trajectories. Relate these to policy probability/value sensitivities.

Fit candidate partner-related directions/subspaces on discovery data using matched profile contrasts, dPCA/regression components, or a cross-validated probe. Estimate them within one policy only; do not use held-out intervention outcomes to choose the subspace. Demonstrate separation from layout/current-action differences rather than naming a decoder direction a belief axis automatically.

### E2. Interventions and paired controls

At matched new-round starts or another justified pre-observation point, perturb the carry and then run the original update and action selection. Distinguish this from perturbing the post-observation state immediately before policy readout.

Run:

- Natural whole-carry swaps between matched histories as a coarse memory intervention; these can change many remembered variables, so do not claim partner-specificity from them alone.
- Partner-subspace swaps: `h' = h + P(h_donor-h)`, with P a held-out-validated orthogonal projector onto the candidate partner subspace. Also test bounded scaling toward the donor.
- A justified partner-component attenuation/removal, using a training-derived centering reference.
- Sham/same-profile donor swaps, norm-matched random-direction perturbations, and rank/norm-matched non-partner-subspace perturbations.
- Separate an opposite-faster-goal donor from a same-faster-goal/different-absolute-delay donor. The latter is particularly important for testing information beyond the binary allocation rule.

Clone the recipient environment state and PRNG state for intervention/control branches, changing only the specified hidden component. Preserve the true recipient partner. Within each protocol, measure immediate allocation probabilities, sampled allocation, subsequent switches, value output, completion time, collisions if logged, and joint return/success. v2 t=0 action is forced, so assess the first legal learned allocation after it; adapt the timing to the exact v1 implementation.

Assess proximity to empirical hidden-state support and use a perturbation-magnitude sweep. An intervention that indiscriminately destroys policy function is evidence of disruption, not of selective partner-memory use. Show directional and selective effects compared with controls. No-influence agents may still use partner information for navigation, so do not require their effects to be zero by definition.

Support both open-loop network-response tests and paired closed-loop environment rollouts. Report the effect with uncertainty across independent episodes and learner seeds. Treat post-intervention observational changes as consequences of the intervention, not extra matching covariates.

### E3. Optional fixed/slow points

After the core pipeline works, optimize `||G(h,e_fixed)-h||²` from observed-state initializations for a small set of justified fixed encoded inputs. Deduplicate solutions, report residuals, distance from observed support, and local stability. For discrete-time fixed points, local stability requires all Jacobian eigenvalues to have modulus below one; distinguish slow points from converged fixed points.

This tests the GRU under clamped inputs. It does not establish an attractor of the full policy–environment loop. Absence of isolated fixed points does not disprove memory carried by transient or input-dependent dynamics. Keep this supplementary if it is computationally expensive or uninformative.

## 9. Statistical and interpretive requirements

- Treat learner seeds and partner episodes as independent sampling units; timesteps within an episode are not independent replicates. Use a prespecified hierarchical or appropriately grouped bootstrap and display individual-seed results. For paired interventions, resample paired recipient/control episodes together.
- Use balanced profile weights and report cell counts, missingness, exposure, and weak policies. Do not drop unsuccessful agents because they weaken a result. Performance-matched comparisons may be secondary, with explicit selection rules.
- Use label permutations at episode/profile levels appropriate to the estimand, preserving temporal dependence. For time structure, use within-round blocked/circular controls only where their boundary and stationarity assumptions are defensible.
- Bound extensive sweeps and label exploratory analyses. Prespecify a small set of primary tests: transfer of partner structure across task contexts; representational information beyond faster-goal identity; persistence/update of history-dependent state; and selective behavioral effects of partner-subspace interventions.
- Maintain distinctions between decodability, structure, estimated dynamical mechanisms, and causal behavioral influence. Similarity metrics do not prove that a network has a human-like theory of mind or an explicit Bayesian posterior.

## 10. Implementation, outputs, and validation

Use an additive package such as `eval/partner_dynamics/` with modules for data/provenance, splits, controlled collection, geometry, InputDSA, JAX adapters/local dynamics, interventions, plotting, and report generation. A single CLI can dispatch phases. Reuse shared valid-length/provenance helpers where sensible without changing the existing probe analysis's behavior. Do not introduce a duplicate environment implementation.

Provide a YAML/JSON configuration with explicit batch/protocol/condition/seed selection, local data roots, grouping rules, diagnostic layout subset, event definitions, rank/delay settings, capture limits, intervention magnitudes, device, and analysis seed. Include `--dry-run` inventory and `--smoke` options and resumable cached phases. Document exact commands that actually run in this checkout, rather than leaving only pseudocode.

Suggested output root: `eval/partner_dynamics_results/<batch>/<protocol>/`, separate from `eval/representation_results/`. Save:

1. `data_audit.json`, provenance, configuration, split manifests, and a data dictionary with timing conventions.
2. Geometry/dPCA/transfer tables, figures in PNG/PDF, fitted projections, and uncertainty/cell counts.
3. Dynamics-fit diagnostics, compact A/B operators, state/input/joint distance matrices, and sensitivity results.
4. Local Jacobian summaries, perturbation controls, paired intervention outcomes, and any optional fixed-point results.
5. `report.md` explaining each main figure, the evidence for each hypothesis, negative results, limitations, and what the task's design can and cannot identify.
6. A concise methods document with reference links, upstream revisions, and all reproducing commands.

Large HDF5 captures, raw Jacobian stacks, checkpoints, and bulky caches remain local and ignored by Git. Track analysis source, configs, compact summaries, and selected figures consistent with repository conventions. Add narrow ignore rules for the new large outputs rather than blanket rules that hide useful summaries. Stream HDF5 by episode/chunk; bound memory and avoid copies of the full datasets.

Run meaningful tests for terminal/padding masks, time alignment, real versus fabricated round transitions, split leakage, one-step replay equivalence, derivative accuracy, and intervention identity/controls. For the InputDSA adapter, use a small synthetic driven system with known changes to recurrent versus input operators to catch swapping/misalignment of outputs. For geometry, verify null label permutations and train-only preprocessing. Test that a zero-magnitude intervention reproduces the unmodified trajectory with identical keys.

Finish a smoke run before scaling. Then run the core analyses on the available primary-batch recurrent policies and the targeted evaluation subset, retaining all available learner seeds. Run relevant existing tests after code changes. If a phase lacks prerequisites, complete independent phases and explain exactly which data or checkpoint/source mismatch blocks it; do not fabricate outputs or stop after writing unexecuted scripts.

At completion, return the changed files, exact commands run, tests and replay checks, data coverage, primary quantitative findings, and unresolved limitations. A successful deliverable explains whether partner information forms a reusable task-conditioned representation, how it evolves, and whether it changes behavior—even if the best-supported explanation is only a faster-goal heuristic.
