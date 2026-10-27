# Joint learning and communication pilots A–E

**Status:** draft for discussion; no implementation or training launch is authorized by this document.

**Experiment family proposed:** `joint_ippo_comm_v1`.

**Scope:** five pilots, A–E, one matched training seed each, exactly 20,000,000 joint environment transitions per pilot. This is a new experiment family alongside the maintained single-learner `v1_matched`, `v2`, and `v3` experiments.

## 1. Scientific contract

The task remains a 7×7 grid with two starts and two colored goals. Both agents learn to move and communicate. Each controls its own actions and, subject to the goal-interface decision below, its own pursued goal. Neither agent assigns the other a goal, and the environment never enforces complementary choices.

- Two independent recurrent PPO learners: separate parameters, local critics, optimizers, random streams, and recurrent states. No parameters or hidden states are shared between agents. Within one learner, the actor and value heads may retain the current shared encoder/GRU backbone.
- Both receive the same scalar team reward on every simultaneous transition: **+1 if physically occupying distinct goals; −0.01 otherwise**. The successful transition receives +1, without an additional step penalty. No message fee, communication bonus, auxiliary communication loss, allocation bonus, or service work.
- Each agent privately observes its own stable capability profile `(d_R, d_B)`. Both profiles remain fixed for a 20-round partner episode. The other profile is never directly observable.
- Preserve the destination-delay rule on a real goal change: load the newly pursued goal's delay before testing movement eligibility. Repeating a target does not reload a delay.
- A/B/C/E have seven **behavior symbols**: `UP`, `DOWN`, `RIGHT`, `LEFT`, `WAIT`, `SAY_M0`, `SAY_M1`. Saying a token replaces movement. The separate issue of encoding goal choice is discussed in §2.
- Tokens have no predefined meaning. Delivery is to the partner's next observation, with no distance restriction, including in E. Tokens never write either agent's target register.
- Preserve 20 rounds per partner episode and a proposed 100-transition round horizon, including initialization. Memory persists between rounds and resets between partner episodes. Recheck horizon feasibility with two delayed agents before adopting 100.
- Balance generated and sampled task opportunities. Do not force learned goal choices, message counts, token frequencies, or successful allocations to be equal.

One joint transition advances both agents once. It counts as one environment step and supplies one transition to each learner. The full pilot program is 100M joint transitions and 200M per-agent transition records.

## 2. Decisions that must be settled during review

These are missing parts of the task contract, rather than implementation details to decide silently. All later mechanics are proposed defaults, conditional on resolving the first two items.

| ID | Decision | Proposed treatment |
| --- | --- | --- |
| P1 | How does an agent select/revise its own pursued goal with a seven-symbol behavior alphabet? | Add a **private binary self-target head** `{RED, BLUE}` to each policy, alongside its behavior head. Its output affects only that agent. This gives seven behavior symbols but 14 behavior/target combinations after initialization; it is not literally seven total policy outcomes. Confirm that this interpretation is intended. |
| P2 | What makes a selected target the goal actually being pursued? | Resolve the target/speed loophole below before freezing the environment or allocation oracle. A private target head alone does not solve it. |
| P3 | B's private round modifiers | Use temporary changes to movement delays, not workload: independent per-agent, per-goal offsets from `{-1,0,+1}`, fixed within each round. Rebalance their joint sampling for each condition. |
| P4 | C's commitment pressure | Use **higher switching delay**, `switch_extra_wait=3`, as the primary proposal. Do not also add a commitment deadline to this pilot. |
| P5 | E's even-sized crop | A world-aligned 2×2 patch containing self at one of its four corners; choose the corner uniformly per partner episode, independently of task variables, and keep it fixed for its 20 rounds. Provide that corner as proprioceptive metadata. |
| P6 | Horizon if 100 excludes meaningful joint-capability cases | First test 100 with the exact new dynamics. If it fails, propose one common revised horizon for A–E and regenerate every affected audit; do not silently drop all slow cases or change one condition alone. |

### The target/speed loophole

In current v3, the partner's target determines both its delay and its scripted BFS navigation. Removing BFS removes this binding. With an unrestricted movement head, an agent could declare RED to obtain a low `d_R` while walking toward BLUE. Requiring a matching label only when entering a goal still allows most travel at the faster delay, followed by one late switch. The destination-delay rule prevents a free instantaneous switch but does not by itself enforce goal-dependent travel capability.

We therefore need an explicit scientific choice:

1. **Treat target as a freely selectable movement mode.** Keep unrestricted learned movement and the literal destination-delay transition. Strategic use of the faster mode is then legal task behavior. Audit with an unrestricted joint oracle, and report the difference from goal-consistent allocation values. This is the least additional machinery, but it changes the interpretation of `(d_R,d_B)` from reliable capability for completing each goal to selectable movement modes.
2. **Require goal-consistent pursuit.** Specify and approve a binding rule before coding. Restricting moves to shortest-path progress would retain target-dependent travel but constrain learned navigation and can reveal offscreen goal direction through movement outcomes or masks in E. Automatic BFS would remove learned navigation. Retrospective travel charges would change the delay mechanism. None should be introduced implicitly.

**Recommendation:** decide P1 and P2 first, using a small counterexample grid and exact transition traces. If unrestricted movement and goal-specific completion capability are both essential, the task needs a more explicit physical definition of capability. Until then, the plan is actionable for preparation, but the baseline is not launch-ready. Do not claim that a free target label faithfully preserves the old capability mechanism.

## 3. What the repository currently implements

The maintained cohort is the five-seed, phase-fixed, direct-input `v1_matched`/`v2`/`v3` cohort described in [README](../README.md), [architecture](architecture.md), and [observation pipeline](observation_pipeline.md).

| Current mechanism | Joint-learning change |
| --- | --- |
| Only `agent_0` is optimized; `agent_1` navigates by scripted BFS. | Learn both agents' navigation and communication independently. |
| Old action encoding is `5 moves × 3 allocation codes`; only RED/BLUE codes are legal. | New versioned behavior/message/self-target schema; remove inherited allocation masks. |
| Ego's allocation forces the partner to the complementary goal. | Own-target control only; both agents can choose the same goal. |
| Only the scripted partner has a private movement-delay profile. | Both agents have private profiles and cooldowns; each observes its own profile. |
| Grid + previous ego allocation + `is_t0` enter a 128-unit GRU. | Agent-relative observation with own capability/state and received token. No previous-ego-allocation field. |
| Memory/recurrent PPO distinguish pre-observation reset from post-step termination. | Preserve this distinction separately for both learners. |
| Maintained jobs execute verified frozen sources/configs. | Prepare and freeze a new joint family; do not patch old scientific snapshots. |

Relevant source anchors are [environment](../jaxmarl/environments/coordination_grid/coordination_grid.py), [trainer template](../train/ippo_rnn_coordination_grid.py), [counterbalanced scheduler](../train/counterbalanced_scheduler.py), [preparation](../train/prepare_experiment.py), and [current PPO configuration](../train/config/ippo_coordination_grid.yaml). The active trainer is a preparation template; copying it is not equivalent to modifying the already frozen v3 training implementation. The historical `episode_scheduler.py` random sampler is not the maintained balancing mechanism.

## 4. Proposed environment and communication rules

### 4.1 State and observations

Store positions `[2,2]`, private base delays `[2,2]`, effective delays `[2,2]`, own-target registers `[2]`, cooldowns `[2]`, inbox tokens `[2]`, round/episode identity, round time, and the scheduled 20 layouts. B/C/D additionally store modifiers; E stores each agent's crop corner. Preserve collision and direct-swap blocking from the current task, with symmetric simultaneous resolution.

Each learner and its critic receive only:

- A/B/C/D: the five-channel full grid, ordered as walls, RED, BLUE, **self**, **other**.
- E: the five-channel 2×2 crop, an in-bounds mask, and the crop-corner encoding; out-of-bounds cells are padded as walls. No full-grid positions, goal coordinates, global map, offscreen partner location, BFS distances, or global legality hints.
- Own base profile, own effective profile/modifier when applicable, own target including `UNSET` at reset, own cooldown, `is_t0`, and round-local time/horizon fraction. Keep these proprioceptive fields consistent across compatible conditions.
- Inbox in `{NO_MESSAGE, M0, M1}`. `NO_MESSAGE` is not a third learned token.

Do not expose the other target, profile, modifier, cooldown, policy identity, scheduler index, oracle label, or debug `info`. An independent critic sees the same information as its actor. Full-grid visibility permits inference from visible motion; it does not reveal private registers. Both agents use the same field definitions, with self/other channel order exchanged.

### 4.2 Initialization, delays, and a single transition

Proposed v3-style initialization: at `t=0`, both choose their own initial targets, remain physically stationary, and may `WAIT` or send a token. Initial target choice is not charged as a switch; cooldowns remain zero. This is one rewarded, counted transition. D also forces its locomotion component to `WAIT` at `t=0`. There is no random complementary allocation.

For `t>=1`, after P2 is settled:

1. Sample both policies from their respective pre-step observations and independent recurrent states. Apply both self-target requests simultaneously. Never condition an agent's action on its partner's same-step action.
2. If target changes, replace its counter with the effective **destination** delay. In C add the configured switching wait. Unchanged requests preserve the counter. Physical movement/pursuit eligibility then follows the approved P2 rule.
3. For each agent, a positive counter suppresses movement and decrements by one. At zero, apply a requested directional move, subject to the task's physical rules, and reload that target's delay. A ready `WAIT` also reloads the delay; `SAY_M0/M1` follows exactly that same `WAIT` update. A blocked directional attempt consumes the ready slot too. Thus neither speaking nor wall-bumping bypasses cooldowns.
4. Resolve proposed positions simultaneously, then score physical success. No target-consistency reward is added. A round ends on physical success or the horizon.
5. Deliver each emitted token to the other agent's next observation. An absent token produces `NO_MESSAGE` on that next observation; there is no persistent last-token buffer. Speaking is allowed while cooling down.

Examples: with counter 2, `WAIT` and `SAY_M0` both leave the sender still and set counter 1. With counter 0 and delay 3, either leaves it still and sets counter 3; saying additionally delivers M0. A switch to destination delay 3 first loads 3, blocks that tick's movement, and leaves counter 2. Repeating the same target does not restart that wait.

At an intermediate round end, reset physical state, targets, and counters; preserve profiles and both recurrent states. Sample the next scheduled modifiers in B/C/D. Deliver a terminal-step token to the next round's first observation, since the partner is unchanged and delivery is defined as next-observation delivery. At the end of round 20, clear inboxes and recurrent states for the new partner episode. Record terminal actions/rewards before reset; never attribute them to the next layout.

### 4.3 Exact differences between A–E

| Pilot | Capabilities and private information | Behavior/message timing | Commitment and observation |
| --- | --- | --- | --- |
| A | Two stable private profiles for 20 rounds. | Seven behavior symbols; speaking replaces movement, no fee. | Own target revisable under destination-delay rule; full grid. |
| B | A plus each agent's private round modifier. | Same as A. | Same as A. |
| C | Same as B. | Same as B. | Add three ticks to destination delay on each actual post-initialization target change; full grid. |
| D | Same modifiers as B. | Locomotion `{UP,DOWN,RIGHT,LEFT,WAIT}` and message `{NONE,M0,M1}` chosen together. | B's switching rule and full grid; messaging can accompany eligible movement. |
| E | A's stable private profiles; no round modifiers. | Same seven behavior symbols as A; delivery independent of field of view. | Only the specified local 2×2 patch. |

**B:** propose `d_eff[i,g] = max(0, d_base[i,g] + modifier[i,g])`. Start from all nine two-component modifier vectors per agent, giving 81 joint modifier combinations per round before balancing. Modifiers are resampled independently of previous-round outcomes, fixed for that round, and private. Both base and temporary values are known to their owner from `t=0`. No extra service/work phase. Audit any clipping explicitly. The independent uniform modifier distribution is a candidate proposal, not a guarantee that final scenario weights remain uniform.

**C:** higher switching cost is paid in time, never as an extra reward deduction. With `switch_extra_wait=3`, a genuine change loads `d_eff[new_goal]+3`; subsequent unchanged actions resume ordinary cadence. Initial selection is free. Repeatedly requesting the same target costs nothing extra. Keep the higher-cost mechanism separate from an earlier hard deadline, so C versus B has one defined intervention. Confirm the value 3 using deterministic mechanical examples before freezing; do not tune it on pilot returns. Reversal and repeated-switch behavior must enter the oracle state.

**D:** use a structured categorical policy `p(target) p(move | target) p(message | target, move)`, with the same target interface as other conditions. A/B/C/E use `p(target) p(behavior | target)`. D has 15 move/message combinations before the target factor, versus seven behavior symbols elsewhere. Use the sum of selected component log-probabilities for one joint PPO ratio per agent. Movement obeys the same delay/collision rules; message emission remains possible when movement is suppressed. There is no token delivery before the partner chooses its same-step action. Regularize the exact current-policy joint entropy: `H(target) + E_target H(move|target) + E_target,move H(message|target,move)`; A/B/C/E use target entropy plus expected conditional behavior entropy. Enumerating the small 14/30-outcome spaces avoids weighting conditionals by old replay actions. Log all entropy components and apply the common coefficient to the full joint entropy, with the action-space difference reported.

**E:** a 2×2 patch has no central cell. Define upper-left crop offsets as `{(-1,-1),(-1,0),(0,-1),(0,0)}` in row/column coordinates relative to self: `patch[r,c] = grid[self_r+offset_r+r, self_c+offset_c+c]` for `r,c∈{0,1}`. Self appears at patch index `(-offset_r,-offset_c)`. Crop contents stay in world row/column order. Sample corners independently for both agents, balance all 16 joint corner assignments, and retain them for the episode. D4 transforms must also transform crop corners. Do not hide a full-grid mask inside the policy action distribution. Allow attempted moves into locally unknown cells and let ordinary walls/collisions determine outcomes, subject to the approved P2 rule. Any global pursuit restriction would need a fresh review of E's information contract.

## 5. Data selection: reproduce the old reasoning, not its labels

### 5.1 What the 1,096 examples actually guarantee

Saved provenance, rather than current generator defaults, describes the historical corpus:

| Selection stage | Survivors |
| --- | ---: |
| Unique geometrically valid candidates | 20,000 |
| Scripted-partner distance to both goals at least 3 | 12,587 |
| Allocation-balance screen across the old 24 partner profiles | 10,290 |
| Old feasibility screen | 10,290 |
| Oracle-minus-best-fixed reward gap at least 0.0358333 | 7,728 |
| Worst-case analytical completion below 60.8052 | 7,645 |
| Rank closest to 50/50 allocation; break ties by reward gap | 2,000 |
| Both **ego** distances in 1–8 | 1,597 |
| Sample without replacement for exact ego-distance marginals | 1,096 |

The source selection used seed 2026; its reward-gap threshold was the 25th percentile, and its horizon threshold was mean plus three standard deviations. The final maximum-entropy joint-bin selection and rounding gave **137 layouts at each distance 1–8 for each ego-to-goal marginal**. The selected 2,000 layouts all had 12 RED/12 BLUE optimal old assignments with no ties.

Evidence: [source manifest](../data_prep/grids_capability_selected_2000/manifest.json), [source filter stages](../data_prep/grids_capability_selected_2000/diagnostics/filter_stages.csv), [balanced manifest](../data_prep/grids_capability_selected_balanced_1096/manifest.json), [final validation](../data_prep/grids_capability_selected_balanced_1096/diagnostics/validation.json), [balancer](../data_prep/balance_ego_distances.py), and [capability selection](../data_prep/capability_selection.py). The current [builder](../data_prep/build_final_corpus.py) defaults to retaining all exact-balanced survivors, rather than reconstructing the old capped selection automatically.

Limits of that guarantee matter here:

- It balanced the ego role, not both roles. Partner distances range 4–15, with mean about 6.77, versus ego mean 4.5.
- The final subset no longer has equal D4-transform counts, even though its 2,000-layout source did.
- The saved [geometry audit](../data_prep/grids_capability_selected_balanced_1096/diagnostics/geometry_goal_dependence/summary.json) finds layout-invariant old optimal assignments across the 46 old profiles. That is a property of the old delayed-partner/undelayed-ego task.
- The old analytical completion formula assumes an undelayed ego and scripted partner; it ignores collision/switching dynamics. Its successful-round reward convention is also 0.01 below the actual environment reward. Do not reuse it for the new balance labels.
- The corpus uses familiar layouts for training and evaluation; it is not a held-out-layout generalization split.

### 5.2 Define the training example at the right level

A training opportunity is a **scenario**, not just a JSON grid:

`condition × layout × ordered pair of base profiles × role/color transform × round modifiers × crop assignment × rule version`.

Use the current 24 training profiles for **both** agents as the initial proposal: 576 ordered profile pairs. Retain identical-profile pairs, pairs specializing in the same color, and pairs specializing in opposite colors. These can create legitimate ties and coordination difficulty; do not inherit the old no-tie criterion.

Start with the old 1,096 as a reference candidate pool, and the source 2,000/generated candidates if needed. Compute the new condition-specific values first. Expand candidates only when the required distributions cannot be supported. The new corpus need not contain exactly 1,096 geometries. Log every expansion and exclusion. Do not require both original starts to meet the old ego-only 1–8 cutoff, which would change the task unnecessarily.

Keep the intended private-information prior explicit: initially uniform independent base profiles, independent private modifiers, and independent crop assignments. Prefer a common layout support crossed with all declared private-variable support and factored scenario weights. Filtering on an individual layout/profile/modifier tuple can make public geometry predict hidden partner information even when all color/role marginals look balanced. Remove or retain whole layout symmetry groups across that private-variable support where possible. Any required nonfactored exclusions/weights must be quantified and reviewed before launch.

### 5.3 Allocation values and feasibility

For every condition, define and save values for both complementary physical allocations: `0→RED,1→BLUE` and `0→BLUE,1→RED`. Distinguish terminal physical allocation from initial target choices. Compute the signed return advantage `Δ = V_RB − V_BR`, ties, feasible completion times, and the unrestricted best joint value. A path may revise targets, so a static distance comparison alone is not a valid label.

1. Screen all declared finite scenario combinations with fast, symmetric distance/delay bounds. For B–D include every supported modifier combination, with weights stated explicitly. Cache by effective profile pair and canonical geometry.
2. Validate a reference solver against actual environment transitions on small cases. Its state includes both positions, targets, counters, initialization/round time, and C's switching rules. It controls two learned-action interfaces; it must not reuse the scripted BFS partner. Compute return from actual reward timing.
3. Use exact joint search or certified lower/upper value bounds for ambiguous cases. Full-information search can omit messages because they confer no information to an omniscient controller. Keep paired symmetry cases equivalent, including alternate equal-length paths.
4. Require all selected/scheduled support to have completed feasibility and allocation-value checks. Bound intervals may be used only if they certify the classification and configured selection criterion. An unresolved interval is `UNKNOWN`, never an invented tie or sign. Apply a symmetric exclusion rule or finish solving it before launch.
5. Benchmark the solver before freezing corpus size: include easy/open, bottleneck, high-delay, tied, and switch-heavy cases. Record states expanded, cache reuse, bound-certification rate, peak memory and projected full-audit time. The original 1,096 layouts crossed with 576 pairs and 81 modifiers already yield 51,134,976 tuples for one B–D condition before transforms. Scale computation with symmetry canonicalization, effective-delay caching, staged bounds and exact search only where needed. Save coverage, unresolved counts, wall time, and cache keys. If the benchmark is impractical, revise the candidate geometry count or approved modifier support symmetrically, then repeat the audit; do not hide unresolved cases behind approximate labels.

Successful completion in `T` counted transitions gives `1 − 0.01(T−1)`; a timeout at horizon `H` gives `−0.01H`. Also compute discounted values with the training gamma, since PPO optimizes discounted return. Label which value drives each filter. Never substitute an old reward offset.

The full-information oracle is a physical upper bound. It is **not** the optimum achievable with private observations or E's crop. Separately evaluate feasible decentralized reference strategies and what information each agent has when choosing. E requires an information audit even if its omniscient physical values equal A's. D requires its own audit even where the no-message physical ceiling equals B's.

### 5.4 Balance with complete transformations

Form scenario groups closed under:

- **Color swap:** exchange RED/BLUE locations and both agents' capability components, modifier components, and target labels. Tokens remain uninterpreted; do not attach token identities to colors.
- **Role swap:** exchange starts, private profiles, modifier streams, crop metadata, and self/other observation roles.
- **D4:** rotate/reflect walls, starts, goals, directional actions, and E's crop corners together.

Choose and weight complete groups. A color or role swap reverses signed allocation advantage and preserves optimal team return/completion time; test these identities numerically. Transform the entire 20-round episode consistently for episode-level paired checks. Avoid counting duplicate symmetry members as independent evidence.

Generalize the current maximum-entropy quota approach to four role/color distance marginals and selected difficulty strata. First solve quota feasibility, then select/weight groups. Exact identical histograms for role/color counterparts are achievable through group closure; exact uniform distance counts may not be, and are not required. Preserve a broad stated distribution of distances, wall density, corridors, bottlenecks, and capability-induced value gaps.

Apply data-quality and horizon filters symmetrically. Retain a separately reported tie stratum. Adapt the old observability and oracle-benefit reasoning to the new condition; do not impose an arbitrary minimum communication benefit that predetermines success of the communication hypothesis. No learned policy outputs enter corpus selection.

### 5.5 Required distributions and plots for each pilot

| Audit | Plots/tables | Acceptance interpretation |
| --- | --- | --- |
| Selection provenance | Filter waterfall; rejected/source/final comparisons; exclusion reasons | Every scenario traceable; no unexplained role/color exclusions. |
| Geometry | Four role/color distance histograms; joint-distance heatmaps; starts/goals spatial maps; wall/bottleneck distributions | Paired role/color distributions agree; characterize geometry correlations. |
| Capabilities and allocations | 24×24 profile-pair maps of assignment advantage, ties, return, completion, and feasibility; margin histogram | Global signed advantage is symmetric; preserve capability-conditioned effects. |
| Nuisance shortcuts | Coordinate/direction/density correlations; geometry-only and own-observation predictors with symmetry-group-held-out folds; capability-preserving permutation baselines | Detect extra predictive power induced by selection, especially public-geometry leakage about private variables. Predictability from legitimate local capability is not itself bias. |
| Role and goal information | Per-role initial visible goals/partner; observation-equivalence classes with conflicting preferred allocations | No structural color/role shortcut; report unresolved private-information ambiguity. |
| Sampling | Planned/allocated/started/completed scenario matrices and joint-step-weighted exposure | Distinguish opportunity balance from duration and policy outcomes. |

Condition-specific additions:

| Pilot | Additional checks and visualizations |
| --- | --- |
| A | Same/opposite-specialist and identical-profile strata; value of own-only versus joint capability information; retargeting advantage under approved P2. |
| B | Base/effective delay histograms; clipping rates; modifier signs/magnitudes by role/color/round; allocation reversal maps relative to stable profiles; independence from prior outcomes. |
| C | Recompute values under the actual switching rule; switch burden versus destination delay; higher-cost value loss by role/color; information available before useful revisions. Include zero-, one-, and repeated-switch counterexamples. |
| D | B's checks under D's action/timing semantics; message opportunities during motion/cooldown; information available before the next movement decision. |
| E | All 16 crop-corner pairs; no/one/both-goals-visible counts; goal/partner first-discovery distributions under fixed reference exploration; local wall-pattern heatmaps; conflicting allocations within identical local observations. No privileged oracle-guided exploration presented as a local strategy. |

We cannot certify neutrality with respect to every imaginable measurable output. Predeclare these measurable input/opportunity dimensions, enforce exact transformation identities where possible, and publish remaining correlations. Do not force RED/BLUE to 50/50 inside a fixed raw capability profile: a true specialist should still make one allocation more valuable. Compare that conditional distribution to its correctly transformed counterpart instead.

### 5.6 Acceptance thresholds and audit artifacts

Before selection, freeze thresholds in the new audit configuration:

- Zero privacy/rule/symmetry violations, unreachable required goals, unknown selected feasibility classifications, or unmatched symmetry members.
- Exact equal planned weight for paired color/role groups; exact matching of their corresponding discrete marginals. Value-symmetry discrepancies must be zero for exact integer-time results, or within a documented numerical tolerance, proposed `1e-8` for derived float returns.
- Complete coverage of the declared 576 base profile pairs in each condition. Record any condition-specific support exclusions explicitly; a complete excluded pair requires a design review, not a silent missing row.
- Verify planned `P(partner profile/modifier | own profile, public geometry)` against the declared private-variable prior, plus analogous crop and modifier-history checks. Exact factored tables must agree to numerical precision. For empirical predictor/permutation diagnostics, predeclare folds, permutation count and a proposed 2-percentage-point upper confidence bound on excess held-out accuracy over the capability-preserving baseline. Material support-induced dependence or failure of that check requires corpus redesign or explicit review before launch, even if marginal color/role tests pass. Real information conveyed by visible movement after actions begin is permitted.
- No opaque scalar "balanced" score. Report each conditional distribution, tie mass, feasibility fraction, and difficulty histogram. Report known unequal duration exposure alongside the opportunity checks.
- Dry-run the production asynchronous queue on a fixed, policy-independent round-duration suite. Verify unique dispatch and allocated-count imbalance at most one for equal-weight queue strata. For weighted strata, verify the exact deterministic quota rule. Recompute all promised marginals on its finite prefix, including active workers.
- Save a signed pass/fail report for A, B, C, D, and E separately. A pass under A or B never substitutes for another condition's report.

Each condition's audit directory contains `rules.json`, `candidate_metrics.parquet`, `filter_stages.csv`, `selected_scenarios.json`, `allocation_values.parquet`, `symmetry_checks.json`, `conditional_distributions.csv`, `balance_report.md`, `balance_gate.json`, and exported PNG/PDF figures. Store commands, seeds, software/source hashes, corpus and schedule hashes, support coverage, weights, tie tolerance, and solver version. The gate references those exact bytes.

## 6. Sampling and realized exposure

Extend the maintained counterbalanced scheduler's principles: shared 20-layout packets, without-replacement layout passes, balanced joint-profile/scenario order, and unique dispatch from one global asynchronous queue. Keep each episode's two base profiles fixed, while supplying 20 scheduled modifier vectors where applicable. For a layout pool of size `N`, `20/gcd(N,20)` passes close an unpadded packet cycle; the old `N=1096` needs five passes.

Use matched root seeds: propose learner seed 1, schedule seed 2026, evaluation seed 12345. Derive independent deterministic substreams for learner 0/1 initialization, action sampling, layout order, profile pairs, modifiers, and crop corners. A shared seed must not initialize identical policies or accidentally correlate private modifiers. Keep policy RNG consumption from changing future scenario IDs.

Match A/E and B/C/D schedules wherever support is shared. Use common scenario IDs and the same base layout/profile/modifier draws, with condition-specific selection maps or weights. If a common balanced intersection supports all conditions without distorting them, prefer it. Otherwise keep separate passing corpora and report differences; also evaluate all conditions on a common eligible reference panel. Matching seeds cannot guarantee identical round starts or episode counts at a fixed joint-step budget because behavior changes completion speed.

During and after training record planned, allocated, episode-started, round-started, round-completed, and step-weighted exposure separately. Join actual transition IDs to scenario manifests and retain unfinished worker tails at the cutoff. Save audits after every update and a full report at the final checkpoint. Analyze role/color/profile/modifier/crop slices, not only marginal layout counts.

Slower tasks naturally occupy more steps; successful policies complete more rounds. Do not adapt sampling to equalize learned action, token, success, or allocation frequencies. Predeclare a fixed-step-prefix exposure diagnostic against the policy-independent queue simulation: report counterpart weight differences and total-variation differences by each major slice. Flag an observed difference above 5 percentage points for review, with queue-tail and duration explanations. It is a reporting trigger, not an automatic reason to force outcome equality. Scheduler duplication, missing logged steps, or use of unapproved scenarios is a hard failure.

## 7. Independent PPO pipeline

### 7.1 Rollout and update

Maintain `train_state[0]` and `train_state[1]`, separate carries and keys, and a shared environment/scheduler state. Collect one rollout with both policies fixed. At each tick, batch each agent's local observations, obtain its own actions/log-probabilities/value, and call the environment **once** with both actions.

Store local observations, actions, old component/joint log-probabilities, values, pre-observation reset masks, shared reward, post-step done, and round metadata. Keep agent/time/env axes explicit. Compute GAE and value targets independently, since local critics differ despite identical rewards. Normalize advantages within each learner's batch. Update both learners from the same collected rollout; do not update one, recollect against it, and call that the same batch.

Recurrent minibatches preserve chronological sequences, stored starting carries, and the pre-observation reset masks. Round ends do not reset memory or cut GAE; episode termination does. Rollout boundaries and the 20M budget cutoff bootstrap normally and are not fabricated terminal states. Prevent gradients through the partner's actions, messages, value, or hidden state. Communication learns through each sender's own PPO return signal.

### 7.2 Proposed common settings

| Parameter | Proposal per learner |
| --- | ---: |
| Parallel joint environments | 256 |
| Ordinary rollout length | 256 |
| PPO epochs / recurrent minibatches | 4 / 64 |
| LR | `5e-4` |
| LR schedule | Existing 5% warmup/cosine form, configured for all 306 pilot updates |
| Gamma / GAE lambda | `0.99 / 0.95` |
| PPO clip / value clip | `0.2 / 0.2` |
| Entropy coefficient / value coefficient | `0.01 / 1.0` |
| Gradient norm / Adam epsilon | `0.25 / 1e-5` |
| GRU / policy-value head width | `128 / 128` |
| Grid representation | Direct flattened symbolic inputs; no CNN or input normalization |

These inherit the maintained PPO settings where applicable. New private/message features change the observation width; E's patch produces a smaller grid input. Use the same recurrent/head widths and publish parameter counts. Use the same structured-action architecture for compatible conditions. D's larger behavior space is an intentional condition difference; disclose its entropy decomposition. Do not run an unrequested architecture/seed sweep.

### 7.3 Exact 20M accounting

```text
305 × 256 rollout ticks × 256 environments = 19,988,480
  1 ×  45 rollout ticks × 256 environments =     11,520
total                                     = 20,000,000 joint steps
```

Each learner sees 20M records; do not multiply the budget by two or sum duplicate team rewards. There are 306 PPO updates. Whole-sequence minibatching gives four environment trajectories per minibatch; the final batch uses 45-step sequences. Update schedule and exposure code must support the shorter final rollout instead of assuming a constant `updates × steps × envs`. A smoke prefix retains the full 306-update LR schedule. Log `joint_env_steps`, `agent_transition_records`, update index, actual rollout length, and distinct completed rounds/episodes.

### 7.4 Checkpoints and evaluation compatibility

Save lightweight paired actor/critic weights for evaluation and full resume checkpoints containing both optimizers, both carries, environment states, scheduler cursor/counters, RNG streams, actual joint-step count, and rule/action/observation schema versions. Resume must reproduce the next transitions and dispatch IDs. Always evaluate a pair from the same joint checkpoint unless a separately labeled cross-play analysis is requested.

Adapt the evaluator to two action-producing policies, local critics, own memories, terminal-transition logging, messages and target requests. Existing ego-allocation metrics and specialist labels cannot simply be renamed. Persist pre-reset terminal fields and masks in rollout artifacts; exclude padding and include the final successful action/reward.

## 8. Evaluation and communication interventions

Predefine a small fixed monitor panel for intermediate checkpoints and a larger final panel; evaluation uses separate environment/RNG streams and never consumes training steps. Save checkpoints near 1M, 5M, 10M, and exactly 20M, labeling the actual boundary reached. The primary final panel uses all 576 familiar ordered profile pairs, proposed four 20-round episodes each, with condition-balanced scenarios. Use identical episode IDs across compatible panels. Freeze evaluation scenario sampling/aggregation weights and panels before pilot training, then load the selected trained checkpoint. Primary evaluation samples categorical actions as in the maintained evaluator; any greedy-action result is separately labeled.

Report:

- Team return per round and 20-round episode, counted once; success rate; timeout rate.
- Completion-time distributions on successful rounds, plus timeout fraction and a horizon-censored time measure across all rounds. Separate initialization from later ticks in secondary plots.
- Physical final allocation and value regret under the approved oracle, target revisions, collisions, same-goal pursuit, and capability-conditioned behavior.
- Message rate by sender, M0/M1 proportions, timing, bursts, cooldown use, and frequency before target changes; per-round progression from 1 through 20.
- Conditional metrics by joint capability type, modifiers, crop/visibility, geometry, role and goal. Learned asymmetries are results, not balancing targets.

Separate familiar-profile performance, held-out base profiles for one/both agents, and any actual held-out effective-delay support. The old 22 test profiles can supply a base-profile panel, but **B–D's ±1 modifiers introduce old novel scalar values 0, 5, and 6 during training**. Do not call that panel unseen-scalar generalization. Publish base/effective support intersections and define a new effective-support holdout before making that stronger claim. Layout generalization is optional and separately labeled; split full symmetry groups if added.

### Message-content ablations

Freeze the trained pair and rerun matched evaluation scenarios under these delivery interventions. Sender actions, speaking opportunity costs, and D's locomotion are untouched by the intervention:

| Intervention | Purpose |
| --- | --- |
| Normal delivery | Reference. |
| Drop all received messages to `NO_MESSAGE` | Total dependence on delivered communication; also changes receipt timing information. |
| Replace every delivered token with a fixed token; test both constants | Preserve receipt timing, remove M0/M1 distinction. |
| Replace emitted tokens using an independent, shuffled reference token bank within condition/sender/round/time strata | Preserve message presence on the current trajectory while disrupting content alignment. Preserve donor frequencies in expectation; save the bank and realized replacements. |
| Flip M0↔M1 in one direction, then the other, then both | Test sensitivity to learned token identity and directional dependence. A fixed flip is not equivalent to eliminating all information. |

Build the donor bank from a disjoint normal-delivery evaluation panel, freeze it, and draw replacements only when the counterfactual sender actually emits. Use independent intervention RNG streams, so replacement does not perturb action-sampling randomness. Counterfactual send counts and token frequencies may differ from normal trajectories; do not promise exact preservation of their pooled frequencies. Paired rollouts can diverge after an intervention; compare outcomes with common initial scenarios rather than asserting the resulting trajectories stay matched. Also fork selected saved pre-message states/carries, intervene on one inbox observation, and compare receiver action/target distributions. Exact token permutations are possible for this fixed set of receiver-state probes. This distinguishes correlations in messages from an immediate causal effect on behavior.

Report paired changes in return, success, completion, and receiver actions with intervals obtained by resampling whole 20-round episodes. With one training seed, these intervals quantify evaluation variability, not robustness across training seeds. Token-use correlations alone do not establish useful communication. Drop ablations may be outside the trained observation distribution; timing-preserving content interventions provide a narrower check. No additional no-communication training condition is included in the five requested pilots.

## 9. Repository structure and experiment identity

Proposed additions follow the current top-level layout while using a distinct `joint` namespace:

```text
docs/joint_learning_pilots_plan.md
jaxmarl/environments/coordination_grid_joint/
  __init__.py
  coordination_grid_joint.py       # state, simultaneous transition, local observations
  rules.py                        # A–E rule definitions and action schemas
train/joint/
  experiment.py                   # joint family identity, seed, condition inventory
  networks.py                    # local observation encoder and structured policy heads
  ippo_rnn.py                    # two independent PPO updates, one environment rollout
  scheduler.py                   # joint-profile/scenario queue
  exposure_audit.py
  prepare.py                     # freeze and verify new joint sources/configs
train/config/joint/
  base.yaml
  A.yaml  B.yaml  C.yaml  D.yaml  E.yaml
data_prep/joint/
  build_corpus.py
  allocation_values.py            # new joint dynamics, no scripted-partner formula
  balance_audit.py
  plot_balance.py
eval/joint/
  evaluate.py
  message_ablations.py
  report.py
bash/submit_joint_pilots.py
tests/coordination_grid_joint/
tests/train_joint/
tests/data_prep_joint/
tests/eval_joint/
```

Register `coordination_grid_joint` separately. Reuse pure geometry/population utilities through imports where safe; new scientific rules, trainer and scheduler have explicit modules. Avoid source-string replacements to convert old allocation protocols. Do not move or rewrite existing v1_matched/v2/v3 files, manifests, sealed contracts or results. Add a short README index entry when implementation exists.

Store new logical artifact paths under:

```text
data_prep/joint_corpora/<corpus_version>/<A-E>/{layouts,audits}/
train/manifests/joint/<batch>.json
train/run_snapshots/joint/<batch>/
train/train_logs/joint/<batch>/<A-E>/seed_1/
train/sampling_audits/joint/<batch>/<A-E>/seed_1/
eval/joint_results/<batch>/<A-E>/seed_1/
```

Continue the `GRID_ARTIFACT_ROOT` convention for large arrays/checkpoints/rollouts, with repository-relative logical paths and immutable manifests. Every artifact records `experiment_family`, condition, both learner IDs, joint-step budget, training/evaluation seeds, corpus/schedule hashes, action/observation/rule versions, and source revision plus file hashes. Identifiers such as `joint_ippo_comm_v1_A_seed1` must never be confused with old `v3_rnn_diverse_influence` policies.

## 10. Implementation sequence and launch gates

| Stage | Concrete deliverable | Completion gate |
| --- | --- | --- |
| 0. Review this plan | Agreed P1–P6; exact capability/pursuit/action/timing contract | Resolve P2 with explicit traces and approve the interpretation of seven actions. |
| 1. Environment and reference model | New symmetric environment, all A–E rules, new joint allocation solver | Mechanical, privacy, symmetry and solver agreement checks pass. |
| 2. Corpus preparation | Separate selected support and audit bundle for each A–E | Every condition has a passing balance gate tied to frozen bytes. |
| 3. PPO and scheduling | Two learner states, unique queue, exact budget, resumable checkpoints | Replay, independence, accounting, exposure and resume checks pass. |
| 4. Evaluation | Paired-policy rollouts, metrics, content interventions, report templates | Terminal handling and intervention causal/timing checks pass. |
| 5. Preflight | CPU smoke per condition; production-shape GPU smoke; frozen manifest | Both learners update; finite losses; no leakage; all integrity/audit checks pass. |
| 6. Review launch package | Five resolved configs, audit summaries, seeds, source/schedule hashes, resource estimate | User agrees the plan/package is ready to launch. |
| 7. Requested pilots | One 20M joint-step run per A–E, final paired evaluation | Exact budget, valid paired checkpoints and realized-exposure reports. |

Stages are dependencies, not five independent opportunities to launch early. Estimate wall time and memory from the production-shape smoke before selecting Slurm time/GPU requests. Use the matched single seed only; reruns caused by infrastructure failure resume the same state and remain distinguishable from scientific restarts.

Proposed launcher interface, to implement later:

```bash
python bash/submit_joint_pilots.py prepare --conditions A B C D E --seed 1
python bash/submit_joint_pilots.py verify MANIFEST
python bash/submit_joint_pilots.py validate MANIFEST
python bash/submit_joint_pilots.py submit MANIFEST
```

`prepare` and `verify` never submit training jobs. `submit` verifies the frozen per-condition balance gates, rule/schema hashes, completed preflight and exact budget. The review status must not be inferred from the existence of a Markdown file.

Meaningful verification includes:

1. Either agent's action changes only its own target, attempted movement and emitted token; both same-target choices remain possible; no complementary assignment or BFS control is hidden in the new path.
2. Reward equality, physical success, collision/swap symmetry, t0 counting, timeouts, actual destination-delay loading, and identical WAIT/speech cooldown updates.
3. Token delivery at `t+1`, simultaneous two-way delivery, one-tick inbox lifetime, correct intermediate-round delivery, and final-episode clearing. D can move and send in one transition.
4. Actor/critic privacy, including E's crop and masks. Perturb hidden partner fields while holding an agent's visible observation fixed and verify unchanged outputs. Test that E observations cannot reveal offscreen geometry by direct encoding.
5. B modifiers private/fixed within a round; both base profiles stable for 20 rounds; C charges exactly real switches; crop assignments independent of capability/color.
6. Reference-solver paths replay through actual transitions and reproduce completion/returns; color/role/D4 counterparts preserve the required values. Include the P2 exploitation counterexample.
7. Two real rollout losses update both policies; updating one parameter/optimizer tree leaves the other unchanged. Recurrent collection and replay agree for both agents across round and episode boundaries.
8. The shorter final rollout yields exactly 20M joint steps, all step IDs are unique, no artificial terminal is added, exposure totals reconcile, and checkpoint resume reproduces continuation.
9. Evaluation uses the paired checkpoint and local carries; all valid terminal actions are retained; interventions alter delivered observations at the specified time without rewriting sender movement or reward.

The immediate deliverable is this review draft. Implementation, final corpus filtering, smoke jobs, and the five scientific launches follow only after the task contract and plan are agreed.
