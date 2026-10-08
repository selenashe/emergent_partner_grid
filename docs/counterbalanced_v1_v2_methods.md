# Methods: counterbalanced CoordinationGrid experiments

This draft describes the completed counterbalanced runs recorded in `selenashe/emergent_partner_grid` at commit `e0878e5be345180711e73fc2e5381e249ecac348` (October 7, 2026, Pacific time). Behavioral analyses include learner seeds 1–10. The completed capability-decoding analyses described below include seeds 1–5. The methods describe the executed experiment; implementation qualifications and a source map follow the main text.

**Experimental task and episode structure.** We studied whether a reinforcement-learning agent develops internal information about a partner's hidden capabilities through repeated cooperative interaction. The task, CoordinationGrid, contained two agents, a red goal, a blue goal, and impassable walls on a \(7\times7\) grid. The ego agent was learned; its partner followed a fixed navigation algorithm. A round succeeded when the agents simultaneously occupied different goals, in either red–blue arrangement. The ego controlled its own movement and, in the influence conditions, the partner's assigned goal. Navigation and allocation were optimized using the same team reward.

One partner episode comprised 20 rounds with an unchanged partner capability profile. Each round reset the agents' positions and geometry using the next layout in the episode's layout sequence. Rounds ended on success or after 100 environment transitions. An unsuccessful round also advanced to the next round. The environment signaled episode termination only after the twentieth round; recurrent memory was retained across earlier round boundaries and reset before interacting with the next partner. Thus, an episode contained at most 2,000 transitions and could contain fewer when rounds ended successfully. We distinguish a *transition* (one simulation tick), a *round* (one goal-completion attempt), and a *partner episode* (20 attempts with the same capability).

**Hidden partner capabilities and navigation.** A partner's capability was a pair

\[
\mathbf c=(d_R,d_B),
\]

where \(d_X\) was the number of wait ticks between scheduled partner moves toward goal \(X\). Smaller values indicated faster movement: \(d_X=0\) permitted a move every tick, and \(d_X=k\) permitted a move every \(k+1\) ticks. The ego had no movement delay. For a fixed goal, the partner selected a shortest-path move from a breadth-first-search (BFS) table computed from the walls and goal location. BFS explored neighboring cells to find the fewest moves to the goal, with deterministic directional ordering (up, down, right, left). The partner's cooldown began at zero each round. Both agents stayed still on the initialization tick, \(t=0\), and the partner could first attempt movement on \(t=1\), the second transition counted from round start. After a permitted move attempt, its cooldown was set to the delay for its current goal; on waiting ticks, the counter decreased toward zero. A blocked move attempt still consumed this scheduled opportunity and reloaded the cooldown.

Movement was simultaneous. Attempts to enter walls or leave the grid left that agent in place. If both agents proposed the same cell, or proposed exchanging their occupied cells, both stayed at their previous positions. Consequently, observed displacement reflected both the partner's movement cadence and interactions with the ego. The partner's capability affected its cadence, while its path-choice algorithm was identical across profiles. The scripted partner was not trained.

The diverse training population contained 24 specialists:

\[
\mathcal C_{\mathrm{train}}=(F\times S)\cup(S\times F),
\qquad F=\{1,2,3\},\quad S=\{4,7,8,9\}.
\]

Each specialist was relatively fast at one goal and slow at the other. The single-partner control was trained only with \((1,4)\). The evaluation population contained 22 disjoint profiles:

\[
\begin{aligned}
\mathcal C_{\mathrm{novel}}={}&(U\times V)\cup(V\times U)\\
&\cup(W\times\{0\})\cup(\{0\}\times W),\\
U={}&\{0,1,2,3\},\qquad V=\{5,6\},\qquad W=\{7,8,9\}.
\end{aligned}
\]

Every novel profile included at least one scalar delay from \(\{0,5,6\}\), which was absent from policy training. This tested generalization to new delay values as well as new capability pairs. Capabilities remained fixed throughout evaluation episodes, including when the partner changed its assigned goal.

**Observations and available information.** The ego received a \(7\times7\times5\) binary tensor identifying walls, the red goal, the blue goal, its own current position, and the partner's current position. The partner was visible from round initialization onward. Observations also contained a three-entry one-hot vector for the ego's previous allocation request, with codes NONE, RED, and BLUE, and a Boolean indicator of round initialization. At v2 initialization, the allocation vector instead supplied the environment's initial assignment. The initialization indicator selected the legal-action mask; it was not concatenated into the learned observation embedding. The network received no partner identity, capability values, cooldown counter, round index, numerical time counter, or reward input. Capability labels recorded for analysis were not policy inputs or supervised training targets.

**Allocation protocols.** Both versions encoded an action as a single integer

\[
a_t=3m_t+q_t,
\]

where \(m_t\in\{\mathrm{up},\mathrm{down},\mathrm{right},\mathrm{left},\mathrm{stay}\}\) and \(q_t\in\{\mathrm{NONE},\mathrm{RED},\mathrm{BLUE}\}\). There were 15 output logits, but the legal-action mask restricted which combinations could be sampled at each phase. A RED request assigned BLUE to the partner; a BLUE request assigned RED to the partner. The request expressed the ego's intended role, while the ego's physical navigation remained controlled by its learned movement policy. Success depended on occupying distinct goals, rather than on checking that the ego's occupied goal matched its request. Allocation codes had fixed meanings across partners; no learned vocabulary or partner-dependent symbolic message channel was present.

| Property | v1: fixed allocation | v2: online allocation |
| --- | --- | --- |
| Round initialization | Ego chose RED or BLUE; both agents stayed | Environment sampled RED or BLUE uniformly; both agents stayed |
| Legal initialization actions | STAY + RED or STAY + BLUE | Only STAY + the supplied assignment |
| Legal actions during movement | Five moves, each paired with NONE | Five moves, each paired with RED or BLUE |
| Influence condition | Initialization request fixed the partner's goal for the round | Each movement-tick request could preserve or change the partner's goal |
| No-influence condition | Partner assignment alternated by round parity | Partner retained the random initial assignment for that round |
| NONE code | Required during movement | Reserved and always masked out |

In v1, the partner committed at \(t=1\) to the goal complementary to the request selected at \(t=0\), and retained it until the round ended. Thus, the initial allocation could use information accumulated in previous rounds, but it could not be revised after observing the partner's movement in the current round. In the no-influence control, the internal ego-role assignment was RED on even zero-based rounds and BLUE on odd rounds, assigning the opposite goal to the partner regardless of the ego's request. This gave ten rounds of each partner goal per complete episode.

In v2, a fresh initial assignment was sampled independently of capability at every round reset and made visible to the ego. The initialization action was forced, with probability one. Beginning at \(t=1\), the ego jointly chose movement and a RED or BLUE request on every tick. With influence enabled, the request determined the partner's goal before that tick's movement. Repeating the assignment preserved the cooldown. A genuine goal change loaded the destination goal's delay *before* checking whether the partner could move; this made reassignment carry a timing cost rather than bypass an existing cooldown. A zero destination delay permitted immediate movement. With influence disabled, requests did not change the partner's initial assignment. The allocation observation subsequently echoed the ego's requests in both conditions, rather than exposing an additional internal-assignment variable. V2's first movement-tick decision occurred before seeing displacement in the new round, but later decisions could use current-round behavioral evidence.

**Experimental conditions.** We trained four conditions separately under each protocol:

| Condition | Temporal architecture | Policy-training profiles | Requests controlled partner goal? |
| --- | --- | --- | --- |
| Diverse RNN with influence | 128-dimensional GRU | All 24 specialists | Yes |
| Diverse feedforward control with influence | Dense replacement for the GRU | All 24 specialists | Yes |
| Single-partner RNN with influence | 128-dimensional GRU | Only \((1,4)\) | Yes |
| Diverse RNN without influence | 128-dimensional GRU | All 24 specialists | No |

These comparisons tested the contribution of recurrent memory, diverse partner experience, and control over task allocation. The feedforward control received the same current observation and previous-request channel but did not carry a learned hidden state across ticks. Thus, it removed recurrent memory while retaining any information available in current physical state or the previous request. The design comprised these four comparisons under two protocols, rather than a full factorial crossing of all architecture, population, and influence settings. Ten learner seeds (1–10) yielded 80 policies in total. Seeds 1–5 belonged to the original counterbalanced batch; seeds 6–10 extended that batch using its frozen scientific sources, layouts, schedules, and hyperparameters.

**Layout generation and selection.** Both protocols used exactly the same 1,096 layouts. Candidate grids were generated by sampling a wall probability uniformly from \([0.15,0.60]\), sampling walls independently conditional on that probability, and placing the two starts and two goals in distinct empty cells. Accepted candidates required all four start-to-goal routes to be reachable, a shortest-path start-to-start distance of at least three, and a goal-to-goal distance of at least three. Generation also bounded the absolute difference between the two assignments' summed path lengths by 20. The minimum geometric switching-cost threshold was zero, and the opposite-preference threshold was 99; these were permissive settings, rather than a positive minimum switching-cost requirement. Structural quantities such as junctions, dead ends, and path overlap were recorded as diagnostics.

The recorded source corpus began with 20,000 unique accepted candidates, using seed 2026. Capability-sensitive selection used a static analytical model with independent shortest paths and no collisions or online reassignment. Let \(\ell_{E,X}(L)\) and \(\ell_{P,X}(L)\) denote ego and partner distances to goal \(X\) on layout \(L\). If the ego took goal \(X\), and the partner took the other goal \(\bar X\), the predicted completion time was

\[
T_X(L,\mathbf c)=\max\!\left\{1+\ell_{E,X}(L),\;2+\bigl(\ell_{P,\bar X}(L)-1\bigr)(d_{\bar X}+1)\right\}.
\]

The ego term included the stationary initialization tick. The partner term placed its first move on the second transition and subsequent moves \(d_{\bar X}+1\) ticks apart. Completion required both agents to be on their goals, so the larger arrival time determined joint completion. Selection used the historical score \(Q_X=1-0.01T_X\) when completion was within the selection horizon of 500 ticks, and \(-0.01\times500\) on timeout. This historical score differs by one step penalty from the successful return actually paid by the environment, as detailed below.

Selection required both partner-to-goal distances to be at least three, to provide opportunities to observe its movement cadence. It then required the non-tied fraction of training profiles favoring ego-RED to lie in \([0.25,0.75]\), a tie fraction no greater than 0.5, and an analytical oracle success fraction of at least 0.99. We quantified the value of conditioning on capability as

\[
\Delta(L)=\mathbb E_{\mathbf c}\!\left[\max_{X\in\{R,B\}}Q_X(L,\mathbf c)\right]
-\max_{X\in\{R,B\}}\mathbb E_{\mathbf c}\!\left[Q_X(L,\mathbf c)\right],
\]

with uniform weighting over training profiles. The first term allowed a different assignment for each capability; the second selected one assignment for the layout without capability information. Layouts were retained when \(\Delta(L)\ge0.0358333\), the lower-quartile threshold calculated from the preceding survivors. A horizon screen rejected layouts whose worst-profile oracle completion reached the survivor mean plus three standard deviations (60.8052 ticks). From the remaining candidates, 2,000 layouts were selected by proximity to an ego-RED optimum fraction of 0.5, with larger \(\Delta\) and then candidate index as tie-breakers. Rotations and reflections were distributed across the eight square symmetries in this source corpus. Survival counts were 12,587 after observability, 10,290 after allocation balance and feasibility, 7,728 after the reward-gap filter, and 7,645 after the horizon screen.

The final balancing stage excluded 403 of the 2,000 layouts because either ego-goal distance exceeded eight. From the 1,597 eligible layouts, it sampled 1,096 with exactly 137 layouts at each ego-to-red distance 1–8 and exactly 137 at each ego-to-blue distance 1–8. Simultaneous marginal quotas were obtained using a maximum-entropy fractional selection over joint distance bins, expectation-preserving randomized cycle rounding, and uniform sampling without replacement within each bin, with seed 2026. Other geometry variables were not additional selection criteria at this stage, although conditioning on ego distances could change their distributions. Online symmetry augmentation was disabled during training and evaluation.

An exhaustive recorded audit of the final corpus found 12 red-optimal and 12 blue-optimal training profiles per layout, and 11 of each among novel profiles, with no equal-time ties. Under uniform profile weighting, any classifier given only grid geometry therefore had a 50% ceiling for identifying the best static assignment. Across all 50,416 layout–profile combinations, the best static assignment gave the partner its faster goal; the worst optimal analytical completion was 58 ticks. The minimal capability distinction needed for this static allocation rule was consequently *which goal was faster*, not necessarily the exact values of both delays. Recoverability of the numerical delays was tested separately through representation probes. These properties concern the static analytical reference; they do not establish a globally optimal policy for v2 switching trajectories or collision avoidance. All 1,096 layouts were available during policy training and evaluation; there was no held-out-layout split.

**Counterbalanced partner and layout exposure.** To pair layout histories across profiles, we preallocated a common sequence of 20-layout packets. For each packet, one partner episode was allocated to every profile in a seeded random profile order before the next packet was allocated. All profiles within a packet therefore received exactly the same ordered layout history. The profiles could execute asynchronously; pairing referred to their scheduled histories rather than simultaneous completion.

Layout sequences consisted of independently shuffled passes through all \(N=1096\) layouts without replacement within each pass. With \(R=20\) rounds per episode, the smallest number of complete passes that could be partitioned into full episodes was

\[
P=\frac{R}{\gcd(N,R)}=5.
\]

Each cycle consequently contained \(PN/R=274\) episodes per profile and five visits to every layout per profile. For 24 profiles, this was 6,576 episodes, or 131,520 rounds, per complete scheduling cycle. The single-profile control used the same layout-packet prefix with one profile. The layout and profile-order random streams were separate, both derived from schedule seed 2026, so profile permutations did not change layout ordering. Both protocols and all learner seeds reused the frozen schedules.

Training used one global episode queue across 256 parallel environments. When an environment completed a partner episode, it received the next unique queued episode; simultaneous completions were assigned in stable environment-slot order. At every queue prefix, allocated episode counts differed by at most one between profiles. This counterbalanced scheduled episodes rather than elapsed simulation ticks: slower partners could consume more transitions. The fixed training cutoff left an active episode in each worker, so realized completed-round counts could also differ. Sampling audits retained profile-by-layout counts for rounds started, rounds completed, and transitions, as well as the unfinished episode tails. Preallocated schedule capacity exceeded the maximum needed under the step budget; the preallocation count was not the number of episodes actually experienced.

**Policy and value architecture.** A convolutional neural network (CNN) converted spatial observations into learned grid features. It used six convolution layers with filter-count/kernel-size pairs \((128,1),(128,1),(8,1),(16,3),(32,3),(32,3)\), ReLU activations, and the Flax convolution defaults for stride and padding (unit stride and SAME padding). The flattened output was projected to 64 dimensions. The three-entry allocation vector was projected to eight dimensions with ReLU. Concatenating these two embeddings and applying a dense ReLU layer yielded 128 features, followed by layer normalization.

For recurrent policies, a gated recurrent unit (GRU) updated a 128-dimensional memory:

\[
\mathbf z_t=\operatorname{LayerNorm}(f_\theta(o_t)),\qquad
\mathbf h_t=\operatorname{GRU}_\theta(\mathbf z_t,\mathbf h_{t-1}).
\]

The GRU learned which aspects of the current observation to retain or update. Memory was initialized to zero for each new partner episode. Separate actor and critic heads read the same recurrent state. Each head contained a 128-unit ReLU layer; the actor produced 15 action logits and the critic produced a scalar estimate of future reward. Illegal logits were set to negative infinity before constructing a categorical action distribution. The feedforward control replaced the GRU with a 128-unit dense ReLU layer and ignored its dummy carry. It shared the encoder and head design; this substitution did not exactly match parameter counts.

The explicitly initialized convolution, encoder projection, and feedforward-replacement kernels used orthogonal initialization with gain \(\sqrt2\); head hidden layers used gain 2, the action readout gain 0.01, and the value readout gain 1. Their biases were zero. GRU and layer-normalization initialization followed the frozen Flax module implementation. No auxiliary capability-prediction objective was applied to the policy.

**Reward and reinforcement learning.** The per-transition team reward was

\[
r_t=\begin{cases}
1,&\text{if the agents occupied distinct goals after the transition},\\
-0.01,&\text{otherwise}.
\end{cases}
\]

Success ended the round immediately. A successful round lasting \(K\) transitions therefore returned \(1-0.01(K-1)\); a 100-transition timeout returned \(-1\). Episode return summed the realized rewards across all 20 rounds. This rewarded successful and efficient joint completion, without supplying capability supervision.

We trained only the ego using proximal policy optimization (PPO). PPO alternated between sampling behavior with the current policy and adjusting its parameters, while clipping changes in the probabilities assigned to sampled actions. Each update collected 256 transitions from each of 256 environments, for 65,536 transitions. Generalized advantage estimation (GAE) used

\[
\delta_t=r_t+\gamma(1-D_t)V_{t+1}^{\mathrm{old}}-V_t^{\mathrm{old}},\qquad
\widehat A_t=\delta_t+\gamma\lambda(1-D_t)\widehat A_{t+1},
\]

where \(D_t\) indicated the end of the full partner episode, \(V_t\) was the critic's estimate, \(\gamma=0.99\), and \(\lambda=0.95\). Intermediate round endings did not stop the value bootstrap. The advantage estimated how much better a sampled outcome was than the critic's prediction and was standardized within each optimization minibatch. Value targets were \(\widehat A_t+V_t^{\mathrm{old}}\).

For \(\rho_t(\theta)=\pi_\theta(a_t\mid o_{\le t})/\pi_{\mathrm{old}}(a_t\mid o_{\le t})\), the minimized loss was

\[
\mathcal L(\theta)=-\mathbb E\left[\min\left(\rho_t\widehat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\widehat A_t\right)\right]
+c_V\mathcal L_V-c_H\mathbb E[H(\pi_\theta)],
\]

with \(\epsilon=0.2\), \(c_V=1\), and \(c_H=0.01\). The entropy term encouraged exploration over legal actions. The value loss was half the mean of the larger squared error from the current value prediction and its prediction clipped within \(\pm0.2\) of the old prediction. This used the same clip width as the actor objective.

Each rollout underwent four optimization epochs with 64 minibatches per epoch. Minibatches shuffled environment sequences while retaining their time order; each contained four sequences of 256 ticks (1,024 transitions). Recurrent replay began from each sequence's recorded initial carry. Memory carried forward across rollout blocks during interaction, while gradient backpropagation was limited to the replayed 256-tick block. Adam used learning rate \(5\times10^{-4}\), epsilon \(10^{-5}\), and default moments \((0.9,0.999)\). Gradients were clipped to global norm 0.25. The learning rate increased linearly from zero during the first 45 PPO updates and then followed a cosine decay over the remaining 870 updates. Actions were sampled categorically during both training and evaluation.

| Training setting | Executed value |
| --- | --- |
| Parallel environments | 256 |
| Transitions per environment per rollout | 256 |
| PPO rollout/update count | 915 |
| Optimization epochs / minibatches per epoch | 4 / 64 |
| Minibatch size | 1,024 transitions, preserving four time sequences |
| Nominal / actual transitions per policy | 60,000,000 / 59,965,440 |
| Adam peak learning rate / epsilon | \(5\times10^{-4}\) / \(10^{-5}\) |
| Learning-rate schedule | 45-update linear warmup; 870-update cosine decay |
| Discount / GAE parameter | 0.99 / 0.95 |
| Actor and value clip width | 0.2 |
| Value / entropy coefficient | 1.0 / 0.01 |
| Maximum global gradient norm | 0.25 |
| Learner / layout-schedule / behavioral evaluation seeds | 1–10 / 2026 / 12345 |

Integer division of the nominal transition budget gave \(915\times256\times256=59,965,440\) actual transitions per policy. The seed extension added independently initialized policies rather than continuing optimization of the first five seeds. It reused each protocol's original frozen trainer, including the v1 recurrent-replay behavior qualified below.

**Behavioral evaluation and aggregation.** We evaluated final checkpoints without weight updates on the common 1,096-layout pool. Each policy completed 20 partner episodes for each of the 24 training-population profiles and each of the 22 novel profiles, yielding 480 training-population episodes (9,600 rounds) and 440 novel episodes (8,800 rounds). Evaluation used seed 12345, pinned each episode's capability to the designated profile, and sampled its 20-layout sequence uniformly with replacement from the familiar layout pool. Adaptation within an evaluation episode occurred through the recurrent state, not changes in network parameters.

For the single-partner-trained policy, the 24-profile evaluation slice was the *shared training population*: only \((1,4)\) was actually familiar from its own training. We therefore use this population-level label without implying that all 24 profiles were familiar to every condition. No performance-based seed exclusion or checkpoint selection entered the ten-seed behavioral aggregate.

Primary measurements were the fraction of completed rounds that succeeded, mean undiscounted return per 20-round episode, and mean episode transition count. Success was measured per round, rather than as the fraction of episodes whose final round succeeded or whose every round succeeded. We also measured successful-round completion time and success by round index to describe adaptation with partner experience. Because trajectories were collected using a fixed scan of up to 2,000 transitions, metrics and representation features included steps through the first full-episode terminal transition and excluded all later scan entries.

V1's evaluator additionally measured whether the ego's initialization request maximized the static analytical assignment score and its analytical regret, \(\max_XQ_X-Q_{\mathrm{requested}}\), with tied optima excluded from allocation-accuracy denominators and zero regret on ties. For the no-influence condition, request accuracy described a hypothetical assignment choice; the request did not determine the realized partner goal. V2's evaluator measured the fraction of movement ticks on which the partner was assigned its relatively faster goal, the corresponding fraction for ego requests, and assignment-switch counts. Random initialization ticks were excluded from these online-decision measures. These fractions were weighted over decision ticks, so longer rounds contributed more samples. Faster-goal adherence was not treated as dynamic optimality: remaining distances, collisions, cooldown phase, and reassignment costs could favor retaining a slower-goal assignment at a particular moment.

Each condition–protocol statistic was computed for each trained policy and then averaged with equal weight across ten learner seeds. Reported standard deviations used the sample standard deviation across those policies (denominator \(10-1\)), retaining all seeds. Per-profile and round-index analyses used the same ten-policy aggregation. Separate five-seed selected-policy visualizations chose policies using logged training return; these were supplementary and did not alter the ten-seed aggregate.

**Post hoc capability decoding and visualization.** The completed primary representation analysis covered the three recurrent conditions and learner seeds 1–5 for each protocol (15 networks per protocol). The feedforward control had no recurrent state to analyze. For each network, saved evaluations supplied 920 episodes across all 46 capability profiles. The saved vector was the 128-dimensional GRU state *after processing the current observation and before producing the associated action*. Thus, it was the state the policy used for that tick's decision, rather than the preceding carry or a state updated from the subsequent physical outcome.

For episode \(e\), valid length \(L_e\), and cutoff \(u\), a probe feature was the mean memory vector over the valid episode prefix:

\[
\overline{\mathbf h}_e(u)=\frac1{n_e(u)}\sum_{t=1}^{n_e(u)}\mathbf h_{e,t},
\qquad n_e(u)=\min(u,L_e).
\]

Cutoffs were \(u\in\{1,50,100,150,200,250,300,350,400\}\). Additional curves used the cumulative prefix through each of the 20 round-ending ticks. These were averages of all history up to that round, rather than averages of that round alone. Terminal-producing decisions were included; padded scan entries were excluded. Completion fractions at each absolute-time cutoff were retained to document that some short episodes had already ended.

We fitted separate affine softmax classifiers for red and blue delays, each mapping 128 features to the ten labels 0–9:

\[
p_{\phi,X}(d\mid\overline{\mathbf h})=
\operatorname{softmax}(W_X\overline{\mathbf h}+\mathbf b_X)_d,
\qquad X\in\{R,B\}.
\]

The episode split, determined once with analysis seed 0, selected 16 of the 20 repetitions per profile for fitting and four for testing (736/184 episodes). Repetition indices were reused across policies, targets, and cutoffs. Each network, target, and cutoff had independently fitted parameters and optimizer state; hidden-coordinate systems from different networks were not pooled. Probes used full-batch cross-entropy, Adam at learning rate 0.01 for 1,000 updates, and fixed initialization seeds 10000 and 10001 for red and blue targets. They had no hidden layers, feature scaling, weight decay, early stopping, warm starts, or test-based checkpoint selection. Their primary distance score was

\[
S_X=\frac1{M}\sum_{e=1}^{M}\left(1-\frac{|\widehat d_{X,e}-d_{X,e}|}{9}\right),
\]

which gave partial credit for nearby delay predictions. Exact classification accuracy and mean absolute error were also reported. Baselines fitted the same probes to independent standard-normal features with the actual capability labels, using five random-feature seeds. A paired-label shuffle at the 400-tick cutoff provided a further diagnostic control. Means, sample standard deviations, and 10,000 percentile-bootstrap resamples used the five learner seeds as the uncertainty units.

The probe test split held out episodes within *every* profile, including those novel to policy training. It therefore tested recoverability in fresh episodes; it did not test decoder transfer to capability profiles excluded from probe fitting. Recoverability also did not, by itself, establish causal use of that information by the action policy. A separate UMAP visualization used each episode's final-50-state average, with 920 points per independently embedded network, `n_neighbors=919`, `min_dist=1.0`, and random state 42. UMAP served as a qualitative visualization rather than an inferential test.

**Separate reproduction of the released probe procedure.** The repository also preserved a distinct analysis reproducing the pinned upstream Overcooked probe code. It used AdamW at learning rate 0.01 and weight decay 0.001 for 1,001 updates, selected the highest test-scoring checkpoint among checks after updates 1, 21, …, 1001, and warm-started the next cutoff from those selected weights with new optimizer state. Fresh 80/20 splits were stratified by the scalar target label for each fit; an orientation head was fitted before the red and blue heads. NumPy seed 0 was an explicit reproducibility addition, and the round-index extension used a separate seed-100000 stream. The released random baseline paired normal features with random integer labels; a true-label random-feature control was saved separately. Because checkpoint selection used test scores and changing splits could expose later test episodes' labels in earlier fits, these results reproduced a fitting procedure without providing the primary analysis's independent held-out estimate. Both probe analyses used the same saved policy states and covered the original five learner seeds, not all ten behavioral seeds.

**Implementation qualifications for interpreting v1 versus v2.** The frozen v1 trainer and frozen v2 trainer shared architecture and nominal hyperparameters but differed in recurrent-reset alignment during PPO loss replay. During behavior collection and evaluation, both reset memory according to the preceding transition's full-episode termination. During optimization replay, v1 instead supplied the *current* transition's termination flag before encoding that transition's observation. This reset replay memory on the final decision of an episode and did not align the next episode's initial decision with the collection reset. V2 stored a separate pre-observation reset flag and replayed that correctly. The ten-seed extension preserved each original trainer; it did not retroactively fix v1. Comparisons therefore changed both allocation dynamics and this replay detail, and should not be described as an isolated manipulation of allocation timing. The no-influence assignment rules and the number of legal movement-phase actions also differed as specified above.

The static selection score \(1-0.01T\) charged the penalty on the successful transition, whereas the executed environment returned \(1-0.01(T-1)\). For two feasible static assignments this was a common 0.01 offset and did not change their ranking or reward difference. Saved analytical rewards should nevertheless be identified as scores from that historical convention. Likewise, static oracle completion estimates assumed no collisions and no switching; reported learned-policy rewards came from the full executed environment. Finally, the optimal static allocation on this particular corpus required partner specialization orientation, while exact delay decoding was an additional representational measurement. No claim that successful coordination necessarily required a complete numerical capability model is implied.

**Reproducibility and evidence.** Original and additional batches were `counterbalanced1096_20261002_235609` and `counterbalanced1096_20261007_001529_seeds6to10`. Their manifests retained resolved configurations, source and corpus hashes, schedule hashes, and completion records. Training froze separate v1 and v2 sources plus one common corpus and paired schedules. Each training job requested one GPU with an 80-GB resource constraint, 32 GB host memory, and four CPU cores; the extension recorded JAX 0.4.20 and jaxlib 0.4.20 with the CUDA 12/cuDNN 8.9 build. A specific GPU model should not be inferred solely from the resource constraint. Checkpoints and companion configs were saved separately, evaluation rollouts and hidden states were written to HDF5, and sampling audits recorded actual exposure. Across 80 policies, the executed budget was 4,797,235,200 environment transitions. Full training was not rerun to prepare this methods draft.

## Source map

Links below are pinned to the reviewed commit. Run sources, resolved configs, and manifests take precedence over current preparation defaults. The six inspected frozen environment, trainer, and evaluator files matched the SHA-256 hashes in the original manifest.

| Claim or procedure | Repository evidence |
| --- | --- |
| Original completed counterbalanced design, resources, hashes, schedule capacity | [Original run manifest](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/train/manifests/sbatch_counterbalanced1096_20261002_235609.json) |
| Additional seeds, frozen-source reuse, ten-seed completion | [Seed-extension manifest](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/train/manifests/sbatch_counterbalanced1096_20261007_001529_seeds6to10.json) |
| Exact v1 dynamics, observations, cooldowns, collision handling | [Frozen v1 environment](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/train/run_snapshots/counterbalanced1096_20261002_235609/v1_balanced_training/source/jaxmarl/environments/coordination_grid/coordination_grid.py) |
| Exact v2 dynamics, reassignment and legal actions | [Frozen v2 environment](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/train/run_snapshots/counterbalanced1096_20261002_235609/v2_balanced_training/source/jaxmarl/environments/coordination_grid/coordination_grid.py) |
| Architecture, PPO, v1 replay-reset behavior | [Frozen v1 trainer](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/train/run_snapshots/counterbalanced1096_20261002_235609/v1_balanced_training/source/baselines/IPPO/ippo_rnn_coordination_grid.py) |
| Architecture, PPO and v2 reset alignment | [Frozen v2 trainer](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/train/run_snapshots/counterbalanced1096_20261002_235609/v2_balanced_training/source/baselines/IPPO/ippo_rnn_coordination_grid.py) |
| Run-resolved hyperparameters | [v1 config](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/train/train_logs/v1_balanced_training/counterbalanced1096_20261002_235609/rnn_diverse_influence_seed1_config.json), [v2 config](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/train/train_logs/v2_balanced_training/counterbalanced1096_20261002_235609/rnn_diverse_influence_seed1_config.json) |
| Training and novel capability populations | [Capability definitions](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/jaxmarl/environments/coordination_grid/capability_populations.py) |
| Historical corpus filtering and final 137-per-distance quotas | [Corpus manifest](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/data_prep/grids_capability_selected_balanced_1096/manifest.json), [balancing algorithm](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/data_prep/balance_ego_distances.py) |
| Exhaustive geometry-only ceiling and faster-goal rule | [Geometry audit](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/data_prep/grids_capability_selected_balanced_1096/diagnostics/geometry_goal_dependence/summary.json) |
| Paired packets and asynchronous global queue | [Frozen scheduler](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/train/run_snapshots/counterbalanced1096_20261002_235609/v2_balanced_training/source/baselines/IPPO/counterbalanced_scheduler.py), [actual exposure audit](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/train/sampling_audits/counterbalanced1096_20261002_235609/sampling_verification.json) |
| V1 behavioral evaluation and static-request regret | [Frozen v1 evaluator](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/train/run_snapshots/counterbalanced1096_20261002_235609/v1_balanced_training/source/analysis/evaluate_partner_modelling.py) |
| V2 behavioral evaluation and post-observation hidden-state capture | [Frozen v2 evaluator](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/train/run_snapshots/counterbalanced1096_20261002_235609/v2_balanced_training/source/analysis/evaluate_partner_modelling.py) |
| Ten-seed behavioral aggregation | [Combined results and aggregation definitions](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/eval/protocol_comparison/counterbalanced1096_20261007_001529_seeds6to10/all_10_seeds/RESULTS_SUMMARY.md) |
| Primary representation analysis and fitted settings | [Primary analysis code](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/eval/representation_analysis.py), [completed v2 report](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/eval/representation_results/counterbalanced1096_20261002_235609/v2_balanced_training/README.md) |
| Separate released-code probe replication | [Strict analysis code](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/eval/representation_analysis_strict.py), [completed report](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/eval/representation_results/counterbalanced1096_20261002_235609_strict_replication_of_covercooked/v2_balanced_training/README.md) |
| Implementation disclosures and historical/default distinctions | [Archived implementation audit guide](https://github.com/selenashe/emergent_partner_grid/blob/e0878e5be345180711e73fc2e5381e249ecac348/archive/IMPLEMENTATION_GUIDE.md) |

The separately prepared 2,262-layout corpus was not used for these runs. Selected-seed plots, stricter released-code probes, and later partner-dynamics analyses do not change the training task, completed policy weights, or the ten-seed behavioral sampling procedure. The broader partner-dynamics package is a separate analysis suite and is outside this account of the counterbalanced runs and capability-decoding methods.
