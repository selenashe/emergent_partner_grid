# CoordinationGrid representation analysis

## Data and validation

All 15 recurrent checkpoints (three conditions × policy seeds 1–5) were analyzed. Existing train/test HDF5 files were reused; no evaluation or training was rerun. Every checkpoint passed 920 episodes, 46 authoritative profiles (24 train + 22 novel), 20 repetitions/profile, finite 128-D hidden scans, constant capability on valid steps, a final done, and all 20 ordered round ends. MLP checkpoints are excluded.

`episode_manifest.csv` records every input file, source episode, profile, repetition, valid length and probe partition. `validation.json` and `analysis_metadata.json` record the input sizes/timestamps, checks, software versions and analysis source hash.

| Condition | Valid length min–max across seeds | Mean length across seeds |
| --- | --- | --- |
| Multi-partner RNN | 200–1455 | 593.4 |
| Single-partner RNN | 171–1392 | 661.4 |
| No-influence RNN | 408–1580 | 796.4 |

By the 400-step cutoff, the mean fraction already complete across the five policies is Multi-partner RNN: 24.02%; Single-partner RNN: 20.91%; No-influence RNN: 0.00%. Earlier completion fractions are retained in the CSVs.

## Probe protocol

PCG64 analysis seed 0 permutes repetitions 0–19 once. Probe training repetitions: `[4, 19, 6, 2, 13, 16, 3, 11, 10, 8, 0, 12, 7, 5, 18, 17]`; test repetitions: `[14, 9, 1, 15]`. Each capability pair has 16 train and 4 held-out rollouts (736/184 total). A repetition is its zero-based occurrence within its capability profile in the stored episode order, checked against `capability_index_per_ep`. The split is identical for every condition, policy seed, target and cutoff; episode ordering is canonicalized by `(d_R,d_B,rollout_rep)` before fitting.

Separate 10-class affine probes decode d_R and d_B directly as ordered classes 0–9. Each receives only 128 averaged GRU coordinates. Full-batch softmax cross entropy, Adam lr=0.01, 1000 updates, bias, no hidden layers, no scaling, no weight decay or tuning. Fixed initialization seeds are `{'d_R': 10000, 'd_B': 10001}`. Every policy has its own probes; coordinate systems are never pooled. Independent probes are batched computationally, with separate parameters and Adam moments.

The primary score is `mean(1 − |prediction − true delay|/9)`. Exact accuracy and delay MAE are also saved. A high distance score is not itself evidence of decoding: central-class guesses already score well, so compare with the trained Normal baseline.

At t=1,50,100,…,400, inputs are `mean(hidden[:min(t,L)])`, where L includes the first terminal step. Round curves average from episode start through each round's terminal transition, using its post-observation hidden state. No scan padding is used. Completion fractions at each t are saved in the timestep tables.

All-seed summaries use sample SD and 10,000 percentile bootstrap resamples of the five independent policy seeds (RNG seed 30000). These intervals describe policy-seed variability; five seeds provide limited precision. No rollout-level CI is substituted for model uncertainty.

## Behavior-selected checkpoints

Selection uses the final PPO-update `ep_return_mean` from completed stdout logs that saved each checkpoint (rounded to four decimals). If this metric is unavailable for any seed of a condition, all five candidates in that condition use `train.mean_ep_return` as an explicitly labeled evaluation proxy. Representation scores and novel-test returns never enter selection.

| Condition | Selected seed | Metric | Value |
| --- | --- | --- | --- |
| Multi-partner RNN | 5 | final_training.ep_return_mean (last PPO update; stdout rounded to 4 decimals) | 15.8960 |
| Single-partner RNN | 5 | final_training.ep_return_mean (last PPO update; stdout rounded to 4 decimals) | 17.4740 |
| No-influence RNN | 3 | final_training.ep_return_mean (last PPO update; stdout rounded to 4 decimals) | 13.3662 |

All candidate values and exact source logs are in `selected_seeds.json`. The selected-seed view emphasizes paper fidelity; the all-five-seed view is the main robustness analysis and retains every policy, including weak policies.

## Descriptive probe results

| Condition | Target | t=1 mean | t=400 mean [95% CI] | Seed SD at t=400 | Selected t=400 | Round 0 → 19 mean |
| --- | --- | --- | --- | --- | --- | --- |
| Multi-partner RNN | d_R | 0.637 | 0.838 [0.766, 0.900] | 0.084 | 0.896 | 0.743 → 0.861 |
| Multi-partner RNN | d_B | 0.650 | 0.840 [0.760, 0.907] | 0.095 | 0.922 | 0.748 → 0.861 |
| Single-partner RNN | d_R | 0.637 | 0.914 [0.862, 0.960] | 0.063 | 0.967 | 0.816 → 0.943 |
| Single-partner RNN | d_B | 0.641 | 0.835 [0.816, 0.853] | 0.025 | 0.851 | 0.777 → 0.838 |
| No-influence RNN | d_R | 0.632 | 0.678 [0.665, 0.693] | 0.018 | 0.705 | 0.736 → 0.704 |
| No-influence RNN | d_B | 0.643 | 0.689 [0.654, 0.720] | 0.041 | 0.710 | 0.786 → 0.701 |

Per-policy endpoints (no seed omitted):

| Condition | Target | Seeds 1, 2, 3, 4, 5 at t=400 | Above Normal mean (of 5) |
| --- | --- | --- | --- |
| Multi-partner RNN | d_R | 0.780, 0.886, 0.910, 0.719, 0.896 | 5 |
| Multi-partner RNN | d_B | 0.793, 0.890, 0.901, 0.695, 0.922 | 5 |
| Single-partner RNN | d_R | 0.957, 0.953, 0.831, 0.860, 0.967 | 5 |
| Single-partner RNN | d_B | 0.850, 0.857, 0.809, 0.807, 0.851 | 5 |
| No-influence RNN | d_R | 0.670, 0.678, 0.705, 0.657, 0.682 | 5 |
| No-influence RNN | d_B | 0.633, 0.737, 0.710, 0.661, 0.707 | 4 |

Progression is measured as endpoint differences, without assuming the curve is monotonic:

| Condition | Target | Mean t=1→400 change | Policies increasing (of 5) | Mean round 0→19 change | Policies increasing (of 5) |
| --- | --- | --- | --- | --- | --- |
| Multi-partner RNN | d_R | +0.201 | 5 | +0.118 | 5 |
| Multi-partner RNN | d_B | +0.190 | 5 | +0.113 | 5 |
| Single-partner RNN | d_R | +0.276 | 5 | +0.127 | 5 |
| Single-partner RNN | d_B | +0.193 | 5 | +0.062 | 5 |
| No-influence RNN | d_R | +0.046 | 5 | -0.032 | 1 |
| No-influence RNN | d_B | +0.047 | 4 | -0.085 | 0 |

Multi-partner minus control contrasts pair the nominal model-training seed and bootstrap those five seed differences. They are descriptive comparisons of the trained policies, with limited uncertainty resolution:

| Target | Control | t=400 difference [95% CI] | Round 19 difference [95% CI] |
| --- | --- | --- | --- |
| d_R | Single-partner RNN | -0.076 [-0.148, +0.006] | -0.082 [-0.129, -0.026] |
| d_R | No-influence RNN | +0.160 [+0.101, +0.210] | +0.156 [+0.125, +0.185] |
| d_B | Single-partner RNN | +0.006 [-0.064, +0.072] | +0.022 [-0.030, +0.073] |
| d_B | No-influence RNN | +0.151 [+0.091, +0.195] | +0.160 [+0.119, +0.195] |

## Interpretation

At t=400, the largest all-policy mean distance score is d_R: Single-partner RNN (0.914); d_B: Multi-partner RNN (0.840).

At round 19, the largest all-policy mean distance score is d_R: Single-partner RNN (0.943); d_B: Multi-partner RNN (0.861).

The predicted pattern of the multi-partner condition outperforming both controls on both cooldowns is not observed at t=400. Some control/target comparisons differ from that expectation; the table above preserves their direction and uncertainty.

Evidence for capability information comes from held-out linear decoding above the Normal baseline and its development with history. Condition differences must be assessed for each cooldown and across all five policies; a behavior-selected policy or a UMAP panel alone cannot establish robustness. Decoding demonstrates recoverable information in recurrent state, but does not by itself establish that the action policy uses that information causally. The condition contrasts test selective strength under the current experiment, without modifying the task or excluding poorly trained policies.

Secondary familiar/novel decoding uses the same probe trained on all 46 profiles, then partitions the 184 held-out repetitions (96 familiar / 88 novel). This is novel to policy training, **not** a probe trained without novel labels; it is not a held-out-capability transfer test. For the single-partner policy, the 24-profile 'familiar' slice means the experiment's TRAIN population; only its single training partner was actually familiar to that policy.

| Condition | Target | Familiar t=400 mean | Novel t=400 mean |
| --- | --- | --- | --- |
| Multi-partner RNN | d_R | 0.830 | 0.847 |
| Multi-partner RNN | d_B | 0.848 | 0.832 |
| Single-partner RNN | d_R | 0.915 | 0.912 |
| Single-partner RNN | d_B | 0.832 | 0.838 |
| No-influence RNN | d_R | 0.667 | 0.691 |
| No-influence RNN | d_B | 0.679 | 0.701 |

## Baselines and leakage screen

Normal(0,1) features have shape (920,128), use real labels and the same split, and are fitted with the identical linear probe. Five fixed vector seeds (0–4) are saved. The selected-policy figure uses seed 0; robustness/round figures show the five-vector-seed mean as a time-independent dashed reference. This baseline is not chance exact accuracy.

| Target | Normal seed 0 | Normal mean ± SD | Shuffle t=400 across 15 policies mean ± SD |
| --- | --- | --- | --- |
| d_R | 0.620 | 0.621 ± 0.011 | 0.648 ± 0.018 |
| d_B | 0.644 | 0.637 ± 0.018 | 0.638 ± 0.018 |

The descriptive leakage screen passed 30/30 primary shuffled-target fits. Its predefined rule is absolute distance-score difference from the Normal mean ≤0.10; this broad screen is not a statistical equivalence claim. A fixed permutation of paired capability labels breaks their relation to the t=400 real representation. All per-fit results, including secondary subset screens, are saved.

## UMAP

Each behavior-selected network is embedded separately using 920 final-50 valid-step averages, min_dist=1.0, n_neighbors=919, random_state=42, otherwise UMAP defaults. Library versions and parameters are saved. All panels use one color scale for `relative_red_speed_advantage = d_B − d_R`: positive means RED faster, negative means BLUE faster. `delay_diff = d_R − d_B` is also saved. Companion figures show individual d_R/d_B cooldowns. Layout identity is never a grouping unit.

Visual inspection of the saved panels: The behavior-selected multi-partner and single-partner panels both show substantial separation of warm (RED faster) and cool (BLUE faster) regions, with some blending. The no-influence panel has broadly intermingled colors. Thus this qualitative view supports an influence-related contrast but does not distinguish diversity as necessary for relative-capability structure. These observations concern only the three selected networks and do not quantify representation strength.

UMAP is qualitative visualization only: separation does not measure representation strength or establish causal use of decoded information. Inspect the saved figures together with held-out linear probes.

## Protocol correspondence and deviations

Reference: [Mon-Williams et al., Appendix A.2–A.3](https://arxiv.org/html/2505.17323v1#A2.SS3). The requested grid protocol follows the original Figure-3 analysis; it does not alter the grid experiment to reproduce the original findings.

| Original representation protocol | CoordinationGrid |
| --- | --- |
| Task-1 / task-2 cooldown | d_R / d_B; lower delay means faster |
| 46 evaluated profiles | 24 train + 22 novel = 46 |
| 20 rollout seeds/profile, 920 rollouts | Same counts; repetition recovered from stored episode order |
| RNN hidden state | Post-observation GRU state, dimension 128 |
| Final-50 average, episode prefix average | Same, restricted to valid terminal-inclusive steps |
| Single-layer ordered classification | Two affine 128→10 probes |
| 80/20 rollout-seed split | Same: 16/4 per profile, fixed shared permutation |
| 1000 updates, Adam lr=1e-2 | Same; full-batch implementation, fixed initialization |
| Distance-aware accuracy | Same: 1 − absolute class error/9 |
| Random Normal representation | Same; seed 0 plus four stability repeats |
| UMAP min_dist=1, n_neighbors=N−1 | Same; fixed random_state=42 |
| Best final-training-return policy among five | Same when recoverable; documented eval-return fallback otherwise |
| Fixed 400-step episode | Variable-length 20-round partner episode; clip cutoffs to valid length and report completion fractions |
| Five fixed kitchen layouts | Layouts vary each round from the fixed 1000-layout corpus; analyze whole partner episodes |

The round-index curve, five-policy-seed confidence intervals, familiar/novel subsets and shuffled-label screen are additional diagnostics. The repeated-round analysis extends beyond the t=400 curve for long grid episodes.

## Reproduction and artifacts

```bash
python -m pip install -r requirements/representation.txt
python eval/representation_analysis.py --eval-dir eval/eval_out \
  --out-dir eval/representation_results --analysis-seed 0
```

Primary CSVs: `probe_timestep_per_seed.csv`, `probe_timestep_summary.csv`, `probe_by_round_per_seed.csv`, `probe_by_round_summary.csv`, `random_baseline.csv`. Additional CSVs: `probe_condition_differences.csv`, `shuffled_label_baseline.csv`, `episode_manifest.csv`, `umap_selected.csv`, and the round-table alias `probe_by_round.csv`. Figures are saved in PNG and PDF; fitted affine parameters and selected final-50 feature matrices are saved in NPZ for inspection.
