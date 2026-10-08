# CoordinationGrid representation analysis

Input evaluation directory: `eval/eval_out/v2_balanced_training/counterbalanced1096_20261002_235609`. Allocation protocol: `online_v2`. Training corpus: 1,096 layouts.

## Data and validation

All 15 recurrent checkpoints (three conditions × policy seeds 1–5) were analyzed. Existing train/test HDF5 files were reused; no evaluation or training was rerun. Every checkpoint passed 920 episodes, 46 authoritative profiles (24 train + 22 novel), 20 repetitions/profile, finite 128-D hidden scans, constant capability on valid steps, a final done, and all 20 ordered round ends. MLP checkpoints are excluded.

`episode_manifest.csv` records every input file, source episode, profile, repetition, valid length and probe partition. `validation.json` and `analysis_metadata.json` record the input sizes/timestamps, checks, software versions and analysis source hash.

| Condition | Valid length min–max across seeds | Mean length across seeds |
| --- | --- | --- |
| Multi-partner RNN | 174–1775 | 700.2 |
| Single-partner RNN | 161–1714 | 775.9 |
| No-influence RNN | 389–1727 | 879.0 |

By the 400-step cutoff, the mean fraction already complete across the five policies is Multi-partner RNN: 19.96%; Single-partner RNN: 14.93%; No-influence RNN: 0.02%. Earlier completion fractions are retained in the CSVs.

## Probe protocol

Probe fitting includes all 46 capability profiles; the test split holds out episodes within each profile. Familiar/novel subsets indicate whether capabilities were seen during policy training, not whether they were seen during probe fitting.

PCG64 analysis seed 0 permutes repetitions 0–19 once. Probe training repetitions: `[4, 19, 6, 2, 13, 16, 3, 11, 10, 8, 0, 12, 7, 5, 18, 17]`; test repetitions: `[14, 9, 1, 15]`. Each capability pair has 16 train and 4 held-out rollouts (736/184 total). A repetition is its zero-based occurrence within its capability profile in the stored episode order, checked against `capability_index_per_ep`. The split is identical for every condition, policy seed, target and cutoff; episode ordering is canonicalized by `(d_R,d_B,rollout_rep)` before fitting.

Separate 10-class affine probes decode d_R and d_B directly as ordered classes 0–9. Each receives only 128 averaged GRU coordinates. Full-batch softmax cross entropy, Adam lr=0.01, 1000 updates, bias, no hidden layers, no scaling, no weight decay or tuning. Fixed initialization seeds are `{'d_R': 10000, 'd_B': 10001}`. Every policy has its own probes; coordinate systems are never pooled. Independent probes are batched computationally, with separate parameters and Adam moments.

The primary score is `mean(1 − |prediction − true delay|/9)`. Exact accuracy and delay MAE are also saved. A high distance score is not itself evidence of decoding: central-class guesses already score well, so compare with the trained Normal baseline.

At t=1,50,100,…,400, inputs are `mean(hidden[:min(t,L)])`, where L includes the first terminal step. Round curves average from episode start through each round's terminal transition, using its post-observation hidden state. No scan padding is used. Completion fractions at each t are saved in the timestep tables.

All-seed summaries use sample SD and 10,000 percentile bootstrap resamples of the five independent policy seeds (RNG seed 30000). These intervals describe policy-seed variability; five seeds provide limited precision. No rollout-level CI is substituted for model uncertainty.

## Behavior-selected checkpoints

The additional selected-policy view reuses the existing common training seed across conditions and versions. Its source and hash are recorded in `selected_seeds.json`. All five seeds contribute to the primary analysis; no policy is selected using decoding scores.

| Condition | Selected seed | Metric | Value |
| --- | --- | --- | --- |
| Multi-partner RNN | 5 | final_training.ep_return_mean (common seed selected within counterbalanced batch) | 12.9980 |
| Single-partner RNN | 5 | final_training.ep_return_mean (common seed selected within counterbalanced batch) | 17.4190 |
| No-influence RNN | 5 | final_training.ep_return_mean (common seed selected within counterbalanced batch) | 10.6260 |

All candidate values and exact source logs are in `selected_seeds.json`. The selected-seed view is supplementary; the all-five-seed view is the main robustness analysis and retains every policy, including weak policies.

## Descriptive probe results

| Condition | Target | t=1 mean | t=400 mean [95% CI] | Seed SD at t=400 | Selected t=400 | Round 0 → 19 mean |
| --- | --- | --- | --- | --- | --- | --- |
| Multi-partner RNN | d_R | 0.643 | 0.870 [0.820, 0.920] | 0.065 | 0.811 | 0.819 → 0.884 |
| Multi-partner RNN | d_B | 0.627 | 0.886 [0.839, 0.933] | 0.059 | 0.939 | 0.820 → 0.899 |
| Single-partner RNN | d_R | 0.636 | 0.871 [0.815, 0.928] | 0.073 | 0.946 | 0.795 → 0.898 |
| Single-partner RNN | d_B | 0.642 | 0.811 [0.769, 0.854] | 0.057 | 0.835 | 0.753 → 0.822 |
| No-influence RNN | d_R | 0.634 | 0.680 [0.661, 0.697] | 0.024 | 0.686 | 0.635 → 0.696 |
| No-influence RNN | d_B | 0.645 | 0.667 [0.654, 0.680] | 0.016 | 0.666 | 0.631 → 0.683 |

Per-policy endpoints (no seed omitted):

| Condition | Target | Seeds 1, 2, 3, 4, 5 at t=400 | Above Normal mean (of 5) |
| --- | --- | --- | --- |
| Multi-partner RNN | d_R | 0.941, 0.841, 0.939, 0.819, 0.811 | 5 |
| Multi-partner RNN | d_B | 0.816, 0.949, 0.838, 0.888, 0.939 | 5 |
| Single-partner RNN | d_R | 0.949, 0.790, 0.823, 0.850, 0.946 | 5 |
| Single-partner RNN | d_B | 0.825, 0.888, 0.756, 0.752, 0.835 | 5 |
| No-influence RNN | d_R | 0.696, 0.644, 0.668, 0.704, 0.686 | 5 |
| No-influence RNN | d_B | 0.690, 0.666, 0.644, 0.669, 0.666 | 5 |

Progression is measured as endpoint differences, without assuming the curve is monotonic:

| Condition | Target | Mean t=1→400 change | Policies increasing (of 5) | Mean round 0→19 change | Policies increasing (of 5) |
| --- | --- | --- | --- | --- | --- |
| Multi-partner RNN | d_R | +0.227 | 5 | +0.064 | 5 |
| Multi-partner RNN | d_B | +0.259 | 5 | +0.079 | 5 |
| Single-partner RNN | d_R | +0.235 | 5 | +0.104 | 5 |
| Single-partner RNN | d_B | +0.169 | 5 | +0.070 | 5 |
| No-influence RNN | d_R | +0.045 | 5 | +0.060 | 5 |
| No-influence RNN | d_B | +0.022 | 4 | +0.052 | 5 |

Multi-partner minus control contrasts pair the nominal model-training seed and bootstrap those five seed differences. They are descriptive comparisons of the trained policies, with limited uncertainty resolution:

| Target | Control | t=400 difference [95% CI] | Round 19 difference [95% CI] |
| --- | --- | --- | --- |
| d_R | Single-partner RNN | -0.001 [-0.073, +0.074] | -0.014 [-0.078, +0.053] |
| d_R | No-influence RNN | +0.190 [+0.136, +0.245] | +0.188 [+0.126, +0.252] |
| d_B | Single-partner RNN | +0.075 [+0.028, +0.114] | +0.077 [+0.035, +0.111] |
| d_B | No-influence RNN | +0.219 [+0.169, +0.266] | +0.216 [+0.163, +0.272] |

## Interpretation

At t=400, the largest all-policy mean distance score is d_R: Single-partner RNN (0.871); d_B: Multi-partner RNN (0.886).

At round 19, the largest all-policy mean distance score is d_R: Single-partner RNN (0.898); d_B: Multi-partner RNN (0.899).

The predicted pattern of the multi-partner condition outperforming both controls on both cooldowns is not observed at t=400. Some control/target comparisons differ from that expectation; the table above preserves their direction and uncertainty.

Evidence for capability information comes from held-out linear decoding above the Normal baseline and its development with history. Condition differences must be assessed for each cooldown and across all five policies; a behavior-selected policy or a UMAP panel alone cannot establish robustness. Decoding demonstrates recoverable information in recurrent state, but does not by itself establish that the action policy uses that information causally. The condition contrasts test selective strength under the current experiment, without modifying the task or excluding poorly trained policies.

Secondary familiar/novel decoding uses the same probe trained on all 46 profiles, then partitions the 184 held-out repetitions (96 familiar / 88 novel). This is novel to policy training, **not** a probe trained without novel labels; it is not a held-out-capability transfer test. For the single-partner policy, the 24-profile 'familiar' slice means the experiment's TRAIN population; only its single training partner was actually familiar to that policy.

| Condition | Target | Familiar t=400 mean | Novel t=400 mean |
| --- | --- | --- | --- |
| Multi-partner RNN | d_R | 0.859 | 0.882 |
| Multi-partner RNN | d_B | 0.878 | 0.895 |
| Single-partner RNN | d_R | 0.871 | 0.872 |
| Single-partner RNN | d_B | 0.808 | 0.815 |
| No-influence RNN | d_R | 0.687 | 0.672 |
| No-influence RNN | d_B | 0.659 | 0.675 |

## Baselines and leakage screen

Normal(0,1) features have shape (920,128), use real labels and the same split, and are fitted with the identical linear probe. Five fixed vector seeds (0–4) are saved. The selected-policy figure uses seed 0; robustness/round figures show the five-vector-seed mean as a time-independent dashed reference. This baseline is not chance exact accuracy.

| Target | Normal seed 0 | Normal mean ± SD | Shuffle t=400 across 15 policies mean ± SD |
| --- | --- | --- | --- |
| d_R | 0.620 | 0.621 ± 0.011 | 0.632 ± 0.019 |
| d_B | 0.644 | 0.637 ± 0.018 | 0.631 ± 0.014 |

The descriptive leakage screen passed 30/30 primary shuffled-target fits. Its predefined rule is absolute distance-score difference from the Normal mean ≤0.10; this broad screen is not a statistical equivalence claim. A fixed permutation of paired capability labels breaks their relation to the t=400 real representation. All per-fit results, including secondary subset screens, are saved.

## UMAP

Each behavior-selected network is embedded separately using 920 final-50 valid-step averages, min_dist=1.0, n_neighbors=919, random_state=42, otherwise UMAP defaults. Library versions and parameters are saved. All panels use one color scale for `relative_red_speed_advantage = d_B − d_R`: positive means RED faster, negative means BLUE faster. `delay_diff = d_R − d_B` is also saved. Companion figures show individual d_R/d_B cooldowns. Layout identity is never a grouping unit.

UMAP is qualitative visualization only: separation does not measure representation strength or establish causal use of decoded information. Inspect the saved figures together with held-out linear probes.

## Protocol correspondence and deviations

Reference: [Mon-Williams et al., Appendix B.3](https://arxiv.org/html/2505.17323v2). The requested grid protocol follows the original Figure-3 analysis; it does not alter the grid experiment to reproduce the original findings.

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
| Best final-training-return policy among five | Existing common seed reused across conditions/versions, per requested comparison |
| Fixed 400-step episode | Variable-length 20-round partner episode; clip cutoffs to valid length and report completion fractions |
| Five fixed kitchen layouts | Layouts vary each round from the fixed 1,096-layout corpus; analyze whole partner episodes |

The round-index curve, five-policy-seed confidence intervals, familiar/novel subsets and shuffled-label screen are additional diagnostics. The repeated-round analysis extends beyond the t=400 curve for long grid episodes.

## Reproduction and artifacts

```bash
python -m pip install -r requirements/representation.txt
python eval/representation_analysis.py --eval-dir eval/eval_out/v2_balanced_training/counterbalanced1096_20261002_235609 \
  --out-dir eval/representation_results/counterbalanced1096_20261002_235609/v2_balanced_training --analysis-seed 0 \
  --device cpu --threads 1 --layout-count 1096 --common-seed-record eval/protocol_comparison/counterbalanced1096_20261002_235609/common_training_seed/selection.json --umap-all-seeds
```

Primary CSVs: `probe_timestep_per_seed.csv`, `probe_timestep_summary.csv`, `probe_by_round_per_seed.csv`, `probe_by_round_summary.csv`, `random_baseline.csv`. Additional CSVs: `probe_condition_differences.csv`, `shuffled_label_baseline.csv`, `episode_manifest.csv`, `umap_selected.csv`, and the round-table alias `probe_by_round.csv`. Figures are saved in PNG and PDF; fitted affine parameters and selected final-50 feature matrices are saved in NPZ for inspection.

UMAP was fitted separately to all 15 networks. `umap_all_seeds.csv` contains 13,800 episode embeddings; `umap_by_seed/seed1` through `seed5` contain three-condition panels colored by relative speed, red delay and blue delay. Coordinates from different networks are not directly comparable.
