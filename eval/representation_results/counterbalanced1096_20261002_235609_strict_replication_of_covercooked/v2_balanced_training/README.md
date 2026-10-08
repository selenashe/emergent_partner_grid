# Strict replication of the released Overcooked probe routine

Inputs: `eval/eval_out/v2_balanced_training/counterbalanced1096_20261002_235609`; allocation protocol `online_v2`.

[Pinned upstream source](https://github.com/ruaridhmon/emergent_partner_modelling/blob/35a8430046531bd4b62e923d7507ee9f02fd97ea/analysis/do_ablation_plots.py) is loaded directly for the affine layer, AdamW updates, label-stratified splitting, shuffled full batches, and best-test-checkpoint selection. All three RNN conditions and all five policy seeds are included (15 networks).

Every fit runs 1,001 updates at lr=0.01 and weight decay=0.001. Test scores are checked after updates 1,21,...,1001; the first highest-scoring checkpoint is retained. The selected weights warm-start the next cutoff, with fresh optimizer state. An orientation head is fitted first, matching the source call order, followed by red and blue heads.

Splits are drawn afresh for each fit, stratified by the scalar target label. NumPy MT19937 seed 0 is an explicit reproducibility addition; upstream does not set that seed. Primary time probes and the released baseline follow the source loop order. The round-index extension uses its own NumPy stream (seed 100000), starts new warm-start chains, and does not change the primary analysis.

The released baseline fits random Normal features to random integer labels; it is reproduced as the dashed line. `true_label_random_baseline.csv` separately reports a supplemental random-feature control using actual capability labels. These are distinct controls.

The metric remains `mean(1 - abs(prediction - label)/9)` for red and blue. The primary curves average all five policy seeds; bootstrap intervals resample those five policies. Supplementary selected-policy plots retain the previously chosen common seed 5.

## Interpretation limits

The source selects checkpoints using the reported test scores and changes the test split at later cutoffs while retaining weights trained on earlier splits. Later test episodes can therefore have supplied training labels to earlier probes. `n_test_previously_used_for_training` records that overlap. These scores reproduce the released procedure and should not be described as a fully independent hold-out estimate. Probe fitting includes all 46 capability profiles, even profiles novel to policy training.

The grid trajectories and 20-round episodes differ from the Overcooked environment. The strict label applies to the released fitting procedure, not to recreating its task or reported scores. UMAPs depend on unchanged hidden states and remain available in the preceding analysis; they are not regenerated here.

## Mean endpoint scores

| Condition | Target | t=400 | Round 20 |
| --- | --- | --- | --- |
| Multi-partner RNN | d_R | 0.903 | 0.931 |
| Multi-partner RNN | d_B | 0.914 | 0.934 |
| Single-partner RNN | d_R | 0.906 | 0.939 |
| Single-partner RNN | d_B | 0.846 | 0.874 |
| No-influence RNN | d_R | 0.751 | 0.755 |
| No-influence RNN | d_B | 0.745 | 0.778 |

All actual train/test masks and canonical predictions are in `probe_splits/`; the episode manifest maps those indices to the source HDF5 files. Per-fit test-score traces and selected update numbers are in `checkpoint_traces/`. Fitted parameters are in `probe_models/`. CSVs preserve all five policies and exact-class accuracy.

```bash
JAX_PLATFORMS=cpu python eval/representation_analysis_strict.py --eval-dir eval/eval_out/v2_balanced_training/counterbalanced1096_20261002_235609 \
  --out-dir eval/representation_results/counterbalanced1096_20261002_235609_strict_replication_of_covercooked/v2_balanced_training --analysis-seed 0
```
