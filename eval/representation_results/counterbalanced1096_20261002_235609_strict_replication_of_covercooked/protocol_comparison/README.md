# Probe-protocol comparison on the same frozen RNNs

Source HDF5 file provenance is identical between protocols for all 30 RNN checkpoints. Solid curves use the preceding written-methods implementation; dashed curves use the pinned released-code routine. Colors identify v1/v2, and shading bootstraps all five policy seeds.

The scores use the same normalized absolute-error metric, but test partitions and fitting procedures differ. Score changes are changes to measurement on fixed models, not changes to policy training or behavior. Strict probes select by test score and can carry training information into later resampled test partitions, as the per-version reports document.

| Version | Condition | Target | Previous, round 20 | Strict, round 20 | Change |
| --- | --- | --- | --- | --- | --- |
| v1 | Multi-partner RNN | d_R | 0.904 | 0.940 | +0.037 |
| v2 | Multi-partner RNN | d_R | 0.884 | 0.931 | +0.047 |
| v1 | Multi-partner RNN | d_B | 0.893 | 0.925 | +0.031 |
| v2 | Multi-partner RNN | d_B | 0.899 | 0.934 | +0.035 |
| v1 | Single-partner RNN | d_R | 0.973 | 0.994 | +0.021 |
| v2 | Single-partner RNN | d_R | 0.898 | 0.939 | +0.041 |
| v1 | Single-partner RNN | d_B | 0.829 | 0.863 | +0.034 |
| v2 | Single-partner RNN | d_B | 0.822 | 0.874 | +0.051 |
| v1 | No-influence RNN | d_R | 0.712 | 0.780 | +0.069 |
| v2 | No-influence RNN | d_R | 0.696 | 0.755 | +0.059 |
| v1 | No-influence RNN | d_B | 0.738 | 0.796 | +0.058 |
| v2 | No-influence RNN | d_B | 0.683 | 0.778 | +0.094 |

`endpoint_protocol_differences.csv` also reports exact-class accuracy and MAE, with descriptive paired-policy-seed intervals. The baseline differs between implementations and is omitted from this sensitivity figure; the primary per-protocol figures display their respective baselines.

```bash
python eval/compare_representation_protocols.py --previous-root eval/representation_results/counterbalanced1096_20261002_235609 \
  --strict-root eval/representation_results/counterbalanced1096_20261002_235609_strict_replication_of_covercooked
```
