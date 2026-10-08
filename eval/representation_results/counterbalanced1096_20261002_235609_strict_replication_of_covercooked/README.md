# Strict replication of the released Overcooked probing procedure

This separately labeled analysis uses the same saved evaluation hidden states as the preceding counterbalanced v1/v2 analysis. It does not retrain or change any policy. Both versions include the three RNN conditions and all five training seeds: 30 frozen networks total. The preceding results remain in `../counterbalanced1096_20261002_235609/`.

The affine probe, optimizer, splitting routine, update loop and checkpoint selection are loaded unchanged from the [pinned released source](https://github.com/ruaridhmon/emergent_partner_modelling/blob/35a8430046531bd4b62e923d7507ee9f02fd97ea/analysis/do_ablation_plots.py). The recipe uses AdamW (learning rate 0.01, weight decay 0.001), 1,001 updates, checks every 20 updates starting after update 1, and the first checkpoint with the highest test score. Selected parameters warm-start the next cutoff; optimizer state resets each time. Each fit draws a new scalar-label-stratified 80/20 split. The source's orientation-head call order and random-feature/random-label baseline are also retained.

NumPy seed 0 is an explicit reproducibility addition: the source leaves its NumPy RNG unseeded. The grid-specific 20-round extension uses its own split stream (seed 100000). The strict label applies to probe fitting, not to replicating the Overcooked environment or its reported numerical results. All-five-policy aggregation is retained, with supplementary plots using the previously selected common training seed 5.

## Results

- [v1 report and plots](v1_balanced_training/README.md)
- [v2 report and plots](v2_balanced_training/README.md)
- [v1 versus v2, including every seed](comparison/README.md)
- [Previous versus strict probes on identical frozen networks](protocol_comparison/README.md)
- [Completion and preservation audit](completion_audit.json)

The y-axis remains distance-aware accuracy: `mean(1 - abs(predicted_delay - true_delay)/9)`. It gives partial credit to nearby predictions and is distinct from exact-class accuracy, which is also saved in the CSVs. Probe fitting uses all 46 evaluation capability profiles, including profiles novel to policy training.

## Interpretation

The released procedure selects checkpoints using the reported test score and changes test membership while carrying forward trained parameters. At timestep 400, all 184 test episodes for every red/blue probe had already supplied training labels to an earlier probe in that chain. This is overlap within the probing analysis; it does not alter policy training. Consequently these scores reproduce the released procedure but are not fully independent hold-out estimates. The comparisons document changes to measurement on fixed models.

Actual train/test masks, predictions, fitted parameters, score traces, selected updates and episode identities are retained in each version's output directory. Frozen-state UMAPs are unaffected by the probe-fitting recipe and remain available with the preceding results.

## Reproduce

Install `requirements/representation_strict.txt`, then run into a fresh separately labeled directory (completed results are protected against overwrite):

```bash
for version in v1_balanced_training v2_balanced_training; do
  JAX_PLATFORMS=cpu python eval/representation_analysis_strict.py \
    --eval-dir eval/eval_out/$version/counterbalanced1096_20261002_235609 \
    --out-dir eval/representation_results/counterbalanced1096_20261002_235609_strict_replication_of_covercooked/$version
done
python eval/compare_representation_versions.py \
  --results-root eval/representation_results/counterbalanced1096_20261002_235609_strict_replication_of_covercooked
python eval/compare_representation_protocols.py
```

`strict_runner_source.txt` snapshots the executed runner; `upstream_provenance.json` pins the reference source. `previous_results_sha256.json` records the previous result fingerprints before this analysis; `analysis_environment_versions.json` records its runtime packages. Focused regression tests compare the instrumented routine against the unchanged source, including exact fitted weights, predictions, NumPy RNG state, warm starts and checkpoint selection.
