# Counterbalanced representation analysis: all training seeds

Completed on October 5, 2026, for the non-greedy counterbalanced batch
`counterbalanced1096_20261002_235609`. Both versions use the 1,096-layout corpus.
All three recurrent conditions and all five training seeds were analyzed:
30 networks, each with 920 evaluation episodes across 46 capability profiles.
Saved evaluation hidden states were reused; no policy training or evaluations
were repeated.

- [v1 methods, validation and numerical results](v1_balanced_training/README.md)
- [v2 methods, validation and numerical results](v2_balanced_training/README.md)
- [Version comparison and every-seed curves](comparison/README.md)
- [Completion audit](completion_audit.json)

Each version directory contains per-seed and aggregate decoding CSVs, fitted
affine probes, random-state and shuffled-label controls, and separate UMAPs for
every network under `umap_by_seed/seed1` through `seed5`. The primary figures
include all five seeds. Supplementary selected-policy figures reuse the
previously chosen common seed 5; representation scores do not affect selection.

All 30 checkpoints passed trajectory validation, and all 60 primary
shuffled-label screens passed. The regression suite passed 15 tests. All
networks and versions use the same probe train/test split: 16 fitting and four
held-out episodes per capability profile. Probe fitting includes all 46
profiles; held-out probe episodes do not imply held-out probe capability classes.

The plotted metric is `1 - mean(abs(predicted_delay - actual_delay))/9`.
Exact-class accuracy and mean absolute delay error are also saved. Confidence
intervals bootstrap the five independent policy seeds. Decoding indicates
recoverable capability information; it does not establish causal use by the actor.

The active scripts are `eval/representation_analysis.py` and
`eval/compare_representation_versions.py`. Reproduction commands appear in the
version reports and the repository README. `representation_analysis_source.txt`
preserves the exact source used for fitting; its hash matches both versions'
`entrypoint_sha256`. After fitting, the report generator's paper appendix label
was corrected to B.3; `report_entrypoint_sha256` records that reporting revision.
No fitted probes or metric tables changed in that revision. Full package
versions and input file provenance are recorded in each version's metadata.

The main endpoint pattern is stronger decoding in influence conditions than in
the no-influence control. The multi-partner condition has the highest mean blue
decoding score in both versions, while the single-partner condition has the
highest mean red score. In v2 the red scores of those two conditions overlap
substantially across seeds; diversity does not consistently lead on both targets.
