# Episode-return analyses: direct input and t0 fix

These four result families use only the five-seed
`v1_matched_v2_v3_20261022_080155_t0fix_direct_input` cohort, with step penalty
**0.01**. The previous four analysis directories for the 0.03 cohort were removed
and their methods rerun on the current saved evaluations.

| Results | Coverage and outputs |
| --- | --- |
| [Return distributions](t0fix_direct_input/) | All 60 policies, 55,200 episodes; distributions, summaries, input manifest, extraction caches, frozen producers and completion audit. |
| [Training-selected lowest decile](lowest_decile_analysis/report.md) | Best training seed for each of 12 protocol/condition cells; 1,104 tail episodes, 48 representative cases, behavior diagnoses, actor/critic audit, and six decision-time capability decoders with shuffled-label controls. |
| [All-seed lowest decile](lowest_decile_analysis_all_seeds/report.md) | All 60 policies; 5,520 tail episodes, 240 representative cases, individual learner reports, behavior diagnoses and actor/critic audit. |
| [Single-partner generalization](single_partner_generalization/report.md) | All 30 diverse/single-partner RNNs; return, success, assignment, switching and critic comparisons across the actual training pair, other reference profiles and held-out profiles. |

Each report also has an HTML version. Figures are saved as PNG, PDF and SVG;
underlying tables are CSV or compressed CSV. Raw current evaluations and trained
policies remain their original inputs.

The lowest decile is exactly 92 of 920 episodes per policy, with deterministic
boundary-tie handling. Selected learners maximize the final unsmoothed training
return; evaluation outcomes do not choose the seed. Four distinct cases are
chosen sequentially from the remaining tail: worst return, near cutoff, most
switches, largest late gain. Their labels describe that sequential selection.
The partner-faster assignment measure compares capability delays; it is a proxy,
not a geometry-aware optimum. Separate oracle tables account for layout geometry.

The historical analysis producer was unavailable. The maintained runner
reconstructs methods from the previous reports and saved arrays, using current
shared helpers where appropriate. Reports explain the reconstructed methods and
their limits. Policies are not retrained and no new environment rollouts are
collected. The selected-policy audit fits 36 small diagnostic capability probes
to saved GRU states, with an episode split within each capability profile.

Run from the repository root in the existing representation environment (or one
with the dependencies in `requirements/representation.txt`):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  python eval/rerun_episode_return_analyses.py run
python eval/rerun_episode_return_analyses.py validate \
  --root eval/episode_return_distributions
```

`run` builds an isolated `.t0fix_direct_input_build` tree and validates it. Its
seven independent frozen-network checks use the existing training environment.
The one-time replacement of the historical outputs is complete; its deletion
command has been removed from the maintained runner. The published
[completion audit](t0fix_direct_input/completion_audit.json) records exact coverage,
source checks, hidden-state reconstruction errors, and unchanged policy hashes.
