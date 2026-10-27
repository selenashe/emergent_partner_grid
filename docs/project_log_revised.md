# CoordinationGrid experiment status

The maintained comparison cohort is the t0-fixed, direct-input v1_matched, v2,
and v3 experiments. Each has four model conditions and five learner seeds
(1–5). V1_matched is the fixed-allocation baseline;
its no-influence control uses an independent uniform assignment each round.

Sources, configs, layouts, counterbalanced schedules, training/evaluation job
records, and current results are identified by
`train/manifests/sbatch_v1_matched_v2_v3_20261022_080155_t0fix_direct_input.json`.

Preparation and tests use the retained executed sources. Shared layouts and
counterbalanced schedules support the current cohort. Each saved policy is
evaluated with its matching frozen trainer and environment.

See [methods](counterbalanced_methods.md), [matched baseline](v1_matched.md),
[architecture](architecture.md), and the repository README for reproduction.
