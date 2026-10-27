# Methods: maintained CoordinationGrid allocation experiments

The comparison cohort is `v1_matched`, `v2`, and `v3`. Each experiment contains
diverse-partner RNN + influence, diverse-partner MLP + influence, single-partner
RNN + influence, and diverse-partner RNN without influence. Learner seeds 1–5
are equally weighted; summaries report sample standard deviations across seeds.

## Shared task and training

Only ego is learned; the partner follows scripted movement rules. A partner
episode has 20 rounds on the common 1,096-layout corpus. The diverse training
population contains 24 specialist capability profiles; the single-partner
condition uses (1,4). Evaluation uses 24 familiar and 22 novel partner profiles,
with 20 episodes per profile, categorical actions, and evaluation seed 12345.
Layouts remain familiar during the partner generalization evaluation.

The same counterbalanced layout queues and profile schedules are shared across
experiments. The schedule prefix accounts for active unfinished worker episodes
at the fixed-step cutoff. Training uses 256 environments × 256 steps, four PPO
update epochs, 64 minibatches, and 915 updates: 59,965,440 collected transitions
per policy against a 60M nominal budget. The primary step penalty is 0.01;
the maintained direct-input cohort retains this reward unchanged.

RNN conditions concatenate 245 grid values, three allocation values and the
learned t0 phase bit directly into a 128-unit GRU. The MLP condition uses the
same 249-value observations with a Dense(128) temporal replacement.
Policy and value heads are separate. See [architecture](architecture.md) for
input channels, layer dimensions, action masks, and hidden-state capture timing.

## Allocation protocols

| Experiment | Protocol tag | Allocation rule |
| --- | --- | --- |
| v1_matched | fixed_allocation | Ego chooses at round start; allocation stays fixed. Without influence, draw a fresh uniform assignment each round. |
| v2 | online_v2 | Random initial assignment; subsequent requests can change the partner goal during movement. |
| v3 | online_v3 | Ego chooses the initial assignment with influence; later transitions follow v2. Without influence, initialization remains random. |

All maintained recurrent trainers replay the pre-observation episode-reset flag.
They retain post-transition done for GAE. Memory carries across rounds within a
partner episode and resets at the next episode.

## Evidence and analyses

`train/manifests/` identifies source inventories, configs, schedule/corpus hashes,
checkpoint and rollout locations, seed mappings, and Slurm job records. Large
weights, HDF5s, and schedule arrays live under the scratch artifact root;
repository symlinks expose their logical paths. Regenerating schedules requires
the frozen sampler and exact recorded numerical checksums.

`eval/compare_allocation_protocols.py` compares the three maintained experiments
using the current manifest and all five seeds.
`train/plot_protocol_training_curves.py` uses the same cohort. The relative-speed
analysis uses their three RNN conditions and all five seeds. Within each policy,
probe fitting and held-out episodes remain separated by the recorded split.
