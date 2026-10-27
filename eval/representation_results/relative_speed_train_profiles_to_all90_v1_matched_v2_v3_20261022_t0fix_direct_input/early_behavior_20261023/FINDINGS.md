# What happens around the early decoding rise

The clearest common correlate is the partner's initial movement-and-waiting cadence. Decoding usually improves during an ongoing first-goal attempt, before round feedback and often before pursuing the other goal. The actual assignment of the faster goal stays close to 50% during these early windows.

The analysis uses the latest instantaneous h_t probes fitted only on the 24 training capabilities, with the original four held-out episodes per profile and five policy seeds.

| Experiment | Green: seen | Yellow: one novel delay | Pink: both novel |
|---|---|---|---|
| v1_matched | 4–9 (+16.3 pp) | 2–7 (+12.0 pp) | 3–8 (+23.3 pp) |
| v2 | 12–17 (+11.7 pp) | 3–8 (+9.4 pp) | 11–16 (+15.8 pp) |
| v3 | 4–9 (+17.5 pp) | 3–8 (+12.4 pp) | 5–10 (+20.0 pp) |

Windows maximize five-step accuracy gain among steps 0–50. They are descriptive summaries, not precise biological or causal change points. Bootstrap timing intervals are in onset_summary.csv; v2 pink is especially uncertain.

For v1_matched green, accuracy increases 48.1%→64.4% across steps4–9, while only 1.0% of episodes have completed a round and none have pursued both goals. Yellow rises 46.3%→58.3% across2–7; pink rises46.7%→70.0% across3–8. In both cases no episodes have pursued both goals, and round feedback has reached only1.0% and5.8%, respectively.

Across all nine experiment/category combinations, matched cohorts with no round feedback and only one pursued goal gain11–25 percentage points across their selected windows. These are conditional, descriptive comparisons using identical episodes and profile weights at both endpoints.

v2 is behaviorally different: by its green window end44.8% have pursued both goals and18.8% have received first-round feedback; for pink the corresponding fractions are41.7% and27.5%. The matched one-goal/no-feedback cohorts still improve, so neither event is required.

Motion reconstruction recovered5361/5400 held-out episode prefixes through step30 (99.28%). It replayed the original batch sizes, reset seeds, action keys, and frozen policies; the saved h_t anchored actor inputs and saved h_(t-1) anchored recurrent inputs. Every accepted prefix matches all originally saved behavior and one-step GRU states within an absolute .002 tolerance for measured CPU/GPU matmul differences. Any mismatch invalidates later steps;39 prefixes are excluded from motion summaries. These are reconstructed trajectories, not original position logs.

A repeated move within four transitions or four consecutive off-goal stays after a move describes a fast/slow cadence pattern consistent with the training families. In v1_matched, this pattern has been observed by the end of all three rise windows in100% of the validated cohort (vs17%,0%,30% at the window starts). The corresponding v3 endpoint fractions are97.5%,97.1%,95.8%. These patterns provide temporal context and do not prove that the network implements a particular estimator.

An illustrative held-out episode is v1_matched, seed1, profile(1,7), repetition1 (zero-based). The partner pursues blue throughout. Its moves occur on transitions1,9,17, visible to the policy at observations2,10,18. During the first long wait, the time-specific decoder's red-faster probability jumps from8.4% at step7 to96.6% at step8. There is no goal change or round feedback in the window. A fixed step20 readout becomes confident later, around step11, so the exact sharp jump depends partly on which time-specific readout is used. In the example figure, black squares mark the partner, orange circles the ego, and red/blue cells the goals.

The findings support early availability of information about the partner's timing during the ongoing interaction. They do not separate retrieval in green from inference or predictive coding in yellow/pink. Familiar training pairs couple fast delays1–3 with slow delays4,7,8,9; a readout can exploit the observed goal's speed family before both delays have been sampled. No causal intervention has been performed.

Outputs: decoding_and_behavior.png/pdf; decoding_and_partner_motion.png/pdf; paired_gain_before_feedback.png/pdf; example_partner_waiting.png/pdf; pink_profile_strata.png/pdf; the accompanying episode-level and seed-level CSVs, saved_behavior_audit.json, motion_audit.json, and frozen sources.
