"""Build a reviewable scientific report from completed diagnostic analyses."""
import argparse,html,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.learning_diagnostics import write_json,sha


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis-dir',type=Path,required=True);a=p.parse_args()
    import pandas as pd
    out=a.analysis_dir;t=pd.read_csv(out/'checkpoint_timeline.csv');end=pd.read_csv(out/'endpoint_behaviors.csv');ci=pd.read_csv(out/'paired_interventions.csv')
    final=end[(end.population=='heldout')&end.intervention.eq('baseline')].set_index('index')
    familiar=end[(end.population=='training')&end.intervention.eq('baseline')].set_index('index')
    def name(i):return final.loc[i,'label']
    interpretation={
        2:'Early completion improvement; temporary blue preference; later partner adaptation.',
        3:'Slow completion learning; partner matching stays near chance. Fixed colors do not rescue it.',
        7:'Early blue commitment becomes durable; high completion, little partner adaptation.',
        4:'Early red commitment, then much later productive partner-dependent correction.',
        5:'Switching-dependent policy on one partner; severe transfer asymmetry. Global holding makes it worse.',
        9:'Poor completion with many late switches; holding assignments does not close the seed-10 gap.',
        11:'Efficient red commitment; high success without strong partner matching.',
        6:'Fast early learning; temporary transfer dips, then good final transfer with few switches.',
        10:'Late transfer instability after learning succeeds; suppressing later assignment changes rescues most failures.'}
    rows=[]
    for i in (2,3,7,4,5,11,9,10,6):
        r=final.loc[i]
        rows.append([r.label,f'{r.return_:.2f}',f'{100*r.success:.1f}%',f'{100*r.matching:.1f}%',f'{r.switches:.2f}',interpretation[i]])
    endpoint_table=table(['Target','Held-out return','Success','Partner matching','Switches / round','Diagnosis'],rows)
    rows=[]
    for i in (4,5,9,11,6,10):
        r=ci[(ci['index']==i)&ci.population.eq('heldout')&ci.intervention.eq('hold_after_first_request')].iloc[0]
        base=final.loc[i];held=end[(end['index']==i)&end.population.eq('heldout')&end.intervention.eq('hold_after_first_request')].iloc[0]
        rows.append([name(i),f'{100*base.success:.2f}% → {100*held.success:.2f}%',
            f'{100*r.delta_success:+.2f} pp [{100*r.success_ci_low:+.2f}, {100*r.success_ci_high:+.2f}]',
            f'{r.delta_return_:+.3f} [{r.return__ci_low:+.3f}, {r.return__ci_high:+.3f}]'])
    intervention_table=table(['Target','Baseline → held success','Success change, 95% interval','Episode-return change, 95% interval'],rows)
    windows=json.loads((out/'behavior_windows.json').read_text())['comparisons']
    rows=[]
    for w in windows:
        x=w['lasting_heldout_return_gap_at_least_one']
        rows.append([name(w['left'])+' vs '+name(w['right']),str(x['previous_checkpoint'])+' → '+str(x['first_observed_checkpoint']),
            f"{x['first_observed_transitions']:,}",f"{x['previous_transitions']/1e6:.3f}–{x['first_observed_transitions']/1e6:.3f}M"])
    windows_table=table(['Comparison','Observed onset bracket','First observed transition count','Transition interval'],rows)
    clocks=pd.read_csv(out/'switch_clock_mechanisms.csv');case=json.loads((out/'illustrative_switch_failure.json').read_text())
    audits=json.loads((out/'audit.json').read_text())
    checks=sum(x['scalar_checks'] for x in audits['runs'])
    content=f'''<!doctype html><html><head><meta charset="utf-8"><title>Learning divergence: nine completed RNN targets</title>
<style>body{{max-width:1150px;margin:35px auto;padding:0 24px;font:16px/1.55 system-ui,sans-serif;color:#20252c}}h1,h2{{line-height:1.2}}h2{{margin-top:38px}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}td,th{{padding:9px;border-bottom:1px solid #d6dce3;text-align:left;vertical-align:top}}th{{background:#eef2f6}}img{{max-width:100%;height:auto}}.note{{background:#f3f6fa;border-left:4px solid #668aad;padding:14px}}a{{color:#1764ab}}code{{background:#f0f2f4;padding:2px 4px}}small{{color:#536170}}</style></head><body>
<h1>What diverged during learning in the nine completed RNN targets?</h1>
<p><b>Analysis date: 2026-10-18.</b> The evidence supports several mechanisms, rather than a single seed-collapse explanation: an early completion-learning split, stable color specialization, productive online correction, and later transfer instability. The strongest causal rescue is v3 RNN-single seed 7. Reducing switching globally is harmful for several other policies.</p>
<p>All nine replays completed 915 updates, passed {checks:,} historical scalar comparisons, and reproduced the original final parameters exactly. This validates the reproductions strongly; unsaved original actions cannot independently be compared bit for bit. The three pending MLP targets are excluded.</p>
<h2>1. The first v1_matched split is completion learning, before partner adaptation</h2>
<p>The held-out return gap between RNN-multi seeds 1 and 2 first exceeds one return unit after <b>optimizer update 12, at 786,432 transitions</b>, and remains above that threshold at every later four-packet screen. Checkpoint 11 does not have this gap. On the common panel, success changes from 46.0% versus 38.9% at checkpoint 12 to 64.1% versus 42.7% at checkpoint 16. Partner matching is still near 50% for both.</p>
<p>The saved interventions separate this from merely preferring different colors. At checkpoint 16, forcing ego-red yields <b>68.52% versus 44.55%</b> success for seeds 1/2; forcing ego-blue yields <b>61.88% versus 42.22%</b>. Seed 1 completes more grids under either assignment. This supports an early improvement in assignment-conditioned navigation/completion, rather than early inference of partner ability.</p>
<p>The original “shorter rounds” result is now directly explained by timeout avoidance. In training updates 13–20, seed 1 completes 7,286 rounds at 59.91% success, versus 6,416 at 39.15% for seed 2. Successful rounds average <b>53.57 versus 53.01 ticks</b>—almost the same. Failed rounds take 100 ticks. The mean over completed rounds is therefore <b>72.18 versus 81.60 ticks</b>. These are rounds ending in the window; some begin before it.</p>
<p>Seed 1 briefly becomes strongly blue-biased (under 5% ego-red at checkpoints 33–44), then learns partner dependence. Its first sustained ≥75% matching begins at checkpoint 65 (4.26M transitions). Seed 4 instead becomes ≥95% blue by checkpoint 22 and retains chance-level partner matching. Seed 2 never reaches the matching threshold despite eventually improving completion. By the end, seeds 1/2/4 have completed <b>2,267,348 / 1,002,856 / 1,545,999</b> rounds. Unequal opportunities can amplify learning differences; these observations do not by themselves isolate the causal direction of that feedback.</p>
<p>Seed 2's mixed choices are not interchangeable with constant colors: at the final checkpoint, forcing either color reduces success from 91.88% to about 76.6%. The choices may exploit layout/history or a learned navigation coupling even without partner matching. Chance-level matching does not mean the choices are random or useless.</p>
<p><a href="matched_updates13_20_round_lengths.json">Round lengths</a> · <a href="early_fixed_assignment_controls.csv">Assignment controls</a> · <a href="matched_early_onpolicy_diagnostics.csv">Original on-policy batch diagnostics</a> · <a href="training_exposure_counts.csv">Opportunity counts</a></p>
<img src="focal_first_10M.png" alt="Early behavior learning in v1_matched and v3 comparisons">
<h2>2. When the observed performance gaps become lasting</h2>
<p>These markers concern held-out checkpoint returns, not the earlier completed-training-episode return screen. They use a ≥1 return gap persisting through the last four-packet screen, with at least three qualifying observations. The endpoint is confirmed on twenty packets. Brackets identify observed checkpoints; coarse intervals do not identify a unique optimizer step. For v1_matched seeds 1/2, the consecutive 11→12 bracket nominates optimizer update 12 specifically for finer attribution.</p>
{windows_table}
<p>The older training-return marker at update 19 is compatible with the new checkpoint-12 result: logged rollouts use the preceding policy, completed 20-round episodes lag policy changes, and the older detector used smoothed training returns. They measure different clocks and populations.</p>
<img src="paired_gap_windows.png" alt="Single-seed pair comparisons with sampled gap-onset lines and interval shading">
<h2>3. What the final policy interventions establish</h2>
<p>The intervention below permits the initialization and the first movement-phase assignment request, then holds the effective assignment. It starts afresh in every round and remains active throughout the partner episode. <b>It is not “wait until the first actual switch, then hold,” and it is not a local intervention at the observed burst.</b> Navigation actions continue to be sampled, and observations and GRU states reflect the applied actions.</p>
{intervention_table}
<p>Intervals resample twenty shared grid packets jointly across all partner profiles (5,000 draws). They quantify variation across this evaluation panel, conditional on these policies, profiles, and evaluation random stream; they are not confidence intervals across independent training seeds.</p>
<img src="switch_intervention_effects.png" alt="Paired intervention effects and packet-bootstrap uncertainty">
<p><b>V3 RNN-single seed 7: a meaningful rescue.</b> Limiting assignment changes after the first request raises success from 92.57% to 98.57% (+6.00 pp, interval +4.86 to +7.00) and return by 1.97. Holding from the start of movement also helps, and fixing ego-blue gives 99.17% success. Seed 6 changes little under the same request-limiting intervention. Thus the later assignment-changing behavior contributes to seed 7's transfer failures under these closed-loop interventions.</p>
<p><b>V2 RNN-single seed 8: poor transfer, but no simple switch-suppression rescue.</b> Its actual-training-profile success is 86.5%, versus 49.31% held out. It still averages 15.29 switches per training-profile round and 27.87 held out. The intervention lowers held-out success to 23.72% and training-profile success to 24.25%. Holding from the start and forcing either color also hurt. The learned navigation/interaction is coupled to its switching pattern; a high switch count alone does not demonstrate that suppressing it improves this policy.</p>
<p><b>V3 RNN-multi seed 4: switching does not explain the large gap through this simple remedy.</b> Request-limiting reduces success from 80.67% to 78.85%; even with ego-red fixed, seed 4 succeeds 78.83% versus seed 10's 98.02%. A substantial completion deficit remains with the same assignment imposed. This does not prove every switch is beneficial; it rules out “just prevent switches” as the explanation/remedy for most of the between-seed gap.</p>
<p><b>V2 RNN-multi seed 10: productive correction.</b> It initially learns a near-fixed red policy, then develops partner matching much later: first repeated ≥75% matching around update 363, lasting from 374 (24.51M). At the final checkpoint, request-limiting lowers matching from 91.1% to 72.4% and success from 98.52% to 95.66%. Holding the random initial assignment hurts further. Its online corrections are useful under these tests.</p>
<p>V3 RNN-multi seed 10 achieves high completion with red commitment and little partner matching. Preventing its late switches costs only about 0.23 return and 0.59 pp success. These are different successful solutions, rather than one universal adaptive strategy.</p>
<h2>4. Seed 7 exhibits later transfer decay, not simply an unrecovered bad start</h2>
<p>V3 single seed 7 improves to 95.5% held-out success at checkpoint 96, close to seed 6's 95.9%. A conspicuous later deterioration occurs between <b>checkpoints 144 and 160 (9.44–10.49M transitions)</b>: held-out success falls 94.66%→82.56%, return 12.32→8.14, and switches 2.03→7.73 per round. Over the same interval the fixed actual-training-profile panel improves from return 16.82/success 98.75% to return 17.21/success 100%.</p>
<p>This behavior recovers partially and fluctuates; it is not a single irreversible collapse. Seed 6 also has temporary held-out dips around updates 256 and 304. The later persistent ≥1 return gap favoring seed 6 is first observed at checkpoint 352, after checkpoint 336. Thus 144→160 is a first sharp later failure window, while 336→352 is the lasting-gap bracket. Neither a final training score nor a single early audit would reveal the full story.</p>
<img src="late_transfer_decay.png" alt="Stable training performance and later held-out switching instability">
<h2>5. Failures depend on partner orientation and delay</h2>
<p>V2 single seed 8 succeeds <b>81.09%</b> when the partner is red-faster (the orientation seen in its single training partner), versus <b>17.52%</b> when the partner is blue-faster. Request-limiting changes these to 25.36% and 22.07% respectively: the pooled harmful effect conceals a modest improvement on the already-poor orientation and a large loss on the familiar orientation.</p>
<p>V3 single seed 7 succeeds 99.02% on red-faster profiles versus 86.11% on blue-faster profiles. The request-limiting intervention changes these to 98.43% and 98.70%. On partner (9,0), baseline success is 63.25%, versus seed 6's 94.75%. This pattern implicates transfer to slower red partners under the learned ego-blue commitment; it does not establish that the zero-delay value alone causes failure.</p>
<p>Switches in v2 single seed 8 usually start immediately (median first-switch tick 1), whereas the median is tick 24 in v3 single seed 7 and v3 multi seed 4. Approximately 79% and 83% of switched rounds in the latter two have their first switch after tick 10. These are distinct behavioral profiles. Positive destination delays can reload the partner clock, but the clock statistics alone cannot prove that switching caused an episode's failure.</p>
<img src="partner_profile_success.png" alt="Success conditional on partner capability for all nine policies">
<h2>6. A concrete failure that can be replayed</h2>
<p>V3 single seed 7, final checkpoint, profile (9,0), packet 10, episode 210, round 15, grid <b>{case['layout_id']}</b> (catalog index 960): the ego first reaches blue at tick 16. At that same pre-action tick the partner has not reached red and its delay counter is 5. The policy switches to ego-red/partner-blue. The round subsequently contains 28 switches and times out. The paired request-limited partner episode completes 20/20 rounds versus baseline 10/20.</p>
<p>This is one observed case, not evidence that every burst begins for the same reason. The paired intervention changes earlier rounds and hidden-state history; a local branch immediately before tick 16 is still needed to isolate that first switch's effect. Terminal GIF frames reconstruct the frozen environment's wall/collision/swap movement rather than showing the next grid's reset pose.</p>
<img src="recorded_switch_failure.gif" alt="Recorded burst failure and paired episode-level assignment intervention">
<p><a href="illustrative_switch_failure.json">Exact case metadata</a> · <a href="replayable_failure_cases.json">Fifteen inspectable cases</a> · <a href="switch_clock_mechanisms.csv">Clock and movement statistics</a></p>
<h2>7. Summary of all nine final behaviors</h2>
<p>Values below use the new common fixed-packet panel, not the earlier independently sampled evaluation grids. Small differences from the original phenotype table are expected even though the weights match exactly.</p>
{endpoint_table}
<h2>What remains unresolved, and the next informative checks</h2>
<p>We have localized the early split and separated harmful, useful, and coupled switching behaviors. We have <b>not yet established why seed 1's optimizer update 12 improves completion while seed 2's does not</b>. The observed on-policy advantages and critic explained variance do not show a clean critic-collapse signature; raw advantage means alone are not a diagnosis because the implemented PPO normalizes advantages within minibatches.</p>
<ol><li><b>V1_matched update 12:</b> replay each seed's own saved rollout from checkpoint 11; verify ratio≈1 before SGD, then track movement versus allocation policy change, KL, clipping, value error and gradient norms across the 256 minibatches. Probe common episodes along the update to identify which minibatches improve completion. Selected LR/order forks can test whether those changes caused the later split.</li>
<li><b>V3 single seed 7:</b> refine the 144→160 and 336→352 windows, and branch immediately before the observed late switches on (9,0) and related profiles. Permit/suppress a switch while keeping the same earlier history; compare partner readiness, movement and completion. This distinguishes harmful correction from a reaction to an already difficult state.</li>
<li><b>V2 single seed 8 and v3 multi seed 4:</b> use validated navigation/oracle-informed assignment controls and local branches. Global assignment holding was not a general rescue; interpreting frequent switches as the sole defect would discard the intervention evidence.</li></ol>
<p>These checks use PPO's own on-policy batches and repeated minibatch optimization, as described in <a href="https://arxiv.org/abs/1707.06347">Schulman et al., Proximal Policy Optimization Algorithms</a>. Keeping source/runtime and implementation details fixed is motivated by <a href="https://arxiv.org/abs/2005.12729">Engstrom et al., Implementation Matters</a>. The seed-specific conclusions above come from the local data, not those papers.</p>
<h2>Methods and limits</h2>
<ul><li>Fixed grid packets and held-out partner populations are shared across policies. GRU history spans all twenty rounds, with original episode resets. No RNN-without-influence condition is included.</li>
<li>Most checkpoints use four packets per profile. Update 0, update 915 and selected nominated onsets use twenty. The changed batch shape changes stochastic draws; stars are confirmations, and threshold screens exclude those twenty-packet checkpoints. Single-step causal claims are not inferred from a mixed-panel jump.</li>
<li>Pairs are individual targets from the approved mechanistic cohort: v1_matched 1/2/4, v3 multi 4/10, and v3 single 6/7. No median reference is substituted. Checkpoint behavior for uncollected reference seeds is unavailable; earlier best-training-reference plots remain a separate training-log analysis.</li>
<li>Interventions are paired complete-episode policy changes, not retrospective relabeling or independent round treatments. Confidence intervals resample grid packets jointly across profiles and do not imply population-level effects over training seeds.</li>
<li>Original summary rewards accumulate up to 2,000 float32 values; recomputed ledger episode totals differ by at most {audits['max_return_accumulation_difference']:.6f}. Behavioral aggregates match to numerical precision; return comparisons use a 0.001 validation tolerance. This rounding issue is much smaller than reported effects.</li>
<li>The timeline localizes behavior changes. It does not establish a unique defective grid or optimizer step without the remaining causal forks.</li></ul>
<p><a href="audit.json">Provenance and validation</a> · <a href="checkpoint_timeline.csv">All checkpoint measurements</a> · <a href="paired_interventions.csv">Paired effects and uncertainty</a> · <a href="behavior_windows.json">Window definitions and coordinates</a> · <a href="final_profile_behaviors.csv">Per-profile results</a></p>
</body></html>'''
    (out/'report.html').write_text(content)
    write_json(out/'report_manifest.json',dict(report_sha256=sha(out/'report.html'),report_script_sha256=sha(__file__),
        primary_data_audit_sha256=sha(out/'audit.json'),files={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.png','.gif','.json') and p.name!='report_manifest.json'}))
    print(out/'report.html')

if __name__=='__main__':main()
