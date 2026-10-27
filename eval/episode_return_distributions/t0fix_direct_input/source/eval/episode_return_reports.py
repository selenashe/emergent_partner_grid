"""Reports and figures for current-cohort saved episode-return diagnostics."""
from __future__ import annotations

import html
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from eval.episode_return_analysis import (BASE, BATCH, BUILD, CONDITIONS, ROOT,
    SEEDS, TRAINING_MANIFEST, VERSIONS, identity, key, require, sha, write)

COLORS={'mlp_diverse_influence':'#06b48b','rnn_diverse_influence':'#8e82fe',
        'rnn_single_influence':'#e17701','rnn_diverse_noinfluence':'#5a86ad'}
LABELS={'mlp_diverse_influence':'Diverse MLP','rnn_diverse_influence':'Diverse RNN',
        'rnn_single_influence':'Single-partner RNN','rnn_diverse_noinfluence':'RNN without influence'}
GROUPS=('training_partner','other_reference','held_out')
GROUP_LABELS={'training_partner':'Training pair (1,4)','other_reference':'Other 23 reference profiles','held_out':'22 held-out profiles'}
LIMITATIONS=('All measurements use the original saved 480 familiar and 440 novel episodes per policy. '
    'Terminal transitions are included and scan padding is excluded. The reward is '
    'R = 1.01 × successful rounds − 0.01 × total ticks; there is no explicit switching fine. '
    'An assignment labels the ego goal; the partner takes the other goal. '
    'Capability delay d allows a movement opportunity every d+1 ticks. '
    'Switch counts exclude initialization and round resets. Switching and returns are observational associations, not treatment effects. '
    'Positions and navigation actions were not recorded, so these records cannot distinguish wandering, collisions, and waiting at goals. '
    'A trend inside an outcome-selected tail is not evidence of learning or memory use. '
    'The single-partner RNN trained only on (1,4); the other 23 familiar-reference profiles are unseen for it.')


def export(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'):fig.savefig(path.with_suffix('.'+ext),dpi=150,bbox_inches='tight')
    plt.close(fig)


def table(df,columns=None,limit=None):
    df=df[columns] if columns else df
    if limit:df=df.head(limit)
    def cell(v):
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        return str(v).replace('|','/').replace('\n',' ')
    lines=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    return '\n'.join(lines+['| '+' | '.join(map(cell,row))+' |' for row in df.itertuples(index=False,name=None)])


def report(path,text):
    """Render the small Markdown subset used here without external dependencies."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text+'\n')
    def inline(value):
        value=html.escape(value)
        value=re.sub(r'!\[([^]]*)\]\(([^)]+)\)',r'<img alt="\1" src="\2">',value)
        value=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',value)
        value=re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',value)
        return value
    body=[];inside=False
    for line in text.splitlines():
        if line.startswith('|'):
            if set(line.replace('|','').replace('-','').replace(':','').strip())==set():continue
            if not inside:body.append('<table>');inside=True
            body.append('<tr>'+''.join('<td>'+inline(x.strip())+'</td>' for x in line.strip('|').split('|'))+'</tr>');continue
        if inside:body.append('</table>');inside=False
        if line.startswith('#'):
            n=len(line)-len(line.lstrip('#'));body.append(f'<h{n}>'+inline(line[n:].strip())+f'</h{n}>')
        elif line:body.append('<p>'+inline(line)+'</p>')
    if inside:body.append('</table>')
    path.with_suffix('.html').write_text('<!doctype html><meta charset="utf-8"><title>Direct-input episode analysis</title>'
        '<style>body{font:16px system-ui;max-width:1500px;margin:40px auto;padding:0 24px;line-height:1.5}table{border-collapse:collapse;font-size:13px}td{padding:7px;border:1px solid #ddd}img{max-width:100%}a{color:#1464a0}</style>'+''.join(body))


def cache(root,record):return Path(root)/'t0fix_direct_input/cache'/key(record)


def return_distributions(episodes,policies,root):
    out=Path(root)/'t0fix_direct_input'
    episodes.to_csv(out/'episode_returns.csv.gz',index=False)
    summary=episodes.groupby(['version','condition','population','seed']).episode_return.agg(['mean','std','min','max','count']).reset_index()
    summary.to_csv(out/'distribution_summary.csv',index=False)
    # Pooling within each panel is explicit: five independent learners, 20
    # repetitions for each profile, no best-seed selection.
    for name,by_population in [('all_episode_returns',False),('by_condition_and_population',True)]:
        fig,axes=plt.subplots(4,3,figsize=(14,12),sharex=True,layout='constrained')
        for ci,condition in enumerate(CONDITIONS):
            for vi,version in enumerate(VERSIONS):
                ax=axes[ci,vi];d=episodes[(episodes.version==version)&(episodes.condition==condition)]
                bins=np.arange(-20.25,20.51,.5)
                if by_population:
                    for pop,color in [('train','#2877b8'),('test','#e08729')]:ax.hist(d.loc[d.population==pop,'episode_return'],bins=bins,density=True,histtype='step',color=color,label=pop)
                else:ax.hist(d.episode_return,bins=bins,density=True,color=COLORS[condition],alpha=.8)
                ax.set_title(version+' / '+LABELS[condition],fontsize=10);ax.set_xlabel('20-round episode return');ax.set_ylabel('Density')
        if by_population:axes[0,0].legend()
        fig.suptitle('Direct input + t0 fix · step penalty 0.01 · all five seeds')
        export(fig,out/name)
    sources=[]
    for p in policies:sources.extend(json.loads((cache(root,p)/'completion.json').read_text())['sources'])
    write(out/'provenance.json',dict(cohort=BATCH,manifest=str(TRAINING_MANIFEST),manifest_sha256=sha(TRAINING_MANIFEST),
        step_penalty=.01,seeds=list(SEEDS),policies=60,episodes=len(episodes),sources=sources,
        return_definition='Sum original saved rewards through the first episode done, inclusive; float64 accumulation.',
        pooling='All five learners; 480 reference and 440 held-out episodes per policy. Plots are descriptive pooled densities.',
        validation='Complete ordered 20-round episodes, 20 repeats/profile, finite returns, exact reward ledger, saved summary agreement within 0.0002.'))


def aggregate_math(policies,out,root):
    target=out/'math_audit';target.mkdir(parents=True,exist_ok=True)
    names=['initial_calibration.csv','rnn_round_math.csv.gz','rnn_episode_math.csv','rnn_phase_math.csv',
           'observation_channel_probes.csv.gz','mlp_start_probabilities.csv.gz','switch_event_math.csv.gz','case_action_probabilities.csv.gz']
    combined={};coverage=[]
    for p in policies:
        audit=json.loads((cache(root,p)/'math/completion.json').read_text())
        cfg=json.loads(Path(p['config']).read_text())
        coverage.append(dict(identity(p),**{k:audit[k] for k in ['saved_hidden_states_analyzed','maximum_start_hidden_reconstruction_error','observation_probes','mlp_start_observations']},
            **{k:cfg[k] for k in ['GAMMA','GAE_LAMBDA','CLIP_EPS','ENT_COEF','VF_COEF','MAX_GRAD_NORM','LR']},full_training_trace=False,step_penalty=.01))
    pd.DataFrame(coverage).to_csv(target/'coverage_and_hyperparameters.csv',index=False)
    for name in names:
        frames=[]
        for p in policies:
            path=cache(root,p)/'math'/name
            if path.exists():
                frame=pd.read_csv(path)
                if name=='initial_calibration.csv':frame=frame[frame['round']==1]
                if name=='observation_channel_probes.csv.gz':
                    forced=((frame.version=='v2')|((frame.version=='v3')&(frame.condition=='rnn_diverse_noinfluence')))&(frame.round_tick==0)
                    frame['allocation_choice']=frame.allocation_choice&~forced
                    frame['effective_allocation_control']=frame.condition!='rnn_diverse_noinfluence'
                frames.append(frame)
        if frames:
            data=pd.concat(frames,ignore_index=True);data.to_csv(target/name,index=False)
            if name in ('initial_calibration.csv','rnn_round_math.csv.gz','observation_channel_probes.csv.gz','switch_event_math.csv.gz'):
                combined[name]=data
    initial=combined['initial_calibration.csv'];initial['squared_error']=initial.value_bias**2
    seed=initial.groupby(['version','condition','seed','group'])[['value','discounted_return','value_bias','squared_error']].mean().reset_index()
    seed['rmse']=np.sqrt(seed.squared_error);seed.to_csv(target/'seed_group_critic_calibration.csv',index=False)
    equal=seed.groupby(['version','condition','group']).mean(numeric_only=True).drop(columns='seed').reset_index();equal.to_csv(target/'equal_seed_critic_calibration.csv',index=False)
    calibration=initial.groupby(['version','condition','population','lowest_decile'])[['value','discounted_return','value_bias']].agg(['mean','count'])
    calibration.to_csv(target/'critic_calibration_summary.csv')
    rounds=combined['rnn_round_math.csv.gz']
    cols=['decision_p_partner_faster','decision_p_red','mean_p_switch','mean_subsequent_p_switch','mean_entropy','initial_value_bias','critic_rmse','terminal_td']
    seed_math=rounds.groupby(['version','condition','seed'])[cols].mean().reset_index()
    sw=combined.get('switch_event_math.csv.gz',pd.DataFrame())
    if len(sw):
        event=sw.groupby(['version','condition','seed']).p_switch.agg(switch_events='count',switch_p_median='median',mean_switch_p='mean').reset_index()
        seed_math=seed_math.merge(event,on=['version','condition','seed'],how='left')
    seed_math.to_csv(target/'seed_rnn_math_summary.csv',index=False)
    seed_math.groupby(['version','condition']).mean(numeric_only=True).drop(columns='seed').to_csv(target/'equal_seed_rnn_math_summary.csv')
    rounds.groupby(['version','condition','seed','round'])[['decision_p_partner_faster','mean_p_switch','initial_value_bias']].mean().reset_index().to_csv(target/'seed_round_math_summary.csv',index=False)
    probes=combined['observation_channel_probes.csv.gz'];eligible=probes[probes.allocation_choice]
    columns=['memory_probability_effect','fixed_mask_message_probability_effect','p_partner_faster_factual','p_partner_faster_zero_history']
    sp=eligible.groupby(['version','condition','seed','round_tick'])[columns].mean().reset_index()
    sp.to_csv(target/'seed_observation_probe_summary.csv',index=False)
    sp.groupby(['version','condition','round_tick'])[columns].mean().reset_index().to_csv(target/'equal_seed_observation_probe_summary.csv',index=False)
    probes.groupby(['version','condition','seed']).agg(rows=('reconstruction_within_0_01','size'),valid_rows=('reconstruction_within_0_01','sum'),maximum_hidden_error=('factual_hidden_max_abs_difference','max')).reset_index().to_csv(target/'observation_probe_coverage.csv',index=False)
    # The new phase bit removes the old MLP t0/t1 input alias. Measure the
    # resulting values directly rather than importing the old diagnosis.
    mlp_path=target/'mlp_start_probabilities.csv.gz'
    if mlp_path.exists():
        mlp=pd.read_csv(mlp_path)
        phase=mlp.pivot(index=['version','condition','seed','population','episode_index','round'],columns='round_tick',values='value').reset_index().rename(columns={0:'value_t0',1:'value_t1'})
        phase['value_t1_minus_t0']=phase.value_t1-phase.value_t0
        phase.to_csv(target/'mlp_phase_comparison.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for vi,version in enumerate(VERSIONS):
        for condition in CONDITIONS:
            d=seed[(seed.version==version)&(seed.condition==condition)]
            axes[vi].scatter(d.discounted_return,d.value,c=COLORS[condition],label=LABELS[condition],s=22)
        low=min(axes[vi].get_xlim()[0],axes[vi].get_ylim()[0]);high=max(axes[vi].get_xlim()[1],axes[vi].get_ylim()[1])
        axes[vi].plot([low,high],[low,high],'k--',lw=1);axes[vi].set_title(version);axes[vi].set_xlabel('Recorded discounted episode return');axes[vi].set_ylabel('Initial critic prediction')
    axes[0].legend(fontsize=7);export(fig,target/'critic_calibration')
    text='# Actor, critic and observation-channel audit\n\nDirect input + t0 fix; actual step penalty 0.01.\n\n'+LIMITATIONS
    text+='\n\nActor probabilities and critic values are computed from frozen checkpoint heads applied to original saved post-observation GRU states. The full discounted critic target includes the recorded terminal reward and has zero terminal bootstrap. Policies are not fitted and no new environment rollouts are performed. Separate diagnostic capability decoders are fitted in the selected-policy audit.\n\n'
    text+='MLPs have no saved meaningful hidden state. Their exact round-start and next stationary observations are reconstructed from the recorded layout, own allocation and phase bit. Full-trajectory MLP critic values are not invented. The phase comparison records the current values at t0 and t1; the phase-bit-only counterfactual is saved separately in the start table.\n\n'
    text+='Zero-history probes reset the pre-observation GRU state while preserving the current observation. Allocation-channel probes vary the ego’s own previous request, not a message revealing partner capability. The factual legal mask is held fixed; initial forced-mask changes are marked explicitly. These probes do not establish causal episode-level performance effects.\n\n'
    text+='![Critic calibration](critic_calibration.png)\n\n## Equal-learner RNN summary\n\n'+table(seed_math.groupby(['version','condition'])[cols].mean().reset_index())
    text+='\n\n## Reconstruction coverage\n\n'+table(pd.DataFrame(coverage))
    report(target/'report.md',text)
    write(target/'validation.json',dict(status='passed',policies=len(policies),rnn_policies=sum(p['condition'].startswith('rnn') for p in policies),
        mlp_policies=sum(p['condition'].startswith('mlp') for p in policies),all_start_reconstructions_within_0_01=bool(probes.reconstruction_within_0_01.all()),
        maximum_hidden_reconstruction_error=float(probes.factual_hidden_max_abs_difference.max()),original_saved_hidden=True,new_rollouts=False))
    return seed_math


def case_figure(record,row,rounds):
    d=rounds[(rounds.population==row.population)&(rounds.episode_index==row.episode_index)].sort_values('round')
    fig,axes=plt.subplots(4,1,figsize=(10,7),sharex=True,layout='constrained')
    x=d['round'];axes[0].bar(x,d.round_return,color=np.where(d.timeout,'#ba4555','#5684b5'))
    axes[0].set_ylabel('Round return');axes[1].plot(x,d.steps,'o-',ms=3);axes[1].axhline(100,c='gray',ls='--');axes[1].set_ylabel('Ticks')
    axes[2].step(x,d.final_partner_faster.astype(int),where='mid');axes[2].set_yticks([0,1],['Slower','Faster']);axes[2].set_ylabel('Partner goal')
    axes[3].bar(x,d.switches,color='#9970b9');axes[3].set_ylabel('Switches');axes[3].set_xlabel('Round');axes[3].set_xticks(range(1,21))
    fig.suptitle(f"{key(record)} · {row.case_role} · {row.population} episode {row.episode_index}\nPartner ({row.delay_red},{row.delay_blue}); return {row.episode_return:.3f}; {row.timeouts} timeouts; {row.switches} switches",fontsize=11)
    return fig


def behavioral_reports(policies,episodes,diagnoses,root):
    allout=Path(root)/'lowest_decile_analysis_all_seeds';bestout=Path(root)/'lowest_decile_analysis'
    allout.mkdir(exist_ok=True);bestout.mkdir(exist_ok=True)
    selected={key(p) for p in policies if p['selected']}
    episodes=episodes.copy();episodes['policy_key']=[f'{v}_{c}_seed{s}' for v,c,s in zip(episodes.version,episodes.condition,episodes.seed)]
    chosen=episodes[episodes.policy_key.isin(selected)]
    pd.DataFrame(policies).to_csv(allout/'all_seeds.csv',index=False)
    pd.DataFrame([p for p in policies if p['selected']]).to_csv(bestout/'selected_seeds.csv',index=False)
    pd.DataFrame(policies).to_csv(bestout/'training_candidates.csv',index=False)
    write(bestout/'selection.json',dict(rule='Maximum unsmoothed final-update training return; ties choose smaller seed; no evaluation score used.',selected=[p for p in policies if p['selected']]))
    tail_rounds=[];selected_rounds=[];tail_events=[];cases=[];profiles=[];group_rows=[];profile_behavior=[];round_summary=[];associations=[];matched=[];behavior_class=[]
    (allout/'cases').mkdir(exist_ok=True);(bestout/'cases').mkdir(exist_ok=True);(allout/'seeds').mkdir(exist_ok=True)
    with PdfPages(allout/'cases/all_seed_representative_cases.pdf') as allpdf,PdfPages(bestout/'cases/selected_representative_cases.pdf') as bestpdf:
        for index,p in enumerate(policies,1):
            print('[report]',index,'/60',key(p),flush=True)
            folder=cache(root,p);rd=pd.read_csv(folder/'round_metrics.csv.gz');ep=pd.read_csv(folder/'episode_metrics.csv.gz')
            events=pd.read_csv(folder/'switch_events.csv.gz')
            tails=rd[rd.lowest_decile];tail_rounds.append(tails)
            stable=tails.switches==0
            behavior_class.append(dict(identity(p),stable_slow_fraction=float((stable&~tails.final_partner_faster).mean()),
                stable_slow_timeout_fraction=float((stable&~tails.final_partner_faster&tails.timeout).mean()),
                switched_timeout_fraction=float((~stable&tails.timeout).mean()),ten_plus_timeout_fraction=float(((tails.switches>=10)&tails.timeout).mean()),
                initial_unfinishable_fraction=float(tails.initial_assignment_unfinishable.mean()),
                successful_round_steps=float(tails.loc[~tails.timeout,'steps'].mean()),
                switch_reload_stays_per_round=float(tails.switch_reload_forced_stay_ticks.mean()),
                preempted_ready_switches_per_round=float(tails.switches_preempting_ready_clock.mean())))
            match=(diagnoses.version==p['version'])&(diagnoses.condition==p['condition'])&(diagnoses.seed==p['seed'])
            extra=dict(tail_timeout_rate=tails.timeout.mean(),rest_timeout_rate=rd.loc[~rd.lowest_decile,'timeout'].mean(),
                tail_stable_slow_rounds=int((stable&~tails.final_partner_faster).sum()),tail_stable_slow_timeouts=int((stable&~tails.final_partner_faster&tails.timeout).sum()),
                tail_switched_timeouts=int((~stable&tails.timeout).sum()),tail_10plus_switch_timeouts=int(((tails.switches>=10)&tails.timeout).sum()),
                tail_initial_assignment_unfinishable_rounds=int(tails.initial_assignment_unfinishable.sum()),
                tail_initial_partner_faster=tails.initial_partner_faster.mean(),tail_final_ego_red=(tails.final_ego_assignment==1).mean(),
                all_final_ego_red=(rd.final_ego_assignment==1).mean(),all_final_partner_faster=rd.final_partner_faster.mean(),
                tail_switch_reload_stay_ticks=tails.switch_reload_forced_stay_ticks.sum(),tail_switches_preempting_ready_clock=tails.switches_preempting_ready_clock.sum())
            for name,value in extra.items():diagnoses.loc[match,name]=value
            if p['selected']:selected_rounds.append(rd)
            if len(events):tail_events.append(events[events.lowest_decile])
            examples=pd.read_csv(folder/'representative_cases.csv');cases.append(examples)
            for row in examples.itertuples():
                fig=case_figure(p,row,rd);name=f'{key(p)}_{row.case_role}.png'
                fig.savefig(allout/'cases'/name,dpi=120);allpdf.savefig(fig)
                if p['selected']:fig.savefig(bestout/'cases'/name,dpi=120);bestpdf.savefig(fig)
                plt.close(fig)
            for (pop,dr,db),df in ep.groupby(['population','delay_red','delay_blue']):
                tail=df[df.lowest_decile]
                profiles.append(dict(identity(p),population=pop,delay_red=dr,delay_blue=db,total_episodes=len(df),tail_episodes=len(tail),
                    tail_rate=len(tail)/len(df),enrichment=10*len(tail)/len(df),all_mean_return=df.episode_return.mean(),
                    tail_mean_return=tail.episode_return.mean() if len(tail) else np.nan,seen_during_training=bool(df.seen_during_training.iloc[0])))
            for subset,df in [('all',rd),('tail',tails),('rest',rd[~rd.lowest_decile])]:
                for round_id,rr in df.groupby('round'):
                    round_summary.append(dict(identity(p),subset=subset,round=round_id,round_return=rr.round_return.mean(),timeout_rate=rr.timeout.mean(),
                        round_steps=rr.steps.mean(),switches_per_round=rr.switches.mean(),switches_per_100_movement_ticks=100*rr.switches.sum()/np.maximum(rr.steps-1,1).sum(),
                        initial_partner_faster=rr.initial_partner_faster.mean(),final_partner_faster=rr.final_partner_faster.mean(),
                        oracle_initial_return=rr.oracle_initial_return.mean(),oracle_best_return=rr.oracle_best_return.mean(),execution_gap=rr.execution_gap.mean()))
                bins=pd.cut(df.switches,[-1,0,4,9,np.inf],labels=['0','1–4','5–9','10+'])
                for label,rr in df.groupby(bins,observed=True):
                    associations.append(dict(identity(p),subset=subset,switch_bin=str(label),n_rounds=len(rr),mean_return=rr.round_return.mean(),timeout_rate=rr.timeout.mean(),mean_steps=rr.steps.mean()))
                ee=ep if subset=='all' else ep[ep.lowest_decile if subset=='tail' else ~ep.lowest_decile]
                for (pop,dr,db,g),gg in ee.groupby(['population','delay_red','delay_blue','group']):
                    profile_behavior.append(dict(identity(p),subset=subset,population=pop,delay_red=dr,delay_blue=db,group=g,n_episodes=len(gg),
                        episode_return=gg.episode_return.mean(),timeout_rate=gg.timeouts.mean()/20,switches_per_round=gg.switches.mean()/20,
                        initial_partner_faster=gg.initial_partner_faster.mean(),final_partner_faster=gg.final_partner_faster.mean(),
                        late_minus_early=gg.late_minus_early.mean(),assignment_opportunity_gap=gg.assignment_opportunity_gap.mean(),execution_gap=gg.execution_gap.mean()))
                for group,gg in ee.groupby('group'):
                    profile=gg.groupby(['population','delay_red','delay_blue'])[['episode_return','timeouts','switches','initial_partner_faster','final_partner_faster','late_minus_early','assignment_opportunity_gap','execution_gap']].mean()
                    group_rows.append(dict(identity(p),subset=subset,group=group,n_episodes=len(gg),profiles_represented=len(profile),
                        **{k:profile[k].mean() for k in profile}))
            for cell,rr in rd.groupby(['population','delay_red','delay_blue','round','initial_ego_assignment']):
                zero=rr[rr.switches==0];burst=rr[rr.switches>=10]
                if len(zero) and len(burst):matched.append(dict(identity(p),population=cell[0],delay_red=cell[1],delay_blue=cell[2],round=cell[3],initial_ego_assignment=cell[4],
                    overlap_weight=min(len(zero),len(burst)),duration_difference=burst.steps.mean()-zero.steps.mean(),
                    timeout_difference=burst.timeout.mean()-zero.timeout.mean(),return_difference=burst.round_return.mean()-zero.round_return.mean()))
            diagnosis=diagnoses[(diagnoses.version==p['version'])&(diagnoses.condition==p['condition'])&(diagnoses.seed==p['seed'])].iloc[0]
            content=f"# {key(p)}\n\nDirect input + t0 fix; step penalty 0.01. Final-update training return {p['training_return']:.3f}.\n\n"
            content+='[Checkpoint]('+p['checkpoint']+') · [Saved config]('+p['config']+') · [Matching source]('+p['source_root']+')\n\n'
            content+=table(pd.DataFrame([diagnosis]),['seed','tail_mean_return','rest_mean_return','late_minus_early','delta_ci_low','delta_ci_high','trend','tail_timeout_rounds','tail_switches_per_round'])
            content+='\n\n'+LIMITATIONS+'\n\n## Four distinct recorded tail cases\n'
            for row in examples.itertuples():content+=f'\n### {row.case_role}\n\n![Recorded round ledger](../cases/{key(p)}_{row.case_role}.png)\n\nLayouts: {row.layout_sequence}\n'
            report(allout/'seeds'/(key(p)+'.md'),content)
    tails=pd.concat(tail_rounds,ignore_index=True);events=pd.concat(tail_events,ignore_index=True) if tail_events else pd.DataFrame()
    case=pd.concat(cases,ignore_index=True);profile=pd.DataFrame(profiles);group=pd.DataFrame(group_rows);curves=pd.DataFrame(round_summary)
    curves.to_csv(allout/'seed_round_measurements.csv',index=False);profile.to_csv(allout/'seed_profile_measurements.csv',index=False)
    group.to_csv(allout/'seed_group_profile_balanced_behavior.csv',index=False)
    pd.DataFrame(profile_behavior).to_csv(allout/'seed_profile_behavior.csv',index=False)
    pd.DataFrame(behavior_class).groupby(['version','condition']).mean(numeric_only=True).drop(columns='seed').reset_index().to_csv(allout/'behavior_class_summary.csv',index=False)
    census=diagnoses.groupby(['version','condition']).agg(seeds=('seed','nunique'),tail_return_min=('tail_mean_return','min'),tail_return_max=('tail_mean_return','max'),
        increasing=('trend',lambda x:(x=='increasing').sum()),decreasing=('trend',lambda x:(x=='decreasing').sum()),any_tail_timeouts=('tail_timeout_rounds',lambda x:(x>0).sum())).reset_index()
    census.to_csv(allout/'seed_behavior_census.csv',index=False)
    decomposition=episodes[['version','condition','seed','population','episode_index','lowest_decile','group','switches','subsequent_switches','episode_steps']].copy()
    decomposition['first_free_v2_role_changes']=decomposition.switches-decomposition.subsequent_switches
    decomposition['raw_switches_per_100_movement_ticks']=100*decomposition.switches/(decomposition.episode_steps-20)
    decomposition['subsequent_switches_per_100_movement_ticks']=100*decomposition.subsequent_switches/(decomposition.episode_steps-20)
    decomposition.to_csv(allout/'episode_switch_decomposition.csv.gz',index=False)
    balanced=group.groupby(['version','condition','subset','group']).mean(numeric_only=True).drop(columns='seed').reset_index()
    counts=group.groupby(['version','condition','subset','group']).seed.nunique().rename('seeds_represented').reset_index()
    balanced=balanced.merge(counts,on=['version','condition','subset','group'],validate='one_to_one')
    balanced.to_csv(allout/'group_profile_balanced_summary.csv',index=False)
    pd.DataFrame(associations).to_csv(allout/'seed_switch_associations.csv',index=False)
    if matched:pd.DataFrame(matched).to_csv(allout/'switch_within_profile_comparisons.csv',index=False)
    for out,ep,diag,is_best in [(allout,episodes,diagnoses,False),(bestout,chosen,diagnoses[[key(x) in selected for x in diagnoses.to_dict('records')]],True)]:
        ids={key(x) for x in diag.to_dict('records')}
        is_id=lambda df:np.array([key(x) in ids for x in df[['version','condition','seed']].to_dict('records')])
        tail=ep[ep.lowest_decile].copy();rr=tails[is_id(tails)];cc=case[is_id(case)];pp=profile[is_id(profile)]
        ep.to_csv(out/('all_selected_episode_metrics.csv.gz' if is_best else 'all_episode_metrics.csv.gz'),index=False)
        tail.to_csv(out/'tail_episodes.csv',index=False);rr.to_csv(out/'tail_rounds.csv.gz',index=False)
        cc.to_csv(out/'representative_cases.csv',index=False);pp.to_csv(out/'profile_enrichment.csv',index=False)
        diag.to_csv(out/('policy_diagnoses.csv' if is_best else 'seed_diagnoses.csv'),index=False)
        write(out/('policy_diagnoses.json' if is_best else 'seed_diagnoses.json'),json.loads(diag.to_json(orient='records')))
        if len(events):events[is_id(events)].to_csv(out/'tail_switch_events.csv.gz',index=False)
        notes=tail[['version','condition','seed','population','episode_index','stable_slow_rounds','stable_slow_timeouts','switched_timeouts','ten_plus_switch_timeouts','maximum_switches_in_round','unfinishable_initial_assignments']].copy()
        notes['behavioral_note']=[f'{s} stable partner-slow rounds; {st} stable-slow timeouts; {sw} switched-round timeouts; maximum {mx} switches/round. Descriptive evidence, not causal attribution.' for s,st,sw,mx in zip(notes.stable_slow_rounds,notes.stable_slow_timeouts,notes.switched_timeouts,notes.maximum_switches_in_round)]
        notes.to_csv(out/'episode_behavior_notes.csv',index=False)
        q=curves[is_id(curves)];q.to_csv(out/'round_trajectory_summary.csv',index=False)
        assoc=pd.DataFrame(associations);assoc[is_id(assoc)].to_csv(out/'switch_associations.csv',index=False)
        if matched:
            mt=pd.DataFrame(matched);mt[is_id(mt)].to_csv(out/'switch_within_profile_comparisons.csv',index=False)
        numeric=diag.select_dtypes(include=np.number).columns.difference(['seed'])
        diag.groupby(['version','condition'])[numeric].mean().reset_index().to_csv(out/'equal_seed_summary.csv',index=False)
        index=pd.DataFrame([dict(identity(p),training_return=p['training_return'],report='[Open](seeds/'+key(p)+'.md)') for p in policies if key(p) in ids])
        if not is_best:report(out/'seed_index.md','# Every current learner seed\n\n'+table(index))
        text='# '+('Training-selected policies: lowest-return decile' if is_best else 'Lowest-return decile across all five learner seeds')+'\n\n'
        text+=f'Direct input + t0 fix · step penalty 0.01 · {len(diag)} policies · {len(ep):,} episodes · {len(tail):,} tail episodes · {len(rr):,} tail rounds.\n\n'
        text+=('One seed per protocol/condition is selected by the largest unsmoothed final-update training return, with smaller seed winning a printed-score tie. Evaluation returns never select the learner.\n\n' if is_best else 'All 60 learners are included. Summary means weight the five independent learners equally, rather than treating their episodes as independent training replicates.\n\n')
        text+='Each policy contributes exactly its lowest 92 of 920 episodes, sorted by return, population and episode index. Boundary ties and tie-inclusive changes are recorded. Trend intervals use 4,000 paired whole-episode bootstrap draws within each policy and condition on tail selection; they do not include seed-selection uncertainty.\n\n'+LIMITATIONS+'\n\n'
        text+='Geometry-aware oracle comparisons use the matching 1,096-layout corpus and recorded initial assignment. They grant hidden capability information and informed navigation/switching; they are diagnostic ceilings, not an attainable policy baseline. Assignment opportunity gaps in no-influence controls are imposed by the environment, not ego decision errors.\n\n'
        text+='## Current diagnoses\n\n'+table(diag,['version','condition','seed','training_return','tail_mean_return','rest_mean_return','late_minus_early','trend','tail_timeout_rounds','tail_switches_per_round','tail_final_partner_faster'])
        text+='\n\n[Actor, critic and observation-channel audit](math_audit/report.md)\n\n'
        if not is_best:text+='[Every seed and its four recorded cases](seed_index.md) · [240-page casebook](cases/all_seed_representative_cases.pdf)\n\n'
        else:text+='[48-page selected-policy casebook](cases/selected_representative_cases.pdf)\n\n'
        for metric,title in [('round_return','Round returns'),('timeout_rate','Round timeouts'),('round_steps','Round durations'),('switches_per_round','Effective switches'),('final_partner_faster','Final partner-faster assignment')]:
            fig,axes=plt.subplots(4,3,figsize=(14,12),sharex=True,layout='constrained')
            for ci,condition in enumerate(CONDITIONS):
                for vi,version in enumerate(VERSIONS):
                    ax=axes[ci,vi]
                    for subset,color in [('tail','#b64e65'),('rest','#3a83b6')]:
                        part=q[(q.version==version)&(q.condition==condition)&(q.subset==subset)]
                        for _,seed in part.groupby('seed'):ax.plot(seed['round'],seed[metric],c=color,alpha=.16,lw=.7)
                        mean=part.groupby('round')[metric].mean();sd=part.groupby('round')[metric].std().fillna(0)
                        ax.plot(mean.index,mean,c=color,label=subset)
                        if not is_best:ax.fill_between(mean.index,mean-sd,mean+sd,color=color,alpha=.12)
                    ax.set_title(version+' / '+LABELS[condition],fontsize=9);ax.set_xlabel('Round');ax.set_ylabel(title)
            axes[0,0].legend();fig.suptitle(title+' · direct input + t0 fix · '+('one training-selected learner' if is_best else 'equal five-seed mean ± sample SD'))
            filename=metric;export(fig,out/'plots'/filename);text+=f'\n![{title}](plots/{filename}.png)\n'
        fig,axes=plt.subplots(4,3,figsize=(14,14),layout='constrained')
        for ci,condition in enumerate(CONDITIONS):
            for vi,version in enumerate(VERSIONS):
                d=tail[(tail.version==version)&(tail.condition==condition)].sort_values(['seed','return_rank'])
                image=np.array([[float(v) for v in value.split(';')] for value in d.round_return_sequence])
                ax=axes[ci,vi];im=ax.imshow(image,aspect='auto',vmin=-1,vmax=1,cmap='RdBu');ax.set_title(version+' / '+LABELS[condition],fontsize=9)
                ax.set_xlabel('Round');ax.set_ylabel('Tail episodes: seed then return rank')
                fig.colorbar(im,ax=ax,shrink=.65)
        export(fig,out/'plots/all_tail_episode_heatmaps');text+='\n![Every tail episode](plots/all_tail_episode_heatmaps.png)\n'
        text+='\n## Profile composition and switching checks\n\nThe profile-standardized reference reweights all episodes to the tail’s capability mixture, including the tail itself. It is descriptive, not an independent control. Lost success credit plus extra tick cost exactly decomposes the rest-minus-tail return gap. High-switch versus zero-switch comparisons match profile, round and initial assignment, but still leave layout and time-at-risk confounding.\n'
        report(out/'report.md',text)
        write(out/'provenance.json',dict(cohort=BATCH,manifest=str(TRAINING_MANIFEST),manifest_sha256=sha(TRAINING_MANIFEST),step_penalty=.01,
            policies=len(diag),episodes=len(ep),tail_episodes=len(tail),tail_rounds=len(rr),tail_rule='exactly 92/920 per learner; stable return/population/index order; boundary tie sensitivity recorded',
            seed_selection='final-update training return only' if is_best else 'all five seeds equally weighted',new_rollouts=False))
        write(out/'validation.json',dict(status='passed',policies=len(diag),episodes=len(ep),tail_episodes=len(tail),tail_rounds=len(rr),
            reward_gap_decomposition_max_error=float(np.max(np.abs(diag.rest_mean_return-diag.tail_mean_return-diag.lost_success_credit-diag.extra_tick_penalty))),
            complete_twenty_round_episodes=True,oracle_bounds_checked=True,terminal_transitions_included=True))
    pd.concat(selected_rounds,ignore_index=True).to_csv(bestout/'all_selected_round_metrics.csv.gz',index=False)
    aggregate_math(policies,allout,root)
    aggregate_math([p for p in policies if p['selected']],bestout,root)


def single_partner_generalization(policies,root):
    out=Path(root)/'single_partner_generalization';out.mkdir(exist_ok=True)
    round_rows=[];episode_rows=[]
    for p in policies:
        if p['condition'] not in ('rnn_diverse_influence','rnn_single_influence'):continue
        with np.load(out/'cache'/(key(p)+'.npz'),allow_pickle=False) as f:
            groups=f['group'];pairs=f['capability']
            for group in GROUPS:
                chosen=groups==group;require(chosen.any(),'Missing actual-training/transfer group')
                unique=np.unique(pairs[chosen],axis=0)
                # Each profile has 20 repetitions, but average profiles explicitly.
                masks=[chosen & np.all(pairs==pair,axis=1) for pair in unique]
                er=np.mean([f['episode_return'][mask].mean() for mask in masks])
                value=np.mean([f['initial_value'][mask].mean() for mask in masks]);discount=np.mean([f['discounted_return'][mask].mean() for mask in masks])
                episode_rows.append(dict(identity(p),group=group,profiles=len(unique),episodes=int(chosen.sum()),episode_return=float(er),initial_value=float(value),discounted_return=float(discount),value_bias=float(value-discount)))
                for rd in range(20):
                    row=dict(identity(p),group=group,round=rd+1)
                    for field in ['round_return','round_success','round_steps','round_switches','initial_partner_faster','final_partner_faster']:
                        row[field]=float(np.mean([f[field][mask,rd].mean() for mask in masks]))
                    round_rows.append(row)
    ep=pd.DataFrame(episode_rows);rd=pd.DataFrame(round_rows)
    ep.to_csv(out/'episode_summary_per_seed.csv',index=False);rd.to_csv(out/'round_summary_per_seed.csv',index=False)
    ep.groupby(['version','condition','group']).mean(numeric_only=True).drop(columns='seed').to_csv(out/'equal_seed_summary.csv')
    for name,field,title in [('round_return_adaptation','round_return','Round return'),('switching_by_round','round_switches','Switches per round'),('assignment_adaptation','final_partner_faster','Partner receives faster goal'),('behavior_comparison','round_success','Round success')]:
        fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained',sharex=True)
        for ci,condition in enumerate(('rnn_diverse_influence','rnn_single_influence')):
            for vi,version in enumerate(VERSIONS):
                ax=axes[ci,vi]
                for group,color in zip(GROUPS,['#31865b','#d18b2a','#5d6fc0']):
                    d=rd[(rd.version==version)&(rd.condition==condition)&(rd.group==group)]
                    mean=d.groupby('round')[field].mean();sd=d.groupby('round')[field].std()
                    ax.plot(mean.index,mean,color=color,label=GROUP_LABELS[group]);ax.fill_between(mean.index,mean-sd,mean+sd,color=color,alpha=.12)
                ax.set_title(version+' / '+LABELS[condition]);ax.set_xlabel('Round');ax.set_ylabel(title)
        axes[0,0].legend(fontsize=7);fig.suptitle('Direct input + t0 fix · equal-profile then equal-five-seed means ± sample SD')
        export(fig,out/name)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for vi,version in enumerate(VERSIONS):
        for condition in ('rnn_diverse_influence','rnn_single_influence'):
            d=ep[(ep.version==version)&(ep.condition==condition)];axes[vi].scatter(d.discounted_return,d.initial_value,label=LABELS[condition],c=COLORS[condition])
        low=min(axes[vi].get_xlim()[0],axes[vi].get_ylim()[0]);high=max(axes[vi].get_xlim()[1],axes[vi].get_ylim()[1])
        axes[vi].plot([low,high],[low,high],'k--');axes[vi].set_title(version);axes[vi].set_xlabel('Discounted recorded episode return');axes[vi].set_ylabel('Initial critic prediction')
    axes[0].legend(fontsize=7);export(fig,out/'critic_calibration')
    text='# Single-partner generalization\n\nCurrent direct-input/t0-fixed policies; step penalty 0.01. Compare all five single-partner RNN learners with all five diverse-partner RNN learners in each protocol: 30 policies.\n\n'
    text+='The groups are the actual single training partner (1,4), the other 23 reference profiles, and 22 held-out profiles. Only the first group was familiar during single-partner training; all 24 reference profiles were familiar to diverse-partner learners. Average repetitions within profile, then profiles within group, then the five learner seeds. Bands are sample SD across learners, not episode-level uncertainty.\n\n'+LIMITATIONS+'\n\n'
    for name in ['behavior_comparison','assignment_adaptation','round_return_adaptation','switching_by_round','critic_calibration']:text+=f'\n![{name}]({name}.png)\n'
    text+='\n## Episode and critic summaries\n\n'+table(ep.groupby(['version','condition','group'])[['episode_return','initial_value','discounted_return','value_bias']].mean().reset_index())
    report(out/'report.md',text)
    write(out/'validation.json',dict(status='passed',policies=30,seeds=list(SEEDS),episodes=27600,
        groups=['one actual training partner','23 other reference profiles','22 held-out profiles'],
        critic_input='original saved post-observation GRU state at episode start',new_rollouts=False))


def build_reports(root=BUILD):
    root=Path(root);policies=json.loads((root/'run.json').read_text())['policies']
    episodes=[];diagnoses=[]
    for p in policies:
        require((cache(root,p)/'completion.json').exists() and (cache(root,p)/'math/completion.json').exists(),'Incomplete policy analysis: '+key(p))
        episodes.append(pd.read_csv(cache(root,p)/'episode_metrics.csv.gz'))
        diagnoses.append(json.loads((cache(root,p)/'diagnosis.json').read_text()))
    episodes=pd.concat(episodes,ignore_index=True);diagnoses=pd.DataFrame(diagnoses)
    require(len(episodes)==55200 and episodes.lowest_decile.sum()==5520,'Wrong complete/tail episode coverage')
    return_distributions(episodes,policies,root)
    behavioral_reports(policies,episodes,diagnoses,root)
    single_partner_generalization(policies,root)
    return policies
