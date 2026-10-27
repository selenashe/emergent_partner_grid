"""Audit completed RNN diagnostic timelines and paired interventions.

Whole 20-round episodes are the unit of treatment. Confidence intervals resample
shared grid packets jointly across profiles; no round-level independent-sample
inference or across-training-seed significance claim is made.
"""
import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.learning_diagnostics import targets,target_name,sha,write_json
from repo_paths import prepare_artifact_path,link_artifact


def label(row):
    return f"{row['version']} {'RNN-single' if 'single' in row['condition'] else 'RNN-multi'} s{row['seed']}"


def ledger(folder,u,population,mode):
    import pandas as pd
    p=folder/'evaluation'/f'update{u:04d}_{population}_{mode}.rounds.csv.gz'
    a=pd.read_csv(p)
    assert len(a)%20==0 and a.groupby('episode').size().eq(20).all()
    assert a['success'].dtype==bool and a['final_ego_red'].dtype==bool
    assert a['round'].between(0,19).all()
    assert len(a[['episode','round']].drop_duplicates())==len(a)
    assert a.groupby('profile')['repetition'].nunique().nunique()==1
    assert abs(a.return_.sum()/a.episode.nunique()-(a.success.sum()*1.01-a.steps.sum()*.01)/a.episode.nunique())<3e-5
    return a,p


def aggregate(a):
    import numpy as np
    b=a.groupby('episode').agg(return_=('return_','sum'),success=('success','mean'),steps=('steps','sum'),
        red=('final_ego_red','mean'),matching=('partner_matching','mean'),switches=('switches','mean'),
        profile=('profile','first'),packet=('repetition','first'))
    switched=a.switches>0
    return dict(episodes=len(b),repetitions=int(a.repetition.nunique()),
        return_=float(b.return_.mean()),success=float(a.success.mean()),red=float(a.final_ego_red.mean()),
        matching=float(a.partner_matching.mean()),switches=float(a.switches.mean()),
        switched=float(switched.mean()),tenplus=float((a.switches>=10).mean()),
        mean_steps=float(a.steps.mean()),matching_round1=float(a.loc[a['round']==0,'partner_matching'].mean()),
        matching_late5=float(a.loc[a['round']>=15,'partner_matching'].mean()),
        matching_round20=float(a.loc[a['round']==19,'partner_matching'].mean()),
        success_switched=float(a.loc[switched,'success'].mean()) if switched.any() else None,
        success_unswitched=float(a.loc[~switched,'success'].mean()) if (~switched).any() else None,
        switches_given_switched=float(a.loc[switched,'switches'].mean()) if switched.any() else 0),b


def bootstrap_packet(d,seed=20261018):
    import numpy as np
    rng=np.random.default_rng(seed)
    values=np.asarray(d,float)
    samples=values[rng.integers(0,len(values),size=(5000,len(values)))].mean(axis=1)
    return [float(np.quantile(samples,.025)),float(np.quantile(samples,.975))]


def save_compressed_csv(frame,path):
    frame.to_csv(prepare_artifact_path(path),index=False,compression='gzip')
    link_artifact(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args()
    import numpy as np
    import pandas as pd
    out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    rows=targets()
    available=[];audit=[];timeline=[];endpoints=[];interventions=[];round_curves=[];profile_curves=[];episodes=[]
    candidate_changes=[]
    max_return_accumulation_difference=0.0
    for i,row in enumerate(rows):
        if not row['condition'].startswith('rnn_'):continue
        folder=args.run_dir/'runs'/target_name(row)
        if not (folder/'evaluation_status.json').exists():continue
        assert json.loads((folder/'status.json').read_text())['state']=='complete'
        assert json.loads((folder/'evaluation_status.json').read_text())['state']=='complete'
        fidelity=json.loads((folder/'fidelity.json').read_text())
        assert fidelity['passed'] and fidelity['checked_updates']==915
        assert json.loads((folder/'final_parameter_comparison.json').read_text())['tensors_equal']
        available.append(i)
        entries=json.loads((folder/'evaluation_summary.json').read_text())
        originals={(a['update'],a['population'],a['intervention']):a for a in entries}
        audit.append(dict(index=i,target=target_name(row),scalar_checks=fidelity['scalar_checks'],
            source_summary_sha256=sha(folder/'evaluation_summary.json'),
            artifacts_manifest_sha256=sha(folder/'evaluation_artifacts.json')))
        baseline_cache={}
        for entry in sorted(entries,key=lambda a:(a['update'],a['population'],a['intervention'])):
            u,pop,mode=entry['update'],entry['population'],entry['intervention']
            if mode!='baseline' and u!=915:continue
            a,path=ledger(folder,u,pop,mode)
            stats,b=aggregate(a)
            for metric,key in [('return_','return_'),('success','success'),('red','final_ego_red'),('matching','partner_matching'),('switches','switches_per_round')]:
                difference=abs(stats[metric]-entry[key])
                # Original summary sums up to 2000 float32 rewards in order;
                # round ledgers sum twenty much shorter float32 reductions.
                tolerance=1e-3 if metric=='return_' else 3e-12
                assert difference<tolerance,(i,u,pop,mode,metric,difference)
                if metric=='return_': max_return_accumulation_difference=max(max_return_accumulation_difference,difference)
            metadata=dict(index=i,label=label(row),version=row['version'],condition=row['condition'],seed=int(row['seed']),
                update=u,transitions=u*65536,population=pop,intervention=mode)
            if mode=='baseline':
                timeline.append(dict(**metadata,**stats))
                baseline_cache[(u,pop)]=(a,b)
                for r,g in a.groupby('round'):
                    round_curves.append(dict(**metadata,round=int(r)+1,success=float(g.success.mean()),matching=float(g.partner_matching.mean()),
                        red=float(g.final_ego_red.mean()),switches=float(g.switches.mean()),steps=float(g.steps.mean())))
                # Final and a small set of early/mid checkpoints expose profile dependence.
                if u in (0,12,16,20,32,64,96,128,144,160,176,192,224,256,336,512,768,915):
                    for profile,g in a.groupby('profile'):
                        profile_curves.append(dict(**metadata,profile=int(profile),d_R=int(g.d_R.iloc[0]),d_B=int(g.d_B.iloc[0]),
                            success=float(g.success.mean()),return_=float(g.return_.sum()/g.episode.nunique()),
                            matching=float(g.partner_matching.mean()),red=float(g.final_ego_red.mean()),switches=float(g.switches.mean())))
            if u==915:
                endpoints.append(dict(**metadata,**stats))
                be=b.reset_index();be['index']=i;be['population']=pop;be['intervention']=mode;episodes.append(be)
                if mode!='baseline':
                    base_a,base_b=baseline_cache[(u,pop)]
                    assert np.array_equal(a[['episode','round','profile','repetition','layout_index']].values,
                                          base_a[['episode','round','profile','repetition','layout_index']].values)
                    assert np.array_equal(b.index,base_b.index)
                    delta=b[['return_','success','steps','switches']]-base_b[['return_','success','steps','switches']]
                    delta['packet']=b['packet']
                    packet=delta.groupby('packet').mean()
                    record=dict(**metadata,episodes=len(b),packets=len(packet))
                    for metric in ('return_','success','steps','switches'):
                        record['delta_'+metric]=float(delta[metric].mean())
                        record[metric+'_ci_low'],record[metric+'_ci_high']=bootstrap_packet(packet[metric].values)
                    interventions.append(record)
        print('[analyzed]',i,label(row),flush=True)
    t=pd.DataFrame(timeline).sort_values(['index','population','update'])
    e=pd.DataFrame(endpoints);ci=pd.DataFrame(interventions);rc=pd.DataFrame(round_curves);pc=pd.DataFrame(profile_curves)
    t.to_csv(out/'checkpoint_timeline.csv',index=False)
    e.to_csv(out/'endpoint_behaviors.csv',index=False);ci.to_csv(out/'paired_interventions.csv',index=False)
    save_compressed_csv(rc,out/'matching_by_round.csv.gz');save_compressed_csv(pc,out/'profile_timeline.csv.gz')
    save_compressed_csv(pd.concat(episodes),out/'final_episode_metrics.csv.gz')
    # Descriptive threshold intervals. Screen only checkpoints evaluated with
    # four packets; confirmation batches use another RNG array shape.
    rules={
        'matching_at_least_75pct':lambda d:d.matching>=.75,
        'matching_at_least_90pct':lambda d:d.matching>=.90,
        'success_at_least_90pct':lambda d:d.success>=.90,
        'red_at_least_95pct':lambda d:d.red>=.95,
        'blue_at_least_95pct':lambda d:d.red<=.05,
        'switches_at_most_3':lambda d:d.switches<=3,
    }
    for (index,pop),d in t.groupby(['index','population']):
        d=d[d.repetitions==4].sort_values('update').reset_index(drop=True)
        for name,rule in rules.items():
            mask=np.asarray(rule(d))
            runs=[]
            for k in range(len(d)-2):
                if mask[k:k+3].all() and (k==0 or not mask[k-1]):
                    j=k+3
                    while j<len(d) and mask[j]:j+=1
                    runs.append(dict(first=int(d.iloc[k]['update']),last=int(d.iloc[j-1]['update']),
                        previous=int(d.iloc[k-1]['update']) if k else 0,checkpoints=j-k))
            if runs:
                candidate_changes.append(dict(index=int(index),population=pop,rule=name,first_three_checkpoint_run=runs[0],
                    run_reaching_last_screen=next((r for r in reversed(runs) if r['last']==int(d.iloc[-1]['update'])),None)))
    comparisons=[]
    for left,right in ((2,3),(2,7),(11,9),(10,6)):
        a=t[(t['index']==left)&(t.population=='heldout')&t.repetitions.eq(4)].set_index('update')
        b=t[(t['index']==right)&(t.population=='heldout')&t.repetitions.eq(4)].set_index('update')
        shared=a.index.intersection(b.index)
        gap=a.loc[shared,'return_']-b.loc[shared,'return_']
        mask=gap>=1
        k=len(mask)-1
        while k>=0 and mask.iloc[k]:k-=1
        lasting=None
        if len(mask)-k-1>=3:
            onset=int(shared[k+1]);prev=int(shared[k]) if k>=0 else 0
            lasting=dict(previous_checkpoint=prev,first_observed_checkpoint=onset,
                         previous_transitions=prev*65536,first_observed_transitions=onset*65536)
        comparisons.append(dict(left=left,right=right,lasting_heldout_return_gap_at_least_one=lasting))
    write_json(out/'behavior_windows.json',dict(threshold_definition='Three consecutive four-packet checkpoints; descriptive screen, not significance test.',
        rules=candidate_changes,comparisons=comparisons))
    write_json(out/'audit.json',dict(targets=available,runs=audit,analysis_script_sha256=sha(__file__),
        panel_sha256=sha(args.run_dir/'evaluation_panel.json'),all_final_parameters_exact=True,
        summary_vs_round_ledger_checks=True,max_return_accumulation_difference=max_return_accumulation_difference,
        return_check_tolerance=1e-3,other_metric_check_tolerance=3e-12,bootstrap='5000 resamples of grid packets jointly across all profiles; fixed training seed and profile population',
        caveats=['Four-packet screening, twenty-packet confirmations at update 0,915 and nominated onsets use different RNG batch shapes.',
                 'Shared packets across partner profiles require joint packet resampling, not independent round inference.',
                 'Intervals localize checkpoint changes; they do not identify a unique causal optimizer step.',
                 'Controls selected retrospectively by observed phenotype; estimates do not establish effects across random training seeds.']))
    figures(t,e,ci,out)
    print('OUTPUT',out,flush=True)


def figures(t,e,ci,out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    groups=[('V1_matched: navigation separates before adaptation',[2,3,7]),
            ('V3 multi: efficient red commitment vs poor completion',[11,9]),
            ('V3 single: equal training endpoints, different transfer',[10,6])]
    colors={2:'#1764ab',3:'#ec8a15',7:'#8048b5',11:'#1764ab',9:'#c73539',10:'#1764ab',6:'#c73539',4:'#1764ab',5:'#c73539'}
    for zoom,name in ((False,'focal_timelines'),(True,'focal_first_10M')):
        fig,axes=plt.subplots(3,3,figsize=(15,10),sharex='col')
        for r,(title,indices) in enumerate(groups):
            for index in indices:
                d=t[(t['index']==index)&(t.population=='heldout')].sort_values('update')
                for j,(metric,ylabel) in enumerate((('return_','Held-out episode return'),('success','Round success (%)'),('matching','Partner matching (%)'))):
                    ax=axes[r,j];factor=1 if metric=='return_' else 100
                    screen=d[d.repetitions==4]
                    ax.plot(screen.transitions/1e6,screen[metric]*factor,color=colors[index],label=d.label.iloc[0].split(' ')[-1],lw=1.6)
                    confirm=d[d.repetitions==20]
                    ax.scatter(confirm.transitions/1e6,confirm[metric]*factor,color=colors[index],marker='*',s=48,zorder=3)
                    ax.set_ylabel(ylabel);ax.grid(alpha=.18)
                    if zoom:ax.set_xlim(0,10)
                axes[r,0].set_title(title,loc='left',fontsize=11)
            axes[r,0].legend(loc='lower right',ncol=3,fontsize=8)
            axes[r,2].axhline(50,color='grey',ls=':',lw=1)
            if r==2:axes[r,0].set_ylim(-14,18)
        for ax in axes[-1]:ax.set_xlabel('Training transitions (millions)')
        fig.suptitle('Completed policies on fixed grid packets — line: 4 packets/profile; star: 20-packet confirmation',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.965));fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'));plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(15,10))
    for r,index in enumerate((4,5,6)):
        for pop,style in (('heldout','-'),('training','--')):
            d=t[(t['index']==index)&(t.population==pop)].sort_values('update')
            for j,(metric,ylabel) in enumerate((('return_','Episode return'),('switches','Switches / round'),('success','Round success (%)'))):
                factor=100 if metric=='success' else 1
                screen=d[d.repetitions==4]
                axes[r,j].plot(screen.transitions/1e6,screen[metric]*factor,style,label=pop,lw=1.6)
                final=d[d['update']==915]
                axes[r,j].scatter(final.transitions/1e6,final[metric]*factor,s=45,marker='*')
                axes[r,j].set_ylabel(ylabel);axes[r,j].grid(alpha=.18);axes[r,j].set_xlabel('Training transitions (millions)')
        axes[r,0].set_title(t[t['index']==index].label.iloc[0],loc='left');axes[r,0].legend()
    fig.suptitle('Training-profile vs held-out behavior — separate fixed evaluation populations',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.965));fig.savefig(out/'transfer_timelines.png',dpi=180);fig.savefig(out/'transfer_timelines.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,7),sharey=True)
    # Holding after the first request is the same intervention across online
    # protocols; its effect is not presumed to have the same sign.
    online=[4,5,9,11,6,10]
    for j,(metric,ylabel) in enumerate((('return_','Change in held-out episode return'),('success','Change in success (percentage points)'))):
        for k,index in enumerate(online):
            row=ci[(ci['index']==index)&(ci.population=='heldout')&(ci.intervention=='hold_after_first_request')].iloc[0]
            factor=100 if metric=='success' else 1
            mean=row['delta_'+metric]*factor;lo=row[metric+'_ci_low']*factor;hi=row[metric+'_ci_high']*factor
            axes[j].errorbar(mean,k,xerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o',capsize=4,color=colors[index])
        axes[j].axvline(0,color='grey',ls='--');axes[j].grid(axis='x',alpha=.2);axes[j].set_xlabel(ylabel)
    axes[0].set_yticks(range(len(online)),[t[t['index']==i].label.iloc[0] for i in online]);axes[0].invert_yaxis()
    fig.suptitle('Allow the first online assignment request, then hold — paired full-episode interventions\n95% interval: bootstrap 20 shared grid packets, keeping all partner profiles together',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(out/'switch_intervention_effects.png',dpi=180);fig.savefig(out/'switch_intervention_effects.pdf');plt.close(fig)

if __name__=='__main__':main()
