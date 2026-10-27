"""Plot sampled divergence brackets and partner-specific failure profiles."""
import argparse
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir',type=Path,required=True)
    args=p.parse_args()
    import numpy as np,pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=args.analysis_dir
    t=pd.read_csv(out/'checkpoint_timeline.csv')
    windows=json.loads((out/'behavior_windows.json').read_text())['comparisons']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(4,2,figsize=(14,12))
    for k,w in enumerate(windows):
        left,right=w['left'],w['right']; event=w['lasting_heldout_return_gap_at_least_one']
        title=' vs '.join([t[t['index']==i].label.iloc[0] for i in (left,right)])
        for col in range(2):
            ax=axes[k,col]
            for index,color in ((left,'#1764ab'),(right,'#c73539')):
                d=t[(t['index']==index)&(t.population=='heldout')].sort_values('update')
                s=d[d.repetitions==4]
                ax.plot(s.transitions/1e6,s.return_,color=color,label=d.label.iloc[0].split(' ')[-1],lw=1.5)
                c=d[d.repetitions==20]
                ax.scatter(c.transitions/1e6,c.return_,color=color,marker='*',s=35)
            if event:
                lo=event['previous_transitions']/1e6;hi=event['first_observed_transitions']/1e6
                ax.axvspan(lo,hi,color='grey',alpha=.2)
                ax.axvline(hi,color='black',ls='--',lw=1)
                ax.set_title(title+f"\nLasting sampled gap ≥1 from checkpoint {event['first_observed_checkpoint']} ({hi:.3f}M)",fontsize=10)
                if col:
                    margin=max(1,(hi-lo)*8)
                    ax.set_xlim(max(0,hi-margin),min(60,hi+margin))
            ax.grid(alpha=.2);ax.set_ylabel('Held-out episode return');ax.set_xlabel('Training transitions (millions)');ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Paired seed comparisons — shaded interval: last checkpoint below threshold to first lasting observed gap\n4-packet screens only; stars are separate 20-packet confirmations. Markers are descriptive, not causal update claims.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'paired_gap_windows.png',dpi=180);fig.savefig(out/'paired_gap_windows.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for index,color in ((10,'#1764ab'),(6,'#c73539')):
        for pop,metric,ax,ylabel in [('training','return_',axes[0,0],'Training-profile episode return'),
             ('heldout','return_',axes[0,1],'Held-out episode return'),('heldout','success',axes[1,0],'Held-out round success (%)'),
             ('heldout','switches',axes[1,1],'Held-out switches / round')]:
            d=t[(t['index']==index)&(t.population==pop)&t.repetitions.eq(4)].sort_values('update')
            d=d[d['update']>=96];factor=100 if metric=='success' else 1
            ax.plot(d.transitions/1e6,d[metric]*factor,color=color,label=d.label.iloc[0].split(' ')[-1],lw=1.5)
            ax.axvspan(144*65536/1e6,160*65536/1e6,color='grey',alpha=.2)
            ax.set_ylabel(ylabel);ax.grid(alpha=.2)
    axes[0,0].legend();axes[0,1].legend();axes[0,0].set_ylim(16.8,17.6)
    for ax in axes[-1]:ax.set_xlabel('Training transitions (millions)')
    fig.suptitle('V3 RNN-single seeds 6 and 7 — training stays strong while transfer fluctuates\nShading: checkpoint 144→160, the first large later drop for seed 7 (9.44→10.49M transitions).',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(out/'late_transfer_decay.png',dpi=180);fig.savefig(out/'late_transfer_decay.pdf');plt.close(fig)
    profiles=pd.read_csv(out/'final_profile_behaviors.csv')
    indices=[2,3,7,4,5,11,9,10,6]
    meta=profiles[profiles['index']==2].sort_values(['partner_red_faster','d_R','d_B'],ascending=[False,True,True])
    order=meta.profile.tolist()
    grid=profiles.pivot(index='profile',columns='index',values='success').loc[order,indices]*100
    fig,ax=plt.subplots(figsize=(13,9))
    im=ax.imshow(grid,aspect='auto',vmin=0,vmax=100,cmap='viridis')
    ax.set_xticks(range(len(indices)),[t[t['index']==i].label.iloc[0] for i in indices],rotation=35,ha='right')
    ax.set_yticks(range(len(order)),[f'({int(x.d_R)}, {int(x.d_B)})' for _,x in meta.iterrows()])
    ax.set_ylabel('Partner delay (red, blue) — lower is faster')
    for y in range(len(order)):
        for x in range(len(indices)):
            value=grid.iloc[y,x];ax.text(x,y,f'{value:.0f}',ha='center',va='center',fontsize=8,color='white' if value<65 else 'black')
    ax.axhline(10.5,color='white',lw=2)
    fig.colorbar(im,ax=ax,label='Held-out round success (%)')
    ax.set_title('Failures depend strongly on partner profile\nFinal checkpoint, identical 20 grid packets for every profile and policy')
    fig.tight_layout();fig.savefig(out/'partner_profile_success.png',dpi=180);fig.savefig(out/'partner_profile_success.pdf');plt.close(fig)

if __name__=='__main__':main()
