"""Raw-state geometry, competing models, context transfer and temporal persistence."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from .numeric import Ridge, select_ridge, r2, grouped_ci, cross_distance
from .data import write_json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PHASE_NAMES=('start','midpoint','end')

def savefig(fig,out,name):
    fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=150);fig.savefig(out/(name+'.pdf'));plt.close(fig)

def analyze(hidden,index,out,config):
    # Audit guide:
    # Fit PCA and ridge predictors on discovery data, choose penalties on validation
    # episodes, and score test episodes. Compare capability explanations with
    # phase/layout nuisance predictors, transfer across time/populations, and compute
    # cross-validated profile distances. Episode grouping governs uncertainty; pairwise
    # profile distances share profiles and are not independent samples. These geometry
    # measures describe recoverable structure rather than causal use by the policy.
    #
    train=(index.split=='train').to_numpy();val=(index.split=='validation').to_numpy();test=(index.split=='test').to_numpy()
    familiar=(index.capability_slice=='train').to_numpy()
    # Each episode contributes exactly 60 raw samples (20 rounds x 3 phases).
    mean=hidden[train].mean(0)
    covariance=np.cov((hidden[train]-mean).T)
    eig,vectors=np.linalg.eigh(covariance);order=np.argsort(eig)[::-1];eig=eig[order];vectors=vectors[:,order]
    k=min(config['pca_rank'],hidden.shape[1]);basis=vectors[:,:k];z=(hidden-mean)@basis
    np.savez_compressed(out/'projections.npz',mean=mean,basis=basis,eigenvalues=eig)
    pca={}
    for name,mask in [('within_profile_test',test),('familiar_test',test&familiar),('novel_test',test&~familiar)]:
        residual=(hidden[mask]-mean)-z[mask]@basis.T
        pca[name]=dict(explained_fraction=float(1-np.sum(residual**2)/np.sum((hidden[mask]-mean)**2)),
            reconstruction_error=grouped_ci(np.mean(residual**2,1),index.episode[mask],config['analysis_seed'],config['bootstrap']))
    cap=index[['d_red','d_blue']].to_numpy(float)
    orientation=(cap[:,0]<cap[:,1]).astype(float)
    cap_features=np.c_[orientation,cap,cap[:,0]-cap[:,1],np.abs(cap[:,0]-cap[:,1]),cap.mean(1)]
    # Redundant columns are explicit competing predictors; unique attribution is not claimed.
    layout_cols=['ego_to_red','ego_to_blue','partner_to_red','partner_to_blue','ego_row','ego_col','partner_row','partner_col']
    nuisance=np.c_[index[layout_cols].to_numpy(float),index['round'].to_numpy()/19,np.eye(3)[index.phase],index.is_t0]
    variance=[]
    for name,x in [('phase_layout',nuisance),('binary',np.c_[nuisance,orientation]),('capability',np.c_[nuisance,cap_features]),
                   ('interactions',np.c_[nuisance,cap_features,nuisance*orientation[:,None]])]:
        model,alpha=select_ridge(x,z,train,val)
        pred=model.predict(x[test])
        variance.append(dict(model=name,test_r2=r2(z[test],pred),alpha=alpha,**grouped_ci(np.mean((z[test]-pred)**2,1),index.episode[test],config['analysis_seed'],config['bootstrap'])))
    pd.DataFrame(variance).to_csv(out/'variance_partition.csv',index=False)
    # Train analysis regressors exclusively on familiar profiles; held-out scalars use regression.
    primary=(index['round']==19)&(index.phase==0)
    decoder=[];predictions=[]
    for scope in ('within_profile','familiar_to_novel','target_layout'):
        if scope=='within_profile':tr=train&primary;va=val&primary;te=test&primary
        elif scope=='familiar_to_novel':tr=train&familiar&primary;va=val&familiar&primary;te=test&~familiar&primary
        else:
            tr=train&primary&(index.layout_split=='train');va=val&primary&(index.layout_split=='train');te=test&primary&(index.layout_split=='test')
        tr,va,te=map(np.asarray,(tr,va,te))
        if not (tr.any() and va.any() and te.any()):continue
        model,alpha=select_ridge(hidden,cap,tr,va);prediction=model.predict(hidden[te])
        baseline=np.asarray([cap[tr & (orientation==o)].mean(0) for o in orientation[te]])
        # Conditional faster-goal population prior is a serious competing explanation.
        errors=np.abs(prediction-cap[te]).mean(1);prior=np.abs(baseline-cap[te]).mean(1)
        rng=np.random.default_rng(config['analysis_seed']);shuffled=cap.copy()
        # Primary has one row per episode; episode-level permutation preserves profile pairs.
        shuffled[tr]=shuffled[np.flatnonzero(tr)[rng.permutation(tr.sum())]]
        null=Ridge().fit(hidden[tr],shuffled[tr],alpha).predict(hidden[te])
        decoder.append(dict(scope=scope,alpha=alpha,r2=r2(cap[te],prediction),mae=float(errors.mean()),
            conditional_faster_goal_prior_mae=float(prior.mean()),improvement_over_prior=float((prior-errors).mean()),
            permutation_mae=float(np.abs(null-cap[te]).mean()),n_train=int(tr.sum()),n_test=int(te.sum()),
            uncertainty=grouped_ci(prior-errors,index.episode[te],config['analysis_seed'],config['bootstrap'])))
        for idx,p,b in zip(np.flatnonzero(te),prediction,baseline):
            predictions.append(dict(scope=scope,episode=index.episode.iloc[idx],d_red=cap[idx,0],d_blue=cap[idx,1],pred_red=p[0],pred_blue=p[1],prior_red=b[0],prior_blue=b[1]))
    pd.DataFrame(predictions).to_csv(out/'capability_predictions.csv',index=False)
    write_json(out/'decoder_transfer.json',decoder)
    # One raw sample per phase and episode; each temporal row is independently fitted on discovery episodes.
    selected_rounds=config['temporal_rounds'];temporal_masks=[np.asarray((index['round']==r)&(index.phase==p)) for r in selected_rounds for p in (0,2)]
    temporal=np.zeros((len(temporal_masks),len(temporal_masks)))
    for a,mask in enumerate(temporal_masks):
        model,_=select_ridge(hidden,orientation,train&mask,val&mask)
        for b,other in enumerate(temporal_masks):
            temporal[a,b]=np.mean((model.predict(hidden[test&other])>.5)==orientation[test&other])
    labels=[f'r{r+1} {PHASE_NAMES[p]}' for r in selected_rounds for p in (0,2)]
    pd.DataFrame(temporal,index=labels,columns=labels).to_csv(out/'cross_temporal_accuracy.csv')
    fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(temporal,vmin=.5,vmax=1,cmap='viridis');ax.set(xticks=range(len(labels)),xticklabels=labels,yticks=range(len(labels)),yticklabels=labels,xlabel='Test phase (disjoint episodes)',ylabel='Training phase');plt.setp(ax.get_xticklabels(),rotation=70);fig.colorbar(im,ax=ax,label='Faster-goal accuracy');savefig(fig,out,'cross_temporal_transfer')
    # Cross-validated diagonal-noise-normalized distances. Noise precision is estimated only on training repetitions.
    late=np.asarray(primary);profiles=np.unique(cap,axis=0);precision=None
    means={};cov_res=[]
    for p in profiles:
        mask=late&train&np.all(cap==p,1);mu=hidden[mask].mean(0);cov_res.extend(hidden[mask]-mu)
    noise=np.var(cov_res,axis=0);precision=1/np.maximum(.8*noise+.2*np.median(noise),1e-6)
    for split in ('train','test'):
        a=[];b=[]
        for p in profiles:
            ids=np.flatnonzero(late&(index.split==split).to_numpy()&np.all(cap==p,1));ids=ids[np.argsort(index.repetition.iloc[ids])]
            a.append(hidden[ids[::2]].mean(0));b.append(hidden[ids[1::2]].mean(0))
        means[split]=(np.asarray(a),np.asarray(b))
    rdm_train=cross_distance(*means['train'],precision);rdm=cross_distance(*means['test'],precision)
    euclidean=cdist((means['test'][0]+means['test'][1])/2,(means['test'][0]+means['test'][1])/2)
    correlation=cdist((means['test'][0]+means['test'][1])/2,(means['test'][0]+means['test'][1])/2,metric='correlation')
    np.savez_compressed(out/'rdms.npz',profiles=profiles,crossvalidated=rdm,train=rdm_train,euclidean=euclidean,correlation=correlation,precision=precision)
    pr=(profiles[:,0]<profiles[:,1]).astype(float);relative=(profiles[:,0]-profiles[:,1])[:,None];overall=profiles.mean(1)[:,None]
    model_rdms={'binary':cdist(pr[:,None],pr[:,None])**2,'full_profile':cdist(profiles,profiles)**2,'relative':cdist(relative,relative)**2,'magnitude':cdist(np.abs(relative),np.abs(relative))**2,'overall':cdist(overall,overall)**2}
    triangle=np.triu_indices(len(profiles),1);rsa=[]
    for name,matrix in model_rdms.items():
        x=matrix[triangle][:,None];model=Ridge().fit(x,rdm_train[triangle],1.)
        rsa.append(dict(model=name,heldout_episode_distance_r2=r2(rdm[triangle],model.predict(x)),
            correlation=float(np.corrcoef(matrix[triangle],rdm[triangle])[0,1])))
    pd.DataFrame(rsa).to_csv(out/'representational_models.csv',index=False)
    same=pr[:,None]==pr[None,:];within=rdm[triangle][same[triangle]]
    # Familiar PCA only, without novel-profile leakage.
    fmean=hidden[train&familiar].mean(0);ev,fb=np.linalg.eigh(np.cov((hidden[train&familiar]-fmean).T));fb=fb[:,np.argsort(ev)[::-1][:k]]
    fz=(hidden-fmean)@fb;fm=primary&train&familiar
    novel_rows=[]
    for sl in ('train','test'):
        mask=np.asarray(primary&test&(index.capability_slice==sl));xz=fz[mask]
        distances=cdist(xz,fz[fm]);residual=(hidden[mask]-fmean)-xz@fb.T
        novel_rows.append(dict(population='familiar' if sl=='train' else 'novel',n=int(mask.sum()),nearest_familiar_distance=float(distances.min(1).mean()),
            familiar_subspace_residual=float(np.mean(np.sum(residual**2,1))),outside_familiar_coordinate_box_fraction=float(np.any((xz<fz[fm].min(0))|(xz>fz[fm].max(0)),1).mean())))
    pd.DataFrame(novel_rows).to_csv(out/'novel_geometry.csv',index=False)
    trajectories=[]
    for r in range(20):
        for p in range(3):
            for o in (0,1):
                mask=test&(index['round']==r)&(index.phase==p)&(orientation==o)
                mu=z[mask].mean(0);se=z[mask].std(0)/np.sqrt(mask.sum())
                trajectories.append(dict(round=r,phase=p,faster_red=o,n=int(mask.sum()),pc1=mu[0],pc2=mu[1],pc1_se=se[0],pc2_se=se[1]))
    pd.DataFrame(trajectories).to_csv(out/'event_trajectories.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    frame=pd.DataFrame(trajectories)
    for o,color in [(0,'#4275b5'),(1,'#c14b4b')]:
        a=frame[(frame.faster_red==o)&(frame.phase==0)];axes[0].plot(a.pc1,a.pc2,'o-',color=color,label=f'Faster {"red" if o else "blue"}')
        axes[1].errorbar(a['round']+1,a.pc1,yerr=1.96*a.pc1_se,color=color)
    axes[0].set(xlabel='PC1',ylabel='PC2',title='Raw round-start states');axes[0].legend(fontsize=8);axes[1].set(xlabel='Round',ylabel='PC1',title='Held-out episode means')
    axes[2].bar([r['model'] for r in rsa],[r['heldout_episode_distance_r2'] for r in rsa]);axes[2].tick_params(axis='x',rotation=60);axes[2].set(ylabel='Test distance R²',title='Competing representational models');savefig(fig,out,'geometry')
    fig,ax=plt.subplots(figsize=(7,4));ax.bar([r['model'] for r in variance],[r['test_r2'] for r in variance]);ax.set(ylabel='Held-out hidden-PC variance explained',title='Regression alternative to sparse crossed dPCA');savefig(fig,out,'factor_conditioned_variance')
    fig,ax=plt.subplots(figsize=(6,5))
    for sl,color in [('train','#5577b3'),('test','#d48236')]:
        mask=primary&test&(index.capability_slice==sl);ax.scatter(fz[mask,0],fz[mask,1],s=12,alpha=.4,c=color,label='Familiar' if sl=='train' else 'Novel')
    ax.set(xlabel='Familiar-training PC1',ylabel='Familiar-training PC2');ax.legend();savefig(fig,out,'novel_profile_geometry')
    result=dict(pca=pca,decoder_transfer=decoder,variance_partition=variance,representational_models=rsa,
        within_faster_group_mean_crossvalidated_distance=float(within.mean()),novel_geometry=novel_rows,
        temporal_accuracy=temporal.tolist(),n_samples=len(hidden),n_episodes=index.episode.nunique(),
        limitations=['Distances use training-only diagonal shrinkage, not full-noise whitening.',
            'RDM model scores generalize over episode repetitions, not independent profile pairs; pairwise correlations are descriptive.',
            'dPCA replaced by multivariate regression because the 46 x 1096 crossed design is sparse.',
            'Layout conditioning uses geometry, phase and round; raw current observations were not recorded.',
            'Target-layout transfer excludes target samples from fitting but preceding histories may include those layouts.',
            'Post-observation states and remembered histories can mix with action and task state; geometry is not causal evidence.',
            'Novel coordinate-box support is descriptive and depends on rank; high-dimensional subspace residuals are also reported.'])
    write_json(out/'geometry_summary.json',result)
    return result
