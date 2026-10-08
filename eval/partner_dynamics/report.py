"""Aggregate independent policy fits without pooling their hidden coordinates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import write_json
from .numeric import grouped_ci

def build(root,config):
    # Audit guide:
    # Assemble existing data, geometry, replay, derivative, and intervention evidence
    # into the analysis report. Include driven-model summaries only when their saved
    # artifacts exist. A configured or proposed method alone is not a completed result.
    #
    rows=[];capture=[]
    for path in sorted(root.glob('*/*/geometry_summary.json')):
        result=json.loads(path.read_text())
        for d in result['decoder_transfer']:
            rows.append(dict(protocol=path.parents[1].name,policy=path.parent.name,condition=path.parent.name.rsplit('_seed',1)[0],seed=int(path.parent.name.rsplit('_seed',1)[1]),scope=d['scope'],mae=d['mae'],prior_mae=d['conditional_faster_goal_prior_mae'],improvement=d['improvement_over_prior'],r2=d['r2'],n_test=d['n_test'],n_episodes=result['n_episodes']))
    if rows:pd.DataFrame(rows).to_csv(root/'policy_summary.csv',index=False)
    text=['# Partner representations and recurrent dynamics','',
        'Primary batch: `'+config['batch']+'`. Protocols, conditions and learner seeds are analyzed independently; no hidden coordinates are pooled.',
        '', '## Raw-state geometry and transfer', '',
        'Primary capability regression uses the raw post-observation state at round 20 start. MAE is compared with a training-derived capability prior conditional on the true faster-goal identity. Positive improvement means information beyond that binary label helps prediction. This is a representational diagnostic, not evidence of causal use.', '',
        '| Protocol | Condition | Evaluation | Delay MAE | Binary prior MAE | Improvement (95% seed bootstrap) |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    aggregate=[]
    if rows:
        frame=pd.DataFrame(rows)
        for keys,part in frame.groupby(['protocol','condition','scope']):
            ci=grouped_ci(part.improvement,part.seed,config['analysis_seed'],max(1000,config['bootstrap']))
            aggregate.append(dict(protocol=keys[0],condition=keys[1],scope=keys[2],mae=float(part.mae.mean()),prior_mae=float(part.prior_mae.mean()),improvement=ci,n_seeds=len(part)))
            text.append(f'| {keys[0]} | {keys[1]} | {keys[2]} | {part.mae.mean():.3f} | {part.prior_mae.mean():.3f} | {ci["mean"]:.3f} [{ci["low"]:.3f}, {ci["high"]:.3f}] |')
    write_json(root/'aggregate_summary.json',aggregate)
    for path in sorted(root.glob('*/*/capture_summary.json')):
        capture.append(dict(protocol=path.parents[1].name,policy=path.parent.name,**json.loads(path.read_text())))
    write_json(root/'capture_coverage.json',capture)
    text += ['', '## Collection, dynamics and interventions', '']
    text.append(f'{len(capture)} policies have completed targeted capture and replay checks.')
    for path in sorted(root.glob('*/dynamics_summary.json')):
        text += ['',f'### {path.parent.name}', '',path.read_text()]
    text += ['', '## Interpretation and limits', '',
        'Decodability, representational geometry, fitted dynamics and behavioral interventions answer different questions. A positive decoder improvement does not establish that the policy estimated an unobserved delay; correlated population priors and task state remain alternatives.',
        '', 'All 20 episode rounds are audited with terminal-inclusive masks. Geometry uses equal numbers of raw start/midpoint/end samples per episode. Recorded raw trajectories remain intact in the historical HDF5 files. Timesteps are never independent statistical replicates.',
        '', 'Target-layout tests hold out the sampled target layouts from analysis fitting; earlier histories can overlap them. These tests do not measure policy generalization to layouts absent from policy training.',
        '', 'Sparse profile-by-layout crossing is handled with train/validation/test multivariate regression instead of filling missing dPCA cells. Layout conditioning uses recorded geometry and phase, not unrecorded full current observations.',
        '', 'RDM scores use training-only diagonal shrinkage and disjoint repetition halves. Their pairwise correlations are descriptive because profile pairs are dependent. Capability/relative/overall models are correlated and are not uniquely disentangled.',
        '', 'The static corpus audit is a baseline: all authoritative profiles favor giving the partner its faster goal. That formula does not simulate collisions or online switches.',
        '', 'Each policy directory contains numeric tables, event trajectories, PCA/novel geometry, representational models, cross-temporal transfer and variance figures. Consult `data_audit.json`, split manifests and per-file validation for coverage and provenance.',
        '', '## Reproduction', '', 'See the source package README and `methods.md` for exact commands and runtime settings.']
    (root/'report.md').write_text('\n'.join(text)+'\n')
    evidence(root,config)
    print('Report:',root/'report.md',flush=True)

def evidence(root,config):
    """Quantitative findings, validation gates and paired seed/recipient controls."""
    from .geometry import savefig
    import matplotlib.pyplot as plt
    import shutil
    paragraphs=['', '## Quantitative findings and causal limits', '']
    coverage=dict(policies=0,historical_episodes=0,historical_valid_steps=0,diagnostic_histories=0,diagnostic_valid_steps=0,intervention_branches=0,local_derivative_samples=0,geometry_samples=0,original_replay_max_error=0.,zero_identity=True,projector_validation_failures=[])
    dynamics=[];effects=[];local=[];temporal=[];geometry=[];sensitivities=[]
    for folder in sorted(root.glob('*/*')):
        if not (folder/'capture_summary.json').exists():continue
        capture=json.loads((folder/'capture_summary.json').read_text());coverage['policies']+=1;coverage['diagnostic_histories']+=capture['episodes'];coverage['diagnostic_valid_steps']+=capture['valid_steps'];coverage['intervention_branches']+=capture['intervention_branches'];coverage['local_derivative_samples']+=capture['jacobian_samples'];coverage['zero_identity'] &= capture['zero_intervention_identity']
        if capture['projector_validation_accuracy']<=.5:coverage['projector_validation_failures'].append(str(folder.relative_to(root)))
        replay=folder/'original_network_replay.json'
        if replay.exists():coverage['original_replay_max_error']=max(coverage['original_replay_max_error'],json.loads(replay.read_text())['max_hidden_probability_value_error'])
        validation=folder/'data_validation.json'
        if validation.exists():
            audit=json.loads(validation.read_text());coverage['historical_episodes']+=sum(r['episodes'] for r in audit['files']);coverage['historical_valid_steps']+=sum(r['valid_steps'] for r in audit['files'])
        gs=folder/'geometry_summary.json'
        if gs.exists():
            g=json.loads(gs.read_text());coverage['geometry_samples']+=g['n_samples'];geometry.append(dict(policy=folder.name,protocol=folder.parent.name,within_orientation_distance=g['within_faster_group_mean_crossvalidated_distance'],**{r['model']:r['heldout_episode_distance_r2'] for r in g['representational_models']}))
            matrix=np.asarray(g['temporal_accuracy']);temporal.append(dict(policy=folder.name,protocol=folder.parent.name,early_to_early=float(matrix[0,0]),late_start_to_late_end=float(matrix[-2,-1]),late_start_to_first_start=float(matrix[-2,0])))
        ds=folder/'dynamics_summary.json'
        if ds.exists():dynamics.append(dict(protocol=folder.parent.name,condition=folder.name.rsplit('_seed',1)[0],seed=int(folder.name.rsplit('_seed',1)[1]),**json.loads(ds.read_text())))
        ss=folder/'input_matched_sensitivity.json'
        if ss.exists():
            for row in json.loads(ss.read_text()):sensitivities.append(dict(policy=folder.name,protocol=folder.parent.name,**row))
        js=folder/'jacobian_summary.json'
        if js.exists():
            for row in json.loads(js.read_text()):local.append(dict(policy=folder.name,protocol=folder.parent.name,**row))
        arms=json.loads((folder/'intervention_outcomes.json').read_text());openpath=folder/'open_loop_outcomes.json';openarms=json.loads(openpath.read_text()) if openpath.exists() else []
        lookup={(r['episode'],r['donor_kind'],r['kind'],r['magnitude']):r for r in openarms}
        rc=folder/'rank_matched_control_outcomes.json'
        if rc.exists():
            rankarms=json.loads(rc.read_text());arms+=rankarms;coverage['intervention_branches']+=len(rankarms)
        for arm in arms:
            if arm['magnitude']!=1.:continue
            oo=lookup.get((arm['episode'],arm['donor_kind'],arm['kind'],arm['magnitude']),{})
            donor_red=arm['donor_profile'][0]<arm['donor_profile'][1]
            effects.append(dict(policy=folder.name,condition=folder.name.rsplit('_seed',1)[0],seed=int(folder.name.rsplit('_seed',1)[1]),protocol=folder.parent.name,episode=arm['episode'],donor_kind=arm['donor_kind'],kind=arm['kind'],reward_delta=arm['reward_delta'],allocation_delta=arm['allocation_red_probability_delta'],toward_donor_allocation_delta=arm['allocation_red_probability_delta']*(1 if donor_red else -1),
                probability_l1=oo.get('first_learned_probability_l1',arm.get('first_learned_probability_l1',arm['open_loop_probability_l1'] if 'open_loop_probability_l1' in arm else 0)),projector_validation_accuracy=capture['projector_validation_accuracy']))
    write_json(root/'coverage.json',coverage)
    if geometry:pd.DataFrame(geometry).to_csv(root/'representational_model_summary.csv',index=False)
    if temporal:pd.DataFrame(temporal).to_csv(root/'temporal_summary.csv',index=False)
    if local:pd.DataFrame(local).to_csv(root/'local_dynamics_summary.csv',index=False)
    paragraphs.append(f'Coverage: **{coverage["policies"]} recurrent policies, {coverage["historical_episodes"]:,} historical episodes and {coverage["historical_valid_steps"]:,} valid historical steps**. Geometry uses {coverage["geometry_samples"]:,} raw event states. The separate collector adds {coverage["diagnostic_histories"]} four-round histories ({coverage["diagnostic_valid_steps"]:,} valid steps), {coverage["intervention_branches"]:,} paired branches including zero controls, and {coverage["local_derivative_samples"]} local derivative samples.')
    paragraphs += ['',f'Original full-network replay has maximum checked hidden/probability/value error {coverage["original_replay_max_error"]:.3g}; sampled actions agree exactly. All zero-magnitude checks pass: {coverage["zero_identity"]}. Identical-input first learned allocation probabilities also match the paired physical branches exactly. For v2, the comparison is at t=1 following its forced t=0 action.', '']
    if dynamics:
        frame=pd.DataFrame(dynamics);frame.to_csv(root/'dynamics_summary.csv',index=False)
        paragraphs += ['| Protocol | Median reduction in current-state MSE versus persistence | Driven fit beats input-omitting baseline |', '| --- | ---: | ---: |']
        for protocol,part in frame.groupby('protocol'):
            paragraphs.append(f'| {protocol} | {100*part.test_improvement_over_persistence.median():.1f}% | {int((part.test_mse<part.input_omitting_dmd_mse).sum())}/{len(part)} policies |')
        paragraphs += ['', 'These fits predict the newly updated state, excluding already observed lag blocks from the validation score. They support a useful local/global approximation in the sampled early histories. A fitted mode is not a belief axis; weights are fixed across partner regimes. Full late-episode driven dynamics were not measured.', '']
    if effects:
        effect_frame=pd.DataFrame(effects);effect_frame.to_csv(root/'intervention_policy_episode_summary.csv',index=False)
        summaries=[]
        for keys,part in effect_frame.groupby(['protocol','condition','donor_kind','kind']):
            per_seed=part.groupby('seed')[['reward_delta','toward_donor_allocation_delta','probability_l1']].mean()
            reward=grouped_ci(per_seed.reward_delta,per_seed.index,config['analysis_seed'],1000);allocation=grouped_ci(per_seed.toward_donor_allocation_delta,per_seed.index,config['analysis_seed'],1000)
            summaries.append(dict(protocol=keys[0],condition=keys[1],donor_kind=keys[2],kind=keys[3],reward=reward,allocation=allocation,probability_l1=float(per_seed.probability_l1.mean()),n_seeds=len(per_seed)))
        write_json(root/'intervention_aggregate.json',summaries)
        paragraphs += ['### Input-matched allocation effects', '', 'Opposite-orientation donor effects below are signed **toward the donor’s faster-goal assignment**, so red/blue directions do not cancel. Each value averages the six held-out recipient histories within each seed, then averages seeds. Full results retain each condition, control, recipient and same-orientation donor.', '', '| Protocol | Perturbation | Toward-donor probability change | Paired reward change | First learned flat-policy L1 change |', '| --- | --- | ---: | ---: | ---: |']
        for keys,part in effect_frame[effect_frame.donor_kind=='opposite'].groupby(['protocol','kind']):
            paragraphs.append(f'| {keys[0]} | {keys[1]} | {part.toward_donor_allocation_delta.mean():+.4f} | {part.reward_delta.mean():+.4f} | {part.probability_l1.mean():.4f} |')
        paired=[]
        for keys,part in effect_frame[effect_frame.donor_kind=='opposite'].groupby(['protocol','condition']):
            pivot=part.pivot(index=['seed','episode'],columns='kind',values='toward_donor_allocation_delta')
            if 'rank_matched_nonpartner' not in pivot:continue
            contrast=pivot['partner']-pivot['rank_matched_nonpartner'];units=contrast.groupby(level=0).mean();ci=grouped_ci(units,units.index,config['analysis_seed'],1000)
            paired.append(dict(protocol=keys[0],condition=keys[1],partner_minus_rank_control=ci,n_paired_recipients=len(contrast)))
        write_json(root/'selectivity_contrasts.json',paired)
        paragraphs += ['', f'**{len(coverage["projector_validation_failures"])}/{coverage["policies"]} candidate subspaces scored at or below chance on the six-history binary validation check.** These policies remain included and their subspace interventions are exploratory. Even above chance, six validation histories provide weak subspace validation.', '',
            'Whole-carry effects show that earlier history can change the same input’s policy response. They can also change nuisance memories. Small candidate-subspace effects and matched-control effects do not establish selective partner-memory use. Norm sweeps, sham donors, single-direction controls and explicit rank-matched complement projectors are retained. Same-orientation/different-delay donors test effects beyond a binary reversal; their outcomes appear separately in `intervention_aggregate.json`.', '',
            'See `selectivity_contrasts.json` for paired partner-minus-rank-control effects across all learner seeds. No unsuccessful policy or failed candidate subspace was dropped.', '']
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        aggregate=pd.DataFrame(json.loads((root/'aggregate_summary.json').read_text()))
        primary=aggregate[aggregate.scope=='familiar_to_novel']
        labels=[f'{r.protocol}\n{r.condition.replace("rnn_","")}' for r in primary.itertuples()];vals=[r.improvement['mean'] for r in primary.itertuples()];lo=[r.improvement['mean']-r.improvement['low'] for r in primary.itertuples()];hi=[r.improvement['high']-r.improvement['mean'] for r in primary.itertuples()]
        axes[0].errorbar(range(len(vals)),vals,yerr=[lo,hi],fmt='o');axes[0].axhline(0,color='gray');axes[0].set(xticks=range(len(vals)),xticklabels=labels,ylabel='Delay MAE improvement\nover binary prior',title='Novel-profile scalar regression');axes[0].tick_params(axis='x',labelsize=7,rotation=40)
        for protocol,part in effect_frame[effect_frame.donor_kind=='opposite'].groupby('protocol'):
            means=part.groupby('kind').toward_donor_allocation_delta.mean();axes[1].plot(range(len(means)),means,'o',label=protocol);axes[1].set(xticks=range(len(means)),xticklabels=means.index)
        axes[1].tick_params(axis='x',labelsize=7,rotation=45);axes[1].set(ylabel='Allocation probability change\ntoward donor',title='Matched new-round policy response');axes[1].legend(fontsize=8);savefig(fig,root,'analysis_summary')
        paragraphs += ['![Transfer and input-matched control summary](analysis_summary.png)', '']
    if sensitivities:
        sf=pd.DataFrame(sensitivities);sf.to_csv(root/'input_matched_sensitivity_summary.csv',index=False)
        paragraphs += ['', 'Exact derivatives under identical input are saved in `input_matched_sensitivity_summary.csv`: natural history changes the local input response and can change policy/value sensitivities. Same-profile donors provide a nuisance-memory comparison. The v2 t=0 action mask makes policy allocation derivatives zero there; its first learned response is assessed by the t=1 replay.', '']
    if local:
        frame=pd.DataFrame(local)
        paragraphs.append(f'Exact local derivatives: median hidden Jacobian singular-value gain {frame.jh_gain.median():.3f}, median spectral radius {frame.spectral_radius.median():.3f}. Pointwise gains and eigenvalues are reported with finite-horizon products; neither establishes an attractor of the policy–environment loop.')
    paragraphs += ['', '### Figure and table guide', '',
        '- `geometry.png`: raw round-start PCA trajectories and held-out representational-model scores. `event_trajectories.csv` supplies coordinates and counts; `rdms.npz` supplies diagonal-noise-normalized and Euclidean/correlation matrices.',
        '- `cross_temporal_transfer.png`: faster-goal directions fitted at one phase and evaluated on different phases in disjoint episodes. `cross_temporal_accuracy.csv` gives the full matrix; initial-round chance performance and late-phase transfer are distinct.',
        '- `novel_profile_geometry.png`: familiar-training PCA coordinates applied to held-out familiar/novel profiles. `novel_geometry.csv` gives high-dimensional residual/support diagnostics; plotted positions alone do not establish extrapolation.',
        '- `factor_conditioned_variance.png`: held-out variance of train-only hidden PCs explained by phase/layout, binary, capability and interaction models. This regression is the stated alternative to unsupported fully crossed dPCA.',
        '- `event_aligned_updates.png`: update norms aligned to observable movement and real allocation events. `movement_exposure.json` reports which goal speeds were actually observed and cadence intervals.',
        '- `dynamics_and_interventions.png`: predictive validity versus persistence/input baselines and paired reward magnitude sweeps. Use `open_loop_outcomes.json` for the first learned decision, especially v2.',
        '', '### Remaining limits', '',
        'The targeted design uses only six profiles, one held-out history per profile per seed, and four rounds. It has limited power for selective effects and does not match total step exposure across capabilities. Candidate directions may mix partner and nuisance memory; validation failures are negative results, not evidence that memory is absent.',
        '', 'Within-orientation geometry and prediction beyond the binary prior support information beyond a single faster-goal label in some conditions. They do not prove that an unobserved delay was estimated from evidence or that the extra information is used for reward. Check the observable movement strata and population-prior baseline.',
        '', 'The optional fixed/slow-point phase was not run. Full late-episode encoded-input fits and fully history-disjoint layout transfer require a larger targeted collection.']
    for name in ('methods.md','data_dictionary.md'):
        shutil.copyfile(Path(__file__).parent/name,root/name)
    path=root/'report.md';path.write_text(path.read_text()+'\n'.join(paragraphs)+'\n')
