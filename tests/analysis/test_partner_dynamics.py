"""Boundary, leakage, driven-operator and intervention identity regressions."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from eval.partner_dynamics.data import split_repetitions,round_boundaries
from eval.partner_dynamics.numeric import Ridge,cross_distance
from eval.partner_dynamics.interventions import perturb
from eval.representation_analysis import episode_valid_length

def test_terminal_inclusive_and_no_fabricated_rounds():
    n=episode_valid_length(np.r_[np.zeros(39,bool),True,True])
    assert n==40
    record={'round_idx':np.repeat(np.arange(20),2),'round_done':np.tile([False,True],20),'is_t0':np.tile([True,False],20)}
    starts,ends=round_boundaries(record)
    assert np.array_equal(starts,np.arange(0,40,2))
    assert np.array_equal(ends,np.arange(1,40,2))
    record['round_idx'][4]=0
    with pytest.raises(ValueError,match='round transition'):round_boundaries(record)

def test_episode_splits_are_deterministic_disjoint():
    split=split_repetitions(20,2026)
    assert split==split_repetitions(20,2026)
    assert split!=split_repetitions(20,1)
    assert [list(split.values()).count(s) for s in ('train','validation','test')]==[12,4,4]
    groups={s:{r for r,x in split.items() if x==s} for s in ('train','validation','test')}
    assert not groups['train']&groups['test'] and not groups['train']&groups['validation']

def test_preprocessing_fits_train_only_and_null_labels_fail():
    rng=np.random.default_rng(2);x=rng.normal(size=(200,8));y=x@rng.normal(size=(8,2))
    fit=Ridge().fit(x[:120],y[:120],.1)
    assert np.mean((fit.predict(x[120:])-y[120:])**2)<.01
    mean=fit.mean.copy();fit.predict(x[120:]+1e6);assert np.array_equal(mean,fit.mean)
    null=Ridge().fit(x[:120],rng.permutation(y[:120]),.1)
    assert np.mean((null.predict(x[120:])-y[120:])**2)>1

def test_crossvalidated_null_distances_are_unbiased():
    rng=np.random.default_rng(4);a=rng.normal(size=(400,30));b=rng.normal(size=(400,30))
    d=cross_distance(a,b,np.ones(30));assert abs(d[np.triu_indices(400,1)].mean())<.05
    assert np.any(d<0)

def test_zero_intervention_identity_and_norm_matched_controls():
    rng=np.random.default_rng(2);h=rng.normal(size=16);donor=rng.normal(size=16);q=np.eye(16)[:,:3]
    for kind in ('whole','partner','attenuate','random','nonpartner','sham'):
        assert np.array_equal(perturb(h,donor,q,np.zeros(16),kind,0,rng),h)
    p=perturb(h,donor,q,np.zeros(16),'partner',1,rng)-h
    r=perturb(h,donor,q,np.zeros(16),'random',1,rng)-h
    n=perturb(h,donor,q,np.zeros(16),'nonpartner',1,rng)-h
    assert np.linalg.norm(p)==pytest.approx(np.linalg.norm(r))
    assert np.linalg.norm(p)==pytest.approx(np.linalg.norm(n))
    assert np.linalg.norm(q.T@n)<1e-12

def driven_segments(a,b):
    rng=np.random.default_rng(2);segments=[]
    for _ in range(5):
        u=rng.normal(size=(180,2)).astype(np.float32);h=np.zeros((180,2),np.float32);h[0]=rng.normal(size=2)
        for t in range(179):h[t+1]=a@h[t]+b@u[t]
        segments.append((h,u))
    return segments

def test_input_dsa_recovers_driven_operators_without_boundary_transitions():
    from eval.partner_dynamics.input_dsa import fit,arrays,pairs,distance
    a=np.array([[.7,.1],[0,.4]],np.float32);b=np.array([[.5,0],[.1,.3]],np.float32)
    segments=driven_segments(a,b);model=fit(segments,2,1,1e-8);result=arrays(model)
    assert model.X.shape[1]==5*179
    np.testing.assert_allclose(result['A_havok_dmd'],a,atol=1e-5)
    np.testing.assert_allclose(result['B_havok_dmd'],b,atol=1e-5)
    x,y,u=pairs(segments,1)
    assert np.mean((x@a.T+u@b.T-y)**2)<1e-10
    identity=distance(result,result)
    assert max(identity.values())<1e-4
    changed_input=arrays(fit(driven_segments(a,2*b),2,1,1e-8))
    metric=distance(result,changed_input)
    assert metric['input']>metric['intrinsic']+1e-3
    changed_state=arrays(fit(driven_segments(.5*a,b),2,1,1e-8))
    assert distance(result,changed_state)['intrinsic']>1e-3

def test_post_observation_input_alignment_and_delay_windows():
    from eval.partner_dynamics.input_dsa import pairs
    # h[t+1] consumes e[t+1]; adapter shifts e before calling DMDc.
    h=np.arange(7,dtype=np.float32)[:,None]
    e=(100+np.arange(7,dtype=np.float32))[:,None]
    shifted=np.r_[e[1:],e[-1:]]
    x,y,u=pairs([(h,shifted),(h+20,shifted+20)],2)
    assert len(x)==2*5
    np.testing.assert_array_equal(x[0],[1,0]);np.testing.assert_array_equal(y[0],[2,1])
    assert u[0,0]==102
    assert not np.any((x[:,0]<7)&(y[:,0]>=20))

def test_original_flax_adapter_derivatives_match_directional_differences():
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    from eval.partner_dynamics.adapters import PolicyAdapter
    width=8;cell=nn.GRUCell(features=width)
    h=jnp.asarray(np.random.default_rng(2).normal(size=width),dtype=jnp.float32);e=jnp.asarray(np.random.default_rng(3).normal(size=width),dtype=jnp.float32)
    original=cell.init(jax.random.PRNGKey(4),h[None],e[None])
    params={'params':{'ScannedRNN_0':{'GRUCell_1':original['params']}}}
    adapter=PolicyAdapter(None,None,{'GRU_HIDDEN_DIM':width})
    np.testing.assert_allclose(adapter.update(params,h[None],e[None]),cell.apply(original,h[None],e[None])[0],atol=1e-6,rtol=1e-6)
    direction=jnp.ones(width)/np.sqrt(width);eps=1e-3
    jh=adapter.jac_h(params,h,e);je=adapter.jac_e(params,h,e)
    finite_h=(adapter.update(params,(h+eps*direction)[None],e[None])-adapter.update(params,(h-eps*direction)[None],e[None]))[0]/(2*eps)
    finite_e=(adapter.update(params,h[None],(e+eps*direction)[None])-adapter.update(params,h[None],(e-eps*direction)[None]))[0]/(2*eps)
    np.testing.assert_allclose(jh@direction,finite_h,atol=1e-4,rtol=2e-3)
    np.testing.assert_allclose(je@direction,finite_e,atol=1e-4,rtol=2e-3)

def test_delay_validation_scores_future_state_not_known_past_blocks():
    import torch
    from types import SimpleNamespace
    from eval.partner_dynamics.dynamics import evaluate
    h=np.arange(1,9,dtype=np.float32)[:,None];u=np.zeros_like(h)
    # The past block is copied exactly, but the next current state is predicted as zero.
    a=torch.tensor([[0.,0.],[1.,0.]])
    model=SimpleNamespace(A_v=a,B_v=torch.zeros((2,1)),A_havok_dmd=a,B_havok_dmd=torch.zeros((2,1)))
    mse,_=evaluate(model,[(h,u)],2)
    assert mse==pytest.approx(np.mean(h[2:]**2))

def test_schema_rejects_placeholder_states_and_misaligned_fields():
    from eval.partner_dynamics.data import validate_schema
    data={'hidden_state':np.zeros((2,10,8),np.float32),'dones':np.zeros((2,10),bool),'capability':np.zeros((2,10,2),np.int32)}
    validate_schema(data)
    data['dones']=np.zeros((2,9),bool)
    with pytest.raises(ValueError,match='shape mismatch'):validate_schema(data)
    data['dones']=np.zeros((2,10),np.int32)
    with pytest.raises(ValueError,match='boolean dtype'):validate_schema(data)
    data['hidden_state']=np.zeros((2,10,1),np.float32)
    with pytest.raises(ValueError,match='real recurrent'):validate_schema(data)
