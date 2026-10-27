"""Check within-panel centering, constant curves, and lag direction."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"eval"))
from analyze_decoding_behavior_correlations import centered_correlation,correlations


def test_between_panel_offsets_do_not_create_or_destroy_within_panel_association():
    t=np.arange(20,dtype=float)
    x=np.stack([t,t+1000])
    y=np.stack([2*t+5000,2*t-5000])
    np.testing.assert_allclose(centered_correlation(x,y),1,atol=1e-14)


def test_constant_behavior_panel_is_retained_without_fabricating_correlation():
    t=np.arange(20,dtype=float)
    x=np.stack([t,np.zeros(20)])
    y=np.stack([t,t])
    np.testing.assert_allclose(centered_correlation(x,y),np.sqrt(.5),atol=1e-14)
    assert np.isnan(centered_correlation(np.zeros_like(x),y))


def test_positive_lag_means_behavior_precedes_decoding():
    rng=np.random.default_rng(71)
    x=rng.normal(size=31);y=rng.normal(size=31);y[3:]=x[:-3]
    values=np.stack([y,x],axis=-1)[None,None,:,:]
    np.testing.assert_allclose(correlations(values,lag=3),[1],atol=1e-14)
    assert abs(correlations(values,lag=0)[0])<.5
    np.testing.assert_allclose(correlations(values,width=5,lag=3),[1],atol=1e-14)
