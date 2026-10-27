"""Check temporal selection and the optimized five-seed bootstrap against definitions."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"eval"))
from analyze_instantaneous_decoding import instant_features,bootstrap_seeds


def test_instant_state_does_not_average_history_or_read_terminal_padding():
    hidden=np.array([[[1.],[9.],[3.],[999.]],[[2.],[4.],[8.],[16.]]])
    got=instant_features(hidden,np.array([3,4]),np.arange(4))[...,0]
    np.testing.assert_array_equal(got,np.array([[1.,2.],[9.,4.],[3.,8.],[3.,16.]]))
    assert got[1,0] != hidden[0,:2,0].mean()


def test_compressed_bootstrap_matches_all_10000_draws_and_linear_quantiles():
    values=np.random.default_rng(27).random((5,3,7))
    values[:,0,:]=np.array([0,.25,.5,.75,1])[:,None]
    draws=np.random.default_rng(30000).integers(0,5,size=(10000,5))
    samples=values[draws].mean(axis=1)
    mean,low,high=bootstrap_seeds(values)
    np.testing.assert_allclose(mean,values.mean(axis=0),atol=1e-14)
    np.testing.assert_allclose(low,np.quantile(samples,.025,axis=0),atol=1e-14)
    np.testing.assert_allclose(high,np.quantile(samples,.975,axis=0),atol=1e-14)
