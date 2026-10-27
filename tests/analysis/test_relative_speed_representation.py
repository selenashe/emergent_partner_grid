"""Scientific invariants of the specialist-only binary representation analysis."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'eval'))
import relative_speed_representation as analysis


def test_cooldown_direction_and_tied_profiles_are_rejected():
    np.testing.assert_array_equal(analysis.relative_classes([[1,7],[8,2],[3,4]]),[0,1,0])
    for pairs in [[[0,0]],[[9,9]],[[1.5,2]]]:
        with pytest.raises(ValueError):analysis.relative_classes(pairs)


def test_binary_balanced_accuracy_is_average_class_recall():
    truth=np.array([0]*4+[1]*12);predicted=np.ones(16,dtype=int)
    metrics,confusion=analysis.class_metrics(truth,predicted)
    assert metrics['accuracy']==pytest.approx(.75)
    assert metrics['balanced_accuracy']==pytest.approx(.5)
    np.testing.assert_array_equal(confusion,[[0,4],[0,12]])
    with pytest.raises(ValueError,match='classes'):analysis.class_metrics([0,0],[0,0])


def test_original_profile_split_has_368_fit_92_test_per_class():
    populations=analysis.original.capability_populations()
    profiles=np.array(sorted(populations['train']|populations['test']))
    pairs=np.repeat(profiles,20,axis=0);repetitions=np.tile(np.arange(20),len(profiles))
    labels=analysis.relative_classes(pairs)
    split=analysis.original.make_probe_split(0)
    train,test=analysis.masks(repetitions,pairs,split)
    assert (train.sum(),test.sum())==(736,184)
    np.testing.assert_array_equal(np.bincount(labels[train]),[368,368])
    np.testing.assert_array_equal(np.bincount(labels[test]),[92,92])
    assert set(repetitions[test])=={14,9,1,15}
    with pytest.raises(ValueError,match='Incorrect profile split'):analysis.masks(repetitions[:-1],pairs[:-1],split)


def test_probe_uses_only_training_labels_and_training_loss_weights():
    torch=pytest.importorskip('torch');torch.set_num_threads(1)
    features=np.random.default_rng(34).normal(size=(1,16,128)).astype(np.float32)
    labels=np.array([[0]*3+[1]*9+[0,1,0,1]])
    train=np.arange(16)<12
    _,_,first=analysis.train_probes(features,labels,train,[10000],steps=8)
    altered=labels.copy();altered[:,~train]=1-altered[:,~train]
    _,_,second=analysis.train_probes(features,altered,train,[10000],steps=8)
    np.testing.assert_array_equal(first['weight'],second['weight'])
    np.testing.assert_array_equal(first['bias'],second['bias'])
    np.testing.assert_allclose(first['class_weights'],[[2,2/3]])
    assert first['weight'].shape==(1,128,2)
    with pytest.raises(ValueError,match='every class'):
        analysis.train_probes(features,np.ones_like(labels),train,[10000],steps=1)
    balanced=np.tile([0,1],8)[None,:]
    _,_,model=analysis.train_probes(features,balanced,train,[10000],steps=1)
    np.testing.assert_array_equal(model['class_weights'],[[1,1]])


def test_summary_requires_all_five_policy_seeds():
    import pandas as pd
    table=pd.DataFrame([dict(version='v2',condition='rnn_diverse_influence',axis='round',cutoff=20,
        kind='primary',population='all',training_seed=1,balanced_accuracy=.8)])
    with pytest.raises(ValueError,match='five independent'):analysis.summary(table)
