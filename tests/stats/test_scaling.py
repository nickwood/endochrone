# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone.stats.scaling import FeatureScaling
from endochrone.utils import lazy_test_runner as ltr

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_min_max():
    X = np.transpose([[1, 2, 5, 7, 8, 9, 10],
                      [7, 8, 9, 10, 11, 12, 20]])
    scaler = FeatureScaling(method='min_max')
    scaler.fit(features=X)
    act = scaler.transform(features=X)

    exp_X1 = np.array([0, 1, 4, 6, 7, 8, 9]) / 9
    exp_X2 = np.array([0, 1, 2, 3, 4, 5, 13]) / 13
    exp = np.transpose([exp_X1, exp_X2])
    assert np.all(act == exp)

    scaler2 = FeatureScaling(method='min_max')
    act2 = scaler2.fit_and_transform(features=X)
    assert np.all(act2 == exp)

    assert np.all(scaler2.reverse(features=act2) == pytest.approx(X))


def test_min_max_with_targets():
    X = np.transpose([[1, 2, 5, 7, 8, 9, 10],
                      [7, 8, 9, 10, 11, 12, 20]])
    Y = np.array([1, 2, 3, 4, 5, 6, 7])
    scaler = FeatureScaling(method='min_max')

    scaler.fit(features=X, targets=Y)
    s_feat, s_targ = scaler.transform(features=X, targets=Y)

    exp_X1 = np.array([0, 1, 4, 6, 7, 8, 9]) / 9
    exp_X2 = np.array([0, 1, 2, 3, 4, 5, 13]) / 13
    exp_feat = np.transpose([exp_X1, exp_X2])
    exp_targ = np.array([0, 1, 2, 3, 4, 5, 6]) / 6

    scaler2 = FeatureScaling(method='min_max')
    s_feat2, s_targ2 = scaler2.fit_and_transform(features=X, targets=Y)
    assert np.all(exp_feat == s_feat2)
    assert np.all(exp_targ == s_targ2)

    feat_2, targ_2 = scaler2.reverse(features=s_feat2, targets=s_targ2)
    assert np.all(feat_2 == X)
    assert np.all(targ_2 == Y)


def test_mean_norm_with_targets():
    X = np.transpose([[1, 2, 5, 7, 8, 9, 10],
                      [7, 8, 9, 10, 11, 12, 20]])
    Y = np.array([1, 2, 3, 4, 5, 6, 7])
    scaler = FeatureScaling(method='mean_norm')
    s_feat, s_targ = scaler.fit_and_transform(features=X, targets=Y)

    exp_X1 = np.array([-5, -4, -1, 1, 2, 3, 4]) / 9
    exp_X2 = np.array([-4, -3, -2, -1, 0, 1, 9]) / 13
    exp_X = np.transpose([exp_X1, exp_X2])
    exp_Y = np.array([-3, -2, -1, 0, 1, 2, 3]) / 6
    assert np.all(s_feat == exp_X)
    assert np.all(s_targ == exp_Y)

    feat_2, targ_2 = scaler.reverse(features=s_feat, targets=s_targ)
    assert np.all(feat_2 == X)
    assert np.all(targ_2 == Y)


def test_mean_zscore():
    X = np.transpose([[1, 2, 5, 7, 8, 9, 10],
                      [7, 8, 9, 10, 11, 12, 20]])
    scaler = FeatureScaling()
    assert scaler.method == 'z_score'
    scaler.fit(features=X)
    act = scaler.transform(features=X)

    exp_X1 = np.array([-5, -4, -1, 1, 2, 3, 4]) / 3.464101615
    exp_X2 = np.array([-4, -3, -2, -1, 0, 1, 9]) / 4.320493799
    exp = np.transpose([exp_X1, exp_X2])
    assert np.all(act == pytest.approx(exp))

    scaler2 = FeatureScaling(method='z_score')
    act2 = scaler2.fit_and_transform(features=X)
    assert np.all(act2 == pytest.approx(exp))

    assert np.all(scaler2.reverse(features=act2) == pytest.approx(X))


def test_z_score_with_targets():
    X = np.transpose([[1, 2, 5, 7, 8, 9, 10],
                      [7, 8, 9, 10, 11, 12, 20]])
    Y = np.array([1, 2, 3, 4, 5, 6, 7])
    scaler = FeatureScaling(method='z_score')
    s_feat, s_targ = scaler.fit_and_transform(features=X, targets=Y)

    exp_X1 = np.array([-5, -4, -1, 1, 2, 3, 4]) / 3.464101615
    exp_X2 = np.array([-4, -3, -2, -1, 0, 1, 9]) / 4.320493799
    exp_X = np.transpose([exp_X1, exp_X2])
    exp_Y = np.array([-3, -2, -1, 0, 1, 2, 3]) / 2.160246899
    assert np.all(s_feat == pytest.approx(exp_X))
    assert np.all(s_targ == pytest.approx(exp_Y))

    feat_2, targ_2 = scaler.reverse(features=s_feat, targets=s_targ)
    assert np.all(feat_2 == pytest.approx(X))
    assert np.all(targ_2 == pytest.approx(Y))


def test_exceptions():
    with pytest.raises(ValueError):
        _ = FeatureScaling(method='dfsg')

    val_X = np.transpose([[1, 2, 3], [1, 2, 3]])
    err_X = np.transpose([[2, 2, 2], [1, 2, 3]])
    val_Y = np.array([2, 3, 4])
    err_Y = np.array([4, 4, 4])

    with pytest.raises(ValueError):
        _ = FeatureScaling(method='min_max').fit(features=err_X, targets=val_Y)

    with pytest.raises(ValueError):
        _ = FeatureScaling(method='mean_norm').fit(features=err_X)

    with pytest.raises(ValueError):
        _ = FeatureScaling(method='z_score').fit(features=err_X, targets=val_Y)

    with pytest.raises(ValueError):
        _ = FeatureScaling(method='min_max').fit(features=val_X, targets=err_Y)

    with pytest.raises(ValueError):
        _ = FeatureScaling(method='z_score').fit(features=val_X, targets=err_Y)

    model = FeatureScaling(method='min_max')
    model.fit(features=val_X)
    with pytest.raises(ValueError):
        model.transform(features=val_X, targets=val_Y)


def test_transform_X_and_Y_separately():
    X = np.transpose([[1, 2, 3], [1, 2, 3]])
    Y = np.array([4, 3, 2])

    model = FeatureScaling(method='mean_norm')
    model.fit(features=X, targets=Y)

    s_X = model.transform(features=X)
    s_Y = model.transform(targets=Y)

    exp_X = np.array([[-0.5, -0.5], [0, 0], [0.5, 0.5]])
    exp_Y = np.array([0.5, 0, -0.5])
    assert np.all(exp_X == s_X)
    assert np.all(exp_Y == s_Y)

    feat = model.reverse(features=s_X)
    targ = model.reverse(targets=s_Y)
    assert np.all(X == feat)
    assert np.all(Y == targ)


ltr()
