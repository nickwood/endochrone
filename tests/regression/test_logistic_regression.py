# -*- coding: utf-8 -*-
import numpy as np
import pytest
import random

from endochrone.regression import logistic_regression as lr
from endochrone.regression import LogisticRegression
from endochrone.utils import lazy_test_runner as ltr

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_log_likelihood():
    X = np.arange(0, 2)[:, np.newaxis]  # [0, 1]
    Y = np.arange(0, 2)
    coefs = {'θ0': 1.0, 'θ1': 3.0}
    assert lr.neg_log_likelihood(X=X, Y=Y, **coefs) == pytest.approx(1.3314116)

    X = np.array([-10, -9, -8, 8, 9, 11])[:, np.newaxis]
    Y = np.array([0]*3 + [1]*3)
    coefs1 = {'θ0': 1.0, 'θ1': 1.0}
    ll1 = 0.001545220299
    assert lr.neg_log_likelihood(X=X, Y=Y, **coefs1) == pytest.approx(ll1)
    coefs2 = {'θ0': 0.3, 'θ1': 0.8}
    ll2 = 0.005595104889
    assert lr.neg_log_likelihood(X=X, Y=Y, **coefs2) == pytest.approx(ll2)
    coefs3 = {'θ0': 1.54, 'θ1': 2.0}
    ll3 = 0.0000006330366284
    assert lr.neg_log_likelihood(X=X, Y=Y, **coefs3) == pytest.approx(ll3)
    coefs4 = {'θ0': 0.0, 'θ1': 2.0}
    ll4 = 0.0000002578703965
    assert lr.neg_log_likelihood(X=X, Y=Y, **coefs4) == pytest.approx(ll4)


def test_evaluate():
    X = np.arange(0, 2)[:, np.newaxis]  # [0, 1]
    coefs = {'θ0': 1.0, 'θ1': 3.0}

    exp = np.array([0.731059, 0.982014])
    assert np.all(lr.evaluate(X=X, **coefs) == pytest.approx(exp))

    # Ensure that warnings are supressed when we try to calculate e^big number
    X = np.array([7, 4, 5, 18, 17, 20])[:, np.newaxis]
    coefs = {'θ0': -7.971497567245022, 'θ1': -46.875555721432754}
    with pytest.warns(None) as record:
        lr.evaluate(X=X, **coefs)
    assert len(record) == 0


def test_1d_fit_and_predict():
    X = np.array([-10, -9, -8, 8, 9, 11])[:, np.newaxis]
    Y = np.array([0]*3 + [1]*3)

    gd_params = {'tol': 0.0001, 'max_iter': 100, 'learning_rate': 2,
                 'boost_prob': 0.7}
    test_lr = LogisticRegression(gd_params=gd_params, suppress_warnings=True)
    test_lr.fit(X_train=X, Y_train=Y)

    X_test1 = np.arange(-10, -1)[:, np.newaxis]
    assert np.all(test_lr.predict(X_test1) == 0)

    X_test2 = np.arange(1, 10)[:, np.newaxis]
    assert np.all(test_lr.predict(X_test2) == 1)


def test_far_from_zero():
    X = np.array([7, 4, 5, 18, 17, 20])[:, np.newaxis]
    Y = np.array([0]*3 + [1]*3)

    gd_params = {'tol': 0.0001, 'max_iter': 1000, 'learning_rate': 3,
                 'boost_prob': 0.4}
    test_lr = LogisticRegression(gd_params=gd_params, suppress_warnings=True)

    test_lr.fit(X_train=X, Y_train=Y)

    X_test1 = np.arange(0, 11)[:, np.newaxis]
    assert np.all(test_lr.predict(X_test1) == 0)

    X_test2 = np.arange(14, 20)[:, np.newaxis]
    assert np.all(test_lr.predict(X_test2) == 1)


def test_mirrored():
    X = np.array([5, 4, 3, 2, 1, 0])[:, np.newaxis]
    Y = np.array([0]*3 + [1]*3)

    gd_params = {'tol': 0.0001, 'max_iter': 1000, 'learning_rate': 3,
                 'boost_prob': 0.4}
    test_lr = LogisticRegression(gd_params=gd_params, suppress_warnings=True)

    test_lr.fit(X_train=X, Y_train=Y)

    X_test1 = np.arange(3, 8)[:, np.newaxis]
    assert np.all(test_lr.predict(X_test1) == 0)

    X_test2 = np.arange(-2, 2.5)[:, np.newaxis]
    assert np.all(test_lr.predict(X_test2) == 1)


def test_overlapping_boundary():
    X = np.array([-3, -2, -1, -2.5, 0.5, 0, 1, 2.3, 3.1, 4])[:, np.newaxis]
    Y = np.array([0]*5 + [1]*5)

    gd_params = {'tol': 0.0001, 'max_iter': 1000, 'learning_rate': 3,
                 'boost_prob': 0.4}
    test_lr = LogisticRegression(gd_params=gd_params, suppress_warnings=True)
    test_lr.fit(X_train=X, Y_train=Y)

    X_test1 = np.arange(-10, 0)[:, np.newaxis]
    assert np.all(test_lr.predict(X_test1) == 0)

    X_test2 = np.arange(1, 10)[:, np.newaxis]
    assert np.all(test_lr.predict(X_test2) == 1)


def test_multi_dimensional():
    X = np.transpose([[5, 4, 3, 2, 1, 0], [0, 1, 2, 6, 7, 8]])
    Y = np.array([0]*3 + [1]*3)

    gd_params = {'tol': 0.0001, 'max_iter': 1000, 'learning_rate': 3,
                 'boost_prob': 0.4}
    test_lr = LogisticRegression(gd_params=gd_params)

    test_lr.fit(X_train=X, Y_train=Y)

    X_test1 = np.array([[5, 1], [4, 2], [3, 0], [3, 2], [3, 4]])
    assert np.all(test_lr.predict(X_test1) == 0)

    X_test2 = np.array([[3, 5], [2, 3.5], [2, 4], [1, 5], [-10, 10]])
    assert np.all(test_lr.predict(X_test2) == 1)


def test_exceptions():
    test_lr = LogisticRegression(suppress_warnings=False)
    X = np.array([0, 1, 2, 3])
    Y = np.array([0, 0, 1, 1])

    # invalid X dimensions
    with pytest.raises(ValueError):
        test_lr.fit(X_train=X, Y_train=Y)

    # invalid Y dimensions
    with pytest.raises(ValueError):
        test_lr.fit(X_train=X[:, np.newaxis], Y_train=Y[:, np.newaxis])

    # invalid Y values
    with pytest.raises(ValueError):
        test_lr.fit(X_train=X[:, np.newaxis], Y_train=np.array([0, 1, 2, 3]))

    # model not fit
    with pytest.raises(AttributeError):
        test_lr.predict(X_test=X)

    # incompatible sizes
    with pytest.raises(ValueError):
        test_lr.fit(X_train=X[:, np.newaxis], Y_train=np.array([0, 1, 1]))

    with pytest.warns(RuntimeWarning):
        test_lr.fit(X_train=X[:, np.newaxis], Y_train=Y)

    # invalid X_test shape
    with pytest.raises(ValueError):
        test_lr.predict(X_test=np.array([[0, 1], [1, 0]]))
    with pytest.raises(ValueError):
        test_lr.predict(X_test=np.array([0, 1, 0]))


ltr()
