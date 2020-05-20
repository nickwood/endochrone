# -*- coding: utf-8 -*-
import numpy as np

from endochrone.clustering import DBSCAN
from endochrone.utils import lazy_test_runner as ltr

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_fit_1d_and_predict():
    X = np.hstack([np.arange(0, 1, 0.1), np.arange(3, 4, 0.1)]).reshape(20, 1)
    dbscan_model = DBSCAN(eps=0.3)
    dbscan_model.fit(features=X)
    assert len(dbscan_model.targets_ == 20)
    assert np.all(dbscan_model.targets_ == np.array([0]*10 + [1]*10))

    assert dbscan_model.predict_single(p=0.5) == 0

    X_test = np.array([-0.5, -0.2, 0.2, 0.5, 0.8, 1.1, 1.4, 2.8, 3.4, 3.8,
                       4.4])[:, np.newaxis]
    Y_true = np.array([-1, 0, 0, 0, 0, 0, -1, 1, 1, 1, -1])
    Y_pred = dbscan_model.predict(features=X_test)
    assert np.all(Y_pred == Y_true)


def test_fit_1d_removes_noise():
    X = np.hstack([np.arange(0, 1, 0.1),
                   [1.7, 2.1, 2.2, 2.5],
                   np.arange(3, 4, 0.1)])[:, np.newaxis]
    dbscan_model = DBSCAN(eps=0.3)
    dbscan_model.fit(features=X)
    assert len(dbscan_model.targets_ == 20)
    assert np.all(dbscan_model.targets_ == np.array([0]*10 + [1]*10))
    assert not np.any([noise in dbscan_model.features_
                       for noise in [1.7, 2.1, 2.2, 2.5]])


def test_fit_2d_and_predict():
    X0 = np.vstack([np.arange(0, 1, 0.1), np.arange(1.5, 2.5, 0.1)]).T
    XN = np.vstack([np.arange(1.5, 6.5, 0.5), np.arange(2.6, 6.6, 0.4)]).T
    X1 = np.vstack([np.arange(6.5, 7.5, 0.1), np.arange(6.5, 7.5, 0.1)]).T
    X_all = np.vstack([X0, XN, X1])

    dbscan_model = DBSCAN(eps=0.5)
    dbscan_model.fit(features=X_all)
    assert len(dbscan_model.targets_ == 20)
    assert np.all(dbscan_model.targets_ == np.array([0]*10 + [1]*10))
    assert not np.any([noise in dbscan_model.features_ for noise in XN])

    assert dbscan_model.predict_single(p=[0.5, 2.1]) == 0

    X_test = np.array([[0.5, 2.1], [0.8, 2.6], [0.6, 2.4], [1.0, 2.7],
                       [2.7, 3.0], [4.0, 4.0], [4.7, 8.6], [-0.3, 7.0],
                       [6.7, 7.0], [7.0, 7.0], [7.7, 7.6], [7.3, 7.0]])
    Y_true = np.array([0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1])
    Y_pred = dbscan_model.predict(features=X_test)
    assert np.all(Y_pred == Y_true)


ltr()
