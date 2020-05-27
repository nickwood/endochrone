# -*- coding: utf-8 -*-
import numpy as np

from endochrone.utils import lazy_test_runner as ltr
from endochrone.decomposition import TomekLinks
from endochrone.decomposition import undersampling

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_simple_tomek():
    X = np.array([0, 0.4, 1.5, 2.5, 3.2, 4.3, 4.5, 5.7, 6, 10]).reshape(10, 1)
    Y = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1])

    tomek_model = TomekLinks()
    act_X, act_Y = tomek_model.fit_and_transform(features=X, targets=Y)
    assert np.all(act_X.T == [0, 0.4, 1.5, 4.3, 4.5, 10])
    assert np.all(act_Y == [0, 0, 0, 1, 1, 1])


def test_rescan_when_tomek_link_removed():
    X = np.array([0, 1, 19, 25, 27, 28, 30, 36, 54, 55]).reshape(10, 1)
    Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    tomek_model = TomekLinks()
    act_X, act_Y = tomek_model.fit_and_transform(features=X, targets=Y)
    assert np.all(act_X.T == [0, 1, 54, 55])
    assert np.all(act_Y == [0, 0, 1, 1])


def test_2d_tomek():
    X = np.array([[0, 1, 2],
                  [3, 4.1, 5],
                  [6, 7.3, 8],
                  [9, 10.1, 11],
                  [12, 13.2, 14],
                  [15, 16.4, 17],
                  [18, 19.1, 20],
                  [21, 22.6, 23],
                  [24, 25.2, 26],
                  [27, 28.3, 29]])
    Y = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1])

    tomek_model = TomekLinks()
    act_X, act_Y = tomek_model.fit_and_transform(features=X, targets=Y)
    assert act_X.shape == (6, 3)
    assert np.all(act_X[:, 0] == [0, 3, 12, 15, 18, 27])
    assert np.all(act_Y == [0, 0, 0, 1, 1, 1])


def test_all_tomek_links():
    X = np.arange(0, 4).reshape(4, 1)
    Y = np.array([0, 1, 0, 1])

    tomek_model = TomekLinks()
    act_X, act_Y = tomek_model.fit_and_transform(features=X, targets=Y)
    assert act_X.shape == (0, 1)
    assert act_Y.shape == (0, )


def test_nearest_unremoved_():
    n = np.arange(1, 10)
    assert undersampling.nearest_unremoved_(n, set()) == 1
    assert undersampling.nearest_unremoved_(n, set(range(0, 6))) == 6
    assert undersampling.nearest_unremoved_(n, set(n)) is None

    assert undersampling.nearest_unremoved_([], set()) is None
    assert undersampling.nearest_unremoved_([], {1, 2}) is None


ltr()
