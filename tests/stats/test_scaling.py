# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone.stats import scaling as fs
from endochrone.utils import lazy_test_runner as ltr

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_min_max():
    X = np.transpose([[1, 2, 5, 7, 8, 9, 10],
                      [7, 8, 9, 10, 11, 12, 20]])
    Y = np.array([3, 4, 5, 1, 2, 2, 4])

    exp_X1 = np.array([0, 1, 4, 6, 7, 8, 9]) / 9
    exp_X2 = np.array([0, 1, 2, 3, 4, 5, 13]) / 13
    exp_X = np.transpose([exp_X1, exp_X2])
    exp_Y = np.array([2, 3, 4, 0, 1, 1, 3]) / 4

    with pytest.raises(ValueError):
        fs.min_max(np.array([[0, 0], [0, 1]]))

    act_X = fs.min_max(X)
    act_Y = fs.min_max(Y)
    assert np.all(act_X == pytest.approx(exp_X))
    assert np.all(act_Y == pytest.approx(exp_Y))


def test_mean_norm():
    X = np.transpose([[1, 2, 5, 7, 8, 9, 10],
                      [7, 8, 9, 10, 11, 12, 20]])
    Y = np.array([3, 4, 5, 1, 2, 2, 4])

    exp_X1 = np.array([-5, -4, -1, 1, 2, 3, 4]) / 9
    exp_X2 = np.array([-4, -3, -2, -1, 0, 1, 9]) / 13
    exp_X = np.transpose([exp_X1, exp_X2])
    exp_Y = np.array([0, 1, 2, -2, -1, -1, 1]) / 4

    with pytest.raises(ValueError):
        fs.mean_norm(np.array([[0, 0], [0, 1]]))

    act_X = fs.mean_norm(X)
    act_Y = fs.mean_norm(Y)
    assert np.all(act_X == pytest.approx(exp_X))
    assert np.all(act_Y == pytest.approx(exp_Y))


def test_zscore():
    X = np.transpose([[1, 2, 5, 7, 8, 9, 10],
                      [7, 8, 9, 10, 11, 12, 20]])
    Y = np.array([3, 4, 5, 1, 2, 2, 4])

    exp_X1 = np.array([-5, -4, -1, 1, 2, 3, 4]) / 3.464101615
    exp_X2 = np.array([-4, -3, -2, -1, 0, 1, 9]) / 4.320493799
    exp_X = np.transpose([exp_X1, exp_X2])
    exp_Y = np.array([0, 1, 2, -2, -1, -1, 1]) / 1.414213562

    with pytest.raises(ValueError):
        fs.zscore(np.array([[0, 0], [0, 1]]))

    act_X = fs.zscore(X)
    act_Y = fs.zscore(Y)
    assert np.all(act_X == pytest.approx(exp_X))
    assert np.all(act_Y == pytest.approx(exp_Y))


ltr()
