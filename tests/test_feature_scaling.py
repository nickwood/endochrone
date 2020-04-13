# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone import feature_scaling as fs
from endochrone.misc import lazy_test_runner as ltr

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_min_max():
    X = np.transpose([[1, 2, 3, 7, 8, 9, 10]])
    Y = np.transpose([[3, 4, 5, 1, 2, 3, 4]])
    data = np.concatenate([X, Y], axis=1)

    X_scaled = np.transpose([[0, 1, 2, 6, 7, 8, 9]]) / 9
    Y_scaled = np.transpose([[2, 3, 4, 0, 1, 2, 3]]) / 4
    exp = np.concatenate([X_scaled, Y_scaled], axis=1)

    act = fs.min_max(data)
    assert np.all(act == pytest.approx(exp))


def test_mean_norm():
    X = np.transpose([[1, 2, 5, 7, 8, 9, 10]])
    Y = np.transpose([[3, 4, 5, 1, 2, 2, 4]])
    data = np.concatenate([X, Y], axis=1)

    X_scaled = np.transpose([[-5, -4, -1, 1, 2, 3, 4]]) / 9
    Y_scaled = np.transpose([[0, 1, 2, -2, -1, -1, 1]]) / 4
    exp = np.concatenate([X_scaled, Y_scaled], axis=1)

    act = fs.mean_norm(data)
    assert np.all(act == pytest.approx(exp))


ltr()
