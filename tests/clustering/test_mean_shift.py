# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone.utils.misc import lazy_test_runner as ltr
from endochrone.clustering import mean_shift as ms

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_neighbours():
    X = np.arange(0, 10, 1)
    assert np.all(ms.neighbours(3.5, X, 2) == np.array([2, 3, 4, 5]))
    assert np.all(ms.neighbours(5, X, 3) == np.array([2, 3, 4, 5, 6, 7, 8]))

    X2d = np.arange(0, 20, 1).reshape(10, 2)
    exp = np.array([[0, 1], [2, 3], [4, 5]])
    assert np.all(ms.neighbours([3.5, 2.3], X2d, 4.5) == exp)

    X3d = np.arange(0, 30, 1).reshape(10, 3)
    exp = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert np.all(ms.neighbours([3.5, 2.3, 4.2], X3d, 8.5) == exp)


def test_flat_kernel():
    X = np.arange(0, 10)
    p = np.array([4.5])
    assert ms.flat(X, p, 2) == [4.5]

    X2d = np.arange(0, 20).reshape(10, 2)
    p = np.array([9, 9])
    assert np.all(ms.flat(X2d, p, 3) == np.array([9, 10]))

    X3d = np.arange(0, 30).reshape(10, 3)
    p = np.array([22, 23, 24])
    assert np.all(ms.flat(X3d, p, 4) == np.array([22.5, 23.5, 24.5]))


def test_gaussian_kernel():
    X1 = np.arange(1, 4)
    p = np.array([2])
    assert ms.gaussian(X1, p, 1) == 2
    assert ms.gaussian_1d(X1, p, 1) == pytest.approx(2)

    X2d = np.arange(0, 20).reshape(10, 2)
    p = np.array([9, 9])
    exp = np.array([8.88934389, 9.88934389])
    assert np.all(ms.gaussian_2d(X2d, p, 3) == pytest.approx(exp))
    assert np.all(ms.gaussian(X2d, p, 3) == pytest.approx(exp))
    exp = np.array([8.15192519, 9.15192519])
    assert np.all(ms.gaussian_2d(X2d, p, 4) == pytest.approx(exp))
    assert np.all(ms.gaussian(X2d, p, 4) == pytest.approx(exp))

    with pytest.raises(NotImplementedError):
        X3d = np.arange(0, 8).reshape(2, 2, 2)
        p = np.array([[1, 1], [1, 1]])
        ms.gaussian(X3d, p, 4)


def test_simple_fit():
    X1 = np.arange(0, 6, 1)
    clusters = ms.MeanShift(bandwidth=2.5)
    centres1, labels1 = clusters.fit(X1)
    assert centres1 == np.array([2.5])
    assert labels1 == np.array([0])

    X2 = np.hstack([np.arange(0, 3, 1), np.arange(6, 9, 1)])
    clusters = ms.MeanShift(bandwidth=2)
    centres2, labels2 = clusters.fit(X2)
    assert np.all(centres2 == np.array([1, 7]))
    assert np.all(labels2 == np.array([0, 1]))


ltr()
