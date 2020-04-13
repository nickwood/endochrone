# -*- coding: utf-8 -*-
import numpy as np
import pytest
import random

from endochrone import naive_k_means as nkm
from endochrone.misc import lazy_test_runner as ltr

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_calculate():
    X = np.transpose([[1, 2, 3, 7, 8, 9, 10]])
    Y = np.transpose([[3, 4, 5, 1, 2, 3, 4]])

    data = np.concatenate([X, Y], axis=1)
    centroids = np.array([[1, 3], [7, 1]])

    act = nkm.calculate(data, k=2, centroids=centroids)
    exp = np.array([[2, 4], [8.5, 2.5]])
    assert np.all(act == pytest.approx(exp))

    act = nkm.calculate(data, k=2)
    exp = np.array([[2, 4], [8.5, 2.5]])
    assert np.all(act == pytest.approx(exp))


def test_calculate_step():
    X = np.transpose([[1, 2, 3, 7, 8, 9]])
    Y = np.transpose([[3, 4, 5, 1, 2, 3]])

    data = np.concatenate([X, Y], axis=1)
    centroids = np.array([[1, 7], [9, 3]])

    act = nkm.calculate_step(data, centroids, k=2)
    exp = np.array([[2, 4], [8, 2]])

    assert np.all(act == pytest.approx(exp))


def test_forgy_initialisation():
    with pytest.raises(ValueError):
        # too many columns
        assert nkm.initial_centroids(np.array([[1, 2, 3], [3, 4, 5]]))
        # too few columns
        assert nkm.initial_centroids(np.array([[1], [3], [4], [5]]))
        # need at least n rows
        assert nkm.initial_centroids(np.array([[1, 2], [3, 4]]))

    data = np.transpose([random.sample(range(200), 20),
                         random.sample(range(200), 20)])

    # check default is k = 3
    assert nkm.initial_centroids(data).shape == (3, 2)

    for n in [2, 8, 15]:
        centroids = nkm.initial_centroids(data, n)
        assert centroids.shape == (n, 2)
        for i in range(n):
            assert centroids[i] in data


def test_nearest_centroid():
    centroids = np.array([[2, 4], [8, 2]])
    assert nkm.nearest_centroid([0, 0], centroids) == 0
    assert nkm.nearest_centroid([5, 3.1], centroids) == 0
    assert nkm.nearest_centroid([5.1, 3], centroids) == 1
    assert nkm.nearest_centroid([10.1, 0], centroids) == 1


def test_nearest_centroids():
    X = np.transpose([[1, 2, 3, 7, 8, 9]])
    Y = np.transpose([[3, 4, 5, 1, 2, 3]])

    data = np.concatenate([X, Y], axis=1)
    centroids = np.array([[2, 4], [8, 2]])

    exp = np.transpose([[0, 0, 0, 1, 1, 1]])
    act = nkm.nearest_centroids(data, centroids)
    assert np.all(act == exp)
    assert np.all(data == np.concatenate([X, Y], axis=1))


def test_recalculate_centroids():
    X = np.transpose([[1, 2, 3, 7, 8, 9]])
    Y = np.transpose([[3, 4, 5, 1, 2, 3]])

    data = np.concatenate([X, Y], axis=1)
    assignments = np.transpose([[0, 0, 0, 1, 1, 1]])
    assignments_2 = np.transpose([[0, 0, 1, 1, 2, 2]])

    act = nkm.recalculate_centroids(data, assignments, k=2)
    exp = np.array([[2, 4], [8, 2]])
    assert np.all(act == pytest.approx(exp))

    act = nkm.recalculate_centroids(data, assignments_2, k=3)
    exp = np.array([[1.5, 3.5], [5, 3], [8.5, 2.5]])
    assert np.all(act == pytest.approx(exp))
    assert np.all(data == np.concatenate([X, Y], axis=1))


ltr()
