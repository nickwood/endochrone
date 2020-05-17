# -*- coding: utf-8 -*-
import numpy as np
import pytest
import random
from unittest.mock import Mock

from endochrone.clustering import KMeans
from endochrone.utils import lazy_test_runner as ltr

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_given_initials():
    X = np.transpose([[1, 2, 3, 7, 8, 9, 10], [3, 4, 5, 1, 2, 3, 4]])
    init_centroids = np.array([[1, 3], [7, 1]])

    km_test = KMeans(k=2)
    km_test.forgy_centroids_ = Mock(wraps=km_test.forgy_centroids_)

    km_test.fit(features=X, initial_centroids=init_centroids)
    assert not km_test.forgy_centroids_.called

    act = km_test.centroids
    exp = np.array([[2, 4], [8.5, 2.5]])
    assert np.all(act == pytest.approx(exp))


def test_calculate_step():
    X = np.transpose([[1, 2, 3, 7, 8, 9], [3, 4, 5, 1, 2, 3]])

    km_test = KMeans(k=2)
    km_test.centroids = np.array([[1, 7], [9, 3]])

    act = km_test.calculate_step(features=X)
    exp = np.array([[2, 4], [8, 2]])

    assert np.all(act == pytest.approx(exp))


def test_forgy_initialisation():
    X = np.transpose([random.sample(range(200), 20),
                      random.sample(range(200), 20)])

    km_test = KMeans()
    km_test.forgy_centroids_ = Mock(wraps=km_test.forgy_centroids_)

    # check defaults
    assert km_test.n_centroids_ == 3
    km_test.fit(features=X)
    km_test.forgy_centroids_.assert_called()
    centroids = km_test.forgy_centroids_(features=X)

    assert centroids.shape == (3, 2)
    assert np.unique(centroids, axis=0).shape == (3, 2)
    for i in range(3):
        assert centroids[i] in X

    # tests with k = 5, 12
    for N in [5, 12]:
        km_test2 = KMeans(k=N)
        km_test2.forgy_centroids_ = Mock(wraps=km_test2.forgy_centroids_)
        assert km_test2.n_centroids_ == N
        km_test2.fit(features=X)
        km_test2.forgy_centroids_.assert_called()

        centroids = km_test2.forgy_centroids_(features=X)
        assert centroids.shape == (N, 2)
        assert np.unique(centroids, axis=0).shape == (N, 2)
        for i in range(N):
            assert centroids[i] in X


def test_nearest_centroid():
    km_test = KMeans(k=2)
    km_test.centroids = np.array([[2, 4], [8, 2]])
    assert km_test.nearest_centroid(point=np.array([0, 0])) == 0
    assert km_test.nearest_centroid(point=np.array([5, 3.1])) == 0
    assert km_test.nearest_centroid(point=np.array([5.1, 3])) == 1
    assert km_test.nearest_centroid(point=np.array([10.1, 0])) == 1


def test_nearest_centroids():
    X = np.transpose([[1, 2, 3, 7, 8, 9], [3, 4, 5, 1, 2, 3]])
    km_test = KMeans(k=2)
    km_test.centroids = np.array([[2, 4], [8, 2]])

    exp = np.array([0, 0, 0, 1, 1, 1])
    act = km_test.nearest_centroids(features=X)
    assert np.all(act == exp)


def test_recalculate_centroids():
    X = np.transpose([[1, 2, 3, 7, 8, 9], [3, 4, 5, 1, 2, 3]])

    assignments = np.transpose([0, 0, 0, 1, 1, 1])
    assignments2 = np.transpose([0, 0, 1, 1, 2, 2])

    km_test = KMeans(k=2)
    act = km_test.recalculate_centroids(features=X, assignments=assignments)
    exp = np.array([[2, 4], [8, 2]])
    assert np.all(act == pytest.approx(exp))

    km_test2 = KMeans(k=3)
    act = km_test2.recalculate_centroids(features=X, assignments=assignments2)
    exp = np.array([[1.5, 3.5], [5, 3], [8.5, 2.5]])
    assert np.all(act == pytest.approx(exp))


def test_predict():
    X = np.transpose([[1, 2, 3, 7, 8, 9], [3, 4, 5, 1, 2, 3]])
    km_test = KMeans(k=2)
    km_test.fit(features=X, initial_centroids=np.array([[1, 3], [9, 3]]))

    exp_cents = np.array([[2, 4], [8, 2]])
    assert np.all(km_test.centroids == exp_cents)

    x_pred = np.transpose([[2, 4, 3, 6, 10, 8], [6, 4, 3, 0, 2, 2]])
    y_pred = km_test.predict(features=x_pred)
    y_true = np.array([0, 0, 0, 1, 1, 1])
    assert np.all(y_pred == y_true)


ltr()
