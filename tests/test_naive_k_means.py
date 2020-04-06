# -*- coding: utf-8 -*-
# from itertools import combinations
import random
# import matplotlib.pyplot as plt
import numpy as np
import pytest
# import seaborn as sns
# from sklearn import datasets

from endochrone import naive_k_means as nkm

from pprint import pprint as pp

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_irises():
    # iris = datasets.load_iris()
    # iris_data = np.array(iris['data'])
    assert True
    # pair_plot = False
    # if pair_plot:
    #     iris_df = sns.load_dataset("iris")
    #     sns.pairplot(iris_df, hue='species')

    # i = 0
    # # number_features = iris_data.shape[1]
    # # for axis1, axis2 in combinations(range(number_features), 2):
    # for axis1, axis2 in [(0, 3)]:
    #     plt.figure(i)
    #     i += 1
    #     data = np.transpose(iris_data[:, axis1:axis1 + 2])
    #     # Y = np.transpose(iris_data[:, axis2:axis2 + 1])
    #     # pp(X)
    #     pp(Y)
    #     print(nkm.calculate(data, 3))


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


def test_euclidian_dist():
    A = np.array([0, 0])
    B = np.array([3, 4])
    C = np.array([12, 10])
    D = np.array([0, 3])
    E = np.array([0.7, 2.8])

    assert nkm.euclidean_dist(A, B)**2 == pytest.approx(25.0)
    assert nkm.euclidean_dist(B, C)**2 == pytest.approx(117.0)
    assert nkm.euclidean_dist(C, D)**2 == pytest.approx(193.0)
    assert nkm.euclidean_dist(A, C)**2 == pytest.approx(244.0)
    assert nkm.euclidean_dist(B, D)**2 == pytest.approx(10.0)
    assert nkm.euclidean_dist(A, D)**2 == pytest.approx(9.0)
    assert nkm.euclidean_dist(B, E)**2 == pytest.approx(6.73)


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


test_forgy_initialisation()
test_euclidian_dist()
test_nearest_centroid()
test_nearest_centroids()
test_recalculate_centroids()
# test_irises()
