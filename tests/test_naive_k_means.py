# -*- coding: utf-8 -*-
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from endochrone import naive_k_means
# from itertools import combinations
import random
# from pprint import pprint as pp

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_irises():
    iris = datasets.load_iris()
    iris_data = np.array(iris['data'])

    pair_plot = False
    if pair_plot:
        iris_df = sns.load_dataset("iris")
        sns.pairplot(iris_df, hue='species')

    i = 0
    # number_features = iris_data.shape[1]
    # for axis1, axis2 in combinations(range(number_features), 2):
    for axis1, axis2 in [(0, 3)]:
        plt.figure(i)
        i += 1
        X = np.transpose(iris_data[:, axis1:axis1 + 1])
        Y = np.transpose(iris_data[:, axis2:axis2 + 1])
        # pp(X)
        # pp(Y)
        print(naive_k_means.calculate(X, Y, 3))


def test_forgy_initialisation():
    X = np.array(random.sample(range(200), 20))
    Y = np.array(random.sample(range(200), 20))

    # check default is k = 3
    assert naive_k_means.initial_means(X, Y).shape == (2, 3)

    for n in [2, 8, 15]:
        means = naive_k_means.initial_means(X, Y, n)
        assert means.shape == (2, n)
        assert len(set(means[0])) == n
        assert len(set(means[1])) == n
        for i in range(n):
            assert np.where(X == means[0][i]) == np.where(Y == means[1][i])


test_forgy_initialisation()
# test_irises()
