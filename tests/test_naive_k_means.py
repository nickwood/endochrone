# -*- coding: utf-8 -*-
# from itertools import combinations
import random
import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns
from sklearn import datasets

from endochrone import naive_k_means as nkm

from pprint import pprint as pp

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
        pp(X)
        # pp(Y)
        print(naive_k_means.calculate(X, Y, 3))


def test_forgy_initialisation():
    with pytest.raises(ValueError):
        # too many columns
        assert nkm.initial_means(np.array([[1, 2, 3], [3, 4, 5]]))
        # too few columns
        assert nkm.initial_means(np.array([[1], [3], [4], [5]]))
        # need at least n rows
        assert nkm.initial_means(np.array([[1, 2], [3, 4]]))

    data = np.transpose([random.sample(range(200), 20),
                         random.sample(range(200), 20)])

    # check default is k = 3
    assert nkm.initial_means(data).shape == (3, 2)

    for n in [2, 8, 15]:
        means = nkm.initial_means(data, n)
        assert means.shape == (n, 2)
        for i in range(n):
            assert means[i] in data


# test_forgy_initialisation()
# test_irises()
