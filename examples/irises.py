# -*- coding: utf-8 -*-
# from itertools import combinations
# import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
# from pprint import pprint as pp
import numpy as np

from endochrone import naive_k_means as nkm


def test_irises():
    iris = datasets.load_iris()
    iris_data = np.array(iris['data'])
    pair_plot = False

    if pair_plot:
        iris_df = sns.load_dataset("iris")
        sns.pairplot(iris_df, hue='species')

    # i = 0
    # # number_features = iris_data.shape[1]
    # # for axis1, axis2 in combinations(range(number_features), 2):
    for axis1, axis2 in [(0, 3)]:
        # plt.figure(i)
        # i += 1
        data = np.array(iris_data[:, axis1:axis1 + 2])
        # pp(data)
        # Y = np.transpose(iris_data[:, axis2:axis2 + 1])
        # pp(X)
        # pp(Y)
        print(nkm.calculate(data, 3))
