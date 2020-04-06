from itertools import combinations
from math import factorial as fac
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets

from endochrone import naive_k_means as nkm

iris = datasets.load_iris()
iris_data = np.array(iris['data'])
axes = iris['feature_names']
pair_plot = False

if pair_plot:
    iris_df = sns.load_dataset("iris")
    sns.pairplot(iris_df, hue='species')

num_features = iris_data.shape[1]
num_charts = fac(num_features) // fac(2) // fac(num_features-2)
plt.figure(facecolor="w", figsize=(15, 10))

i = 1
for ax1, ax2 in combinations(range(num_features), 2):
    plt.subplot(2, num_charts // 2, i, xlabel=axes[ax1], ylabel=axes[ax2])
    X = np.array(iris_data[:, ax1:ax1 + 1])
    Y = np.array(iris_data[:, ax2:ax2 + 1])
    data = np.concatenate([X, Y], axis=1)
    centroids = nkm.calculate(data, 3)
    assignments = nkm.nearest_centroids(data, centroids)

    plt.scatter(X, Y, c=assignments, s=3, marker='d', cmap='cool')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', c="black")
    i += 1

plt.show()
