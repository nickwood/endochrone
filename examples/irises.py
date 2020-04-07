from itertools import combinations
from math import factorial as fac
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets

from endochrone import naive_k_means as nkm
from endochrone import feature_scaling as fs


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
    X = np.transpose([iris_data[:, ax1]])
    Y = np.transpose([iris_data[:, ax2]])

    # 'r_' denotes 'raw' space, 's_' denotes scaled space
    r_data = np.concatenate([X, Y], axis=1)
    s_data = fs.mean_norm(r_data)
    # s_data = fs.min_max(r_data)

    s_centroids = nkm.calculate(s_data, 3)
    assignments = nkm.nearest_centroids(s_data, s_centroids)
    r_centroids = nkm.recalculate_centroids(r_data, assignments, 3)

    plt.scatter(X, Y, c=assignments, s=3, marker='d', cmap='cool')
    plt.scatter(r_centroids[:, 0], r_centroids[:, 1], marker='o', c="black")
    i += 1

plt.show()
