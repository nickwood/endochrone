from itertools import combinations
from math import factorial as fac
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

from endochrone import naive_k_means as nkm
from endochrone import feature_scaling as fs


iris = datasets.load_iris()
iris_data = np.array(iris['data'])
axes = iris['feature_names']


def iris_k_means():
    num_features = iris_data.shape[1]
    num_charts = fac(num_features) // fac(2) // fac(num_features-2)
    plt.figure(facecolor="w", figsize=(15, 10))

    i = 1
    for ax1, ax2 in combinations(range(num_features), 2):
        plt.subplot(2, num_charts // 2, i, xlabel=axes[ax1], ylabel=axes[ax2])
        X = iris_data[:, ax1][:, np.newaxis]
        Y = iris_data[:, ax2][:, np.newaxis]

        # 'r_' denotes 'raw' space, 's_' denotes scaled space
        r_data = np.concatenate([X, Y], axis=1)
        s_data = fs.mean_norm(r_data)
        # s_data = fs.min_max(r_data)

        s_centroids = nkm.calculate(s_data, 3)
        assignments = nkm.nearest_centroids(s_data, s_centroids)
        r_centroids = nkm.recalculate_centroids(r_data, assignments, 3)

        plt.scatter(X, Y, c=assignments, s=3, marker='d', cmap='cool')
        plt.scatter(r_centroids[:, 0], r_centroids[:, 1], marker='o', c="b")
        i += 1

    plt.show()


def iris_naive_knn(n_runs=10):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from endochrone import naive_knn as knn
    iris_targets = iris['target'][:, np.newaxis]
    s_iris_data = fs.mean_norm(iris_data)
    accuracy = []

    for i in range(n_runs):
        s_Xtrain, s_Xtest, ytrain, ytest = train_test_split(s_iris_data,
                                                            iris_targets)
        ypred = knn.classify(s_Xtrain, ytrain, s_Xtest, k=3)
        accuracy.append(accuracy_score(ytest, ypred))
    print("mean accuracy over %s runs is %.4f" % (n_runs, np.mean(accuracy)))


iris_naive_knn(n_runs=15)
# iris_k_means()
