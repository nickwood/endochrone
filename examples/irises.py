import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


iris = datasets.load_iris()
iris_data = np.array(iris['data'])
axes = iris['feature_names']


def iris_k_means():
    from itertools import combinations
    from math import factorial as fac
    from endochrone import naive_k_means as nkm
    from endochrone import feature_scaling as fs

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
    from endochrone import feature_scaling as fs
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


def iris_pca():
    from mpl_toolkits import mplot3d  # noqa: F401
    from endochrone import pca

    fig = plt.figure(facecolor="w", figsize=(14, 7))
    iris_labels = iris['target']

    pcam_2 = pca.PCA(n_components=2)
    pcam_2.fit(iris_data)
    var_sum_2 = np.abs(np.sum(pcam_2.explained_variance_ratio_))
    title_2 = '%s components, capturing %.4f%% variation' % (2, var_sum_2*100)
    red_iris_data_2 = pcam_2.transform(iris_data)

    X_2 = red_iris_data_2[:, 0]
    Y_2 = red_iris_data_2[:, 1]
    ax_2 = fig.add_subplot(1, 2, 1, title=title_2)
    ax_2.scatter(X_2, Y_2, c=iris_labels, s=3, marker='d', cmap='cool')

    # Compare to 3 component reduction
    pcam_3 = pca.PCA(n_components=3)
    pcam_3.fit(iris_data)
    var_sum_3 = np.abs(np.sum(pcam_3.explained_variance_ratio_))
    title_3 = '%s components, capturing %.4f%% variation' % (2, var_sum_3*100)
    red_iris_data_3 = pcam_3.transform(iris_data)

    X_3 = red_iris_data_3[:, 0]
    Y_3 = red_iris_data_3[:, 1]
    Z_3 = red_iris_data_3[:, 2]
    ax_3 = fig.add_subplot(1, 2, 2, projection='3d', title=title_3)
    ax_3.scatter3D(X_3, Y_3, Z_3, c=iris_labels, s=3, marker='d', cmap='cool')

    plt.show()


# iris_k_means()
# iris_naive_knn()
iris_pca()
