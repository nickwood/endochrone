import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


iris = datasets.load_iris()
i_data = np.array(iris['data'])
i_target = np.array(iris['target'])
axes = iris['feature_names']


def iris_k_means():
    from itertools import combinations
    from math import factorial as fac
    from endochrone.clustering import KMeans
    from endochrone.stats.scaling import FeatureScaling

    num_features = i_data.shape[1]
    num_charts = fac(num_features) // fac(2) // fac(num_features-2)
    plt.figure(facecolor="w", figsize=(15, 10))

    i = 1
    for ax1, ax2 in combinations(range(num_features), 2):
        plt.subplot(2, num_charts // 2, i, xlabel=axes[ax1], ylabel=axes[ax2])
        X = i_data[:, ax1][:, np.newaxis]
        Y = i_data[:, ax2][:, np.newaxis]

        # 'r_' denotes 'raw' space, 's_' denotes scaled space
        r_data = np.concatenate([X, Y], axis=1)
        scale_model = FeatureScaling(method='mean_norm')
        s_data = scale_model.fit_and_transform(features=r_data)

        kmeans_model = KMeans(k=3)
        kmeans_model.fit(features=s_data)
        assignments = kmeans_model.nearest_centroids(features=s_data)
        s_centroids = kmeans_model.centroids
        r_centroids = scale_model.reverse(features=s_centroids)

        plt.scatter(X, Y, c=assignments, s=3, marker='d', cmap='cool')
        plt.scatter(r_centroids[:, 0], r_centroids[:, 1], marker='o', c="b")
        i += 1

    plt.show()


def iris_naive_knn(n_runs=10):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from endochrone.stats.scaling import FeatureScaling
    from endochrone.classification import KNearest

    scale_model = FeatureScaling(method='z_score')
    s_i_data = scale_model.fit_and_transform(features=i_data)
    accuracy = []

    for i in range(n_runs):
        s_Xtr, s_Xtest, Ytr, Ytest = train_test_split(s_i_data, i_target)
        knn_model = KNearest()
        knn_model.fit(features=s_Xtr, targets=Ytr)
        ypred = knn_model.predict(features=s_Xtest)
        accuracy.append(accuracy_score(Ytest, ypred))
    print("mean accuracy over %s runs is %.4f" % (n_runs, np.mean(accuracy)))


def iris_pca():
    from mpl_toolkits import mplot3d  # noqa: F401
    from endochrone.decomposition import PCA

    fig = plt.figure(facecolor="w", figsize=(14, 7))
    i_target = iris['target']

    pcam_2 = PCA(n_components=2)
    pcam_2.fit(features=i_data)
    var_sum_2 = np.abs(np.sum(pcam_2.explained_variance_ratio_))
    title_2 = '%s components, capturing %.4f%% variation' % (2, var_sum_2*100)
    red_i_data_2 = pcam_2.transform(features=i_data)

    X_2 = red_i_data_2[:, 0]
    Y_2 = red_i_data_2[:, 1]
    ax_2 = fig.add_subplot(1, 2, 1, title=title_2)
    ax_2.scatter(X_2, Y_2, c=i_target, s=3, marker='d', cmap='cool')

    # Compare to 3 component reduction
    pcam_3 = PCA(n_components=3)
    pcam_3.fit(features=i_data)
    var_sum_3 = np.abs(np.sum(pcam_3.explained_variance_ratio_))
    title_3 = '%s components, capturing %.4f%% variation' % (2, var_sum_3*100)
    red_i_data_3 = pcam_3.transform(features=i_data)

    X_3 = red_i_data_3[:, 0]
    Y_3 = red_i_data_3[:, 1]
    Z_3 = red_i_data_3[:, 2]
    ax_3 = fig.add_subplot(1, 2, 2, projection='3d', title=title_3)
    ax_3.scatter3D(X_3, Y_3, Z_3, c=i_target, s=3, marker='d', cmap='cool')

    plt.show()


def iris_bdt():
    from sklearn.model_selection import train_test_split
    from endochrone.classification import BinaryDecisionTree
    from endochrone.stats.metrics import MulticlassMetrics

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(i_data, i_target)
    i_bdt = BinaryDecisionTree(max_depth=4)
    i_bdt.fit(Xtrain, Ytrain)
    print(i_bdt)

    Ypred = i_bdt.predict(Xtest)

    metrics = MulticlassMetrics(Ytest, Ypred)
    print(metrics)


# iris_k_means()
# iris_naive_knn()
iris_pca()
# iris_bdt()
