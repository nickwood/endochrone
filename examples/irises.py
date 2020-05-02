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
    from endochrone.clustering import naive_k_means as nkm
    from endochrone.stats import scaling as fs

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
    from endochrone.stats import scaling as fs
    from endochrone.classification import naive_knn as knn

    s_i_data = fs.mean_norm(i_data)
    accuracy = []

    for i in range(n_runs):
        s_Xtr, s_Xtest, Ytr, Ytest = train_test_split(s_i_data,
                                                      i_target[:, np.newaxis])
        ypred = knn.classify(s_Xtr, Ytr, s_Xtest, k=3)
        accuracy.append(accuracy_score(Ytest, ypred))
    print("mean accuracy over %s runs is %.4f" % (n_runs, np.mean(accuracy)))


def iris_pca():
    from mpl_toolkits import mplot3d  # noqa: F401
    from endochrone.decomposition import pca

    fig = plt.figure(facecolor="w", figsize=(14, 7))
    i_target = iris['target']

    pcam_2 = pca.PCA(n_components=2)
    pcam_2.fit(i_data)
    var_sum_2 = np.abs(np.sum(pcam_2.explained_variance_ratio_))
    title_2 = '%s components, capturing %.4f%% variation' % (2, var_sum_2*100)
    red_i_data_2 = pcam_2.transform(i_data)

    X_2 = red_i_data_2[:, 0]
    Y_2 = red_i_data_2[:, 1]
    ax_2 = fig.add_subplot(1, 2, 1, title=title_2)
    ax_2.scatter(X_2, Y_2, c=i_target, s=3, marker='d', cmap='cool')

    # Compare to 3 component reduction
    pcam_3 = pca.PCA(n_components=3)
    pcam_3.fit(i_data)
    var_sum_3 = np.abs(np.sum(pcam_3.explained_variance_ratio_))
    title_3 = '%s components, capturing %.4f%% variation' % (2, var_sum_3*100)
    red_i_data_3 = pcam_3.transform(i_data)

    X_3 = red_i_data_3[:, 0]
    Y_3 = red_i_data_3[:, 1]
    Z_3 = red_i_data_3[:, 2]
    ax_3 = fig.add_subplot(1, 2, 2, projection='3d', title=title_3)
    ax_3.scatter3D(X_3, Y_3, Z_3, c=i_target, s=3, marker='d', cmap='cool')

    plt.show()


def iris_bdt():
    from sklearn.model_selection import train_test_split
    from endochrone.classification.binary_decision_tree\
        import BinaryDecisionTree
    from endochrone.stats.metrics import MulticlassMetrics

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(i_data, i_target)
    i_bdt = BinaryDecisionTree(max_depth=4)
    i_bdt.fit(Xtrain, Ytrain)
    print(i_bdt)

    Ypred = i_bdt.predict(Xtest)

    metrics = MulticlassMetrics(Ytest, Ypred)
    print("mac_prec:", metrics.macro_precision)
    print("mac_recall:", metrics.macro_recall)
    print("mac_f1:", metrics.macro_f1_score)
    print("mic_ prec:", metrics.micro_precision)


iris_k_means()
iris_naive_knn()
iris_pca()
iris_bdt()
