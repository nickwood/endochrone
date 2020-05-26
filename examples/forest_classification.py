import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import time

from endochrone.ensemble.random_forest import RandomForest
from endochrone.classification import binary_tree as bdt
from endochrone.stats.metrics import MulticlassMetrics as mcm

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


source_dict = fetch_covtype()
x = source_dict['data']
y = source_dict['target']

# Changing this to False wil run for the entire ~500k dataset
test_fit = True
if test_fit:
    sub_size = 100
    np.random.seed(seed=1234)
    indexes = np.random.choice(range(0, x.shape[0]), sub_size, replace=False)
    x = x[indexes, :]
    y = y[indexes]


def without_pca():

    xtrain, xtest, ytrain, ytest = train_test_split(x, y)

    t0 = time.process_time()
    forest_model = RandomForest(16, max_tree_depth=12)
    forest_model.fit(xtrain, ytrain)
    t1 = time.process_time()
    print('train time: ', t1-t0)

    ypred = forest_model.predict(xtest)
    metrics = mcm(ytest, ypred)
    print('\nmetrics for test set')
    print('macro_precision', metrics.macro_precision)
    print('macro_recall', metrics.macro_recall)
    print('macro_f1', metrics.macro_f1_score)
    print('micro_f1', metrics.micro_f1_score)

    train_pred = forest_model.predict(xtrain)
    metrics = mcm(ytrain, train_pred)
    print('\nmetrics for training set')
    print('macro_precision', metrics.macro_precision)
    print('macro_recall', metrics.macro_recall)
    print('macro_f1', metrics.macro_f1_score)
    print('micro_f1', metrics.micro_f1_score)


def pca_and_pair_plot():
    from endochrone.stats.scaling import FeatureScaling
    from endochrone.decomposition import PCA

    global x
    if test_fit:
        # TODO: make this part of PCA
        # We need to remove features with no variation, otherwise scaling & PCA
        # will fail
        features_to_keep = [i for i, feat in enumerate(x.T)
                            if len(np.unique(feat)) > 1]
        x = x[:, features_to_keep]

    # first we rescale to stop large floats dominating categories
    scale_model = FeatureScaling(method='z_score')
    scaled_x = scale_model.fit_and_transform(features=x)

    N = 3
    pcam_min = PCA(n_components=N)
    pcam_min.fit(features=scaled_x)
    pca_x = pcam_min.transform(features=scaled_x)
    labels = (list(range(N)) + ['species'])
    df = pd.DataFrame(np.hstack([pca_x, y[:, np.newaxis]]), columns=labels)
    sns.pairplot(df, hue='species', height=1.5)
    plt.show()


def with_pca():
    from endochrone.stats.scaling import FeatureScaling
    from endochrone.decomposition import PCA

    global x
    if test_fit:
        # TODO: make this part of PCA
        # We need to remove features with no variation, otherwise scaling & PCA
        # will fail
        features_to_keep = [i for i, feat in enumerate(x.T)
                            if len(np.unique(feat)) > 1]
        x = x[:, features_to_keep]

    # first we rescale to stop large floats dominating categories
    scale_model = FeatureScaling(method='z_score')
    scaled_x = scale_model.fit_and_transform(features=x)

    # Then build a test PCA model so we can figure out how many components we
    # want
    pcam_test = PCA()
    pcam_test.fit(features=scaled_x)
    cutoff = 0.97  # i.e. we want to retain this % of variance
    # TODO: want to be able to do this with a single PCA model
    n_comp = np.argmax(np.cumsum(pcam_test.explained_variance_ratio_) > cutoff)

    # Optionally see how our variance increases as n_comp increases
    show_pca_variances = False
    if show_pca_variances:
        plt.plot(range(54), np.cumsum(pcam_test.explained_variance_ratio_))
        plt.show()

    # Transform according to the above PCA
    pcam_forest = PCA(n_components=n_comp)
    pca_x = pcam_forest.fit_and_transform(features=scaled_x)

    xtrain, xtest, ytrain, ytest = train_test_split(pca_x, y)

    # Now build our RF model and fit it
    t0 = time.process_time()
    forest_model = RandomForest(64, max_tree_depth=15)
    forest_model.fit(xtrain, ytrain, debug=True)
    t1 = time.process_time()
    print('train time: ', t1-t0)

    # print(forest_model.trees[0])
    ypred = forest_model.predict(xtest)
    metrics = mcm(ytest, ypred)
    print('\nmetrics for test set')
    print('macro_precision', metrics.macro_precision)
    print('macro_recall', metrics.macro_recall)
    print('macro_f1', metrics.macro_f1_score)
    print('micro_f1', metrics.micro_f1_score)

    train_pred = forest_model.predict(xtrain)
    metrics = mcm(ytrain, train_pred)
    print('\nmetrics for training set')
    print('macro_precision', metrics.macro_precision)
    print('macro_recall', metrics.macro_recall)
    print('macro_f1', metrics.macro_f1_score)
    print('micro_f1', metrics.micro_f1_score)

    """Example output:
    tree fitted in 167.765625 seconds
    tree fitted in 164.625000 seconds
    ...
    tree fitted in 136.921875 seconds
    tree fitted in 167.062500 seconds

    train time:  9981.1875

    metrics for test set
    macro_precision 0.8226020096015664
    macro_recall 0.6686569668927956
    macro_f1 0.7376834920962251
    micro_f1 0.8126923368192052

    metrics for training set
    macro_precision 0.8352495807352573
    macro_recall 0.6788309239488616
    macro_f1 0.7489604982883065
    micro_f1 0.8194690184253223
    """


def visualise_entropies():
    feat_idx = 0
    entropies = [bdt.weighted_partition_entropy(p1, p2)
                 for p1, p2 in bdt.generate_partitions(x[:, feat_idx], y)]
    plt.plot(range(len(entropies)), entropies)
    plt.show()


# without_pca()
# pca_and_pair_plot()
with_pca()
# visualise_entropies()
