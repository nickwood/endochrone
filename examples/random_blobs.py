import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from endochrone.classification.binary_tree import BinaryDecisionTree
from endochrone.classification.naive_bayes import NaiveBayes
from endochrone.stats import scaling as fs
from endochrone.stats.metrics import MulticlassMetrics as mcm

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def binary_classifier(plot=True):
    x, y = make_blobs(n_samples=150, n_features=2, random_state=1234,
                      cluster_std=1.5, centers=3)

    bdt_test = BinaryDecisionTree(max_depth=3)
    bdt_test.fit(x, y)
    print(bdt_test)
    if plot:
        plt.scatter(x[:, 0], x[:, 1], c=y)
        plt.show()


def naive_bayes(plot=True):
    x, y = make_blobs(n_samples=4500, n_features=5, cluster_std=3, centers=15)

    x = fs.mean_norm(x)
    if plot:
        plt.scatter(x[:, 0], x[:, 1], c=y)
        plt.show()

    xtrain, xtest, ytrain, ytest = train_test_split(x, y)
    classifier = NaiveBayes()
    classifier.fit(xtrain, ytrain)

    ypred = classifier.predict(xtest)
    perf = mcm(ytest, ypred)
    print('\nTest performance', perf)
    perf.print_confusion_matrix()

    ytrain_pred = classifier.predict(xtrain)
    comp = mcm(ytrain, ytrain_pred)
    print('\nTrain performance', comp)


# binary_classifier(plot=False)
naive_bayes(plot=False)
