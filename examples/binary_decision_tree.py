import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from endochrone.classification.binary_tree import BinaryDecisionTree

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def plot_blobs():
    x, y = make_blobs(n_samples=150, n_features=2, random_state=1234,
                      cluster_std=1.5, centers=3)

    bdt_test = BinaryDecisionTree(max_depth=3)
    bdt_test.fit(x, y)
    print(bdt_test)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()


plot_blobs()
