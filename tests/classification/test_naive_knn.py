# -*- coding: utf-8 -*-
from functools import partial
import numpy as np

from endochrone.classification import naive_knn as knn
from endochrone.utils.misc import lazy_test_runner as ltr

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_classify():
    X_train = np.transpose([[1, 2, 3, 7, 8, 9, 10],
                            [3, 4, 5, 1, 2, 3, 4],
                            [3, 4, 5, 1, 2, 3, 4]])
    X_test = np.transpose([[0, 0, 0, 1, 1, 1, 1]])

    Y_train = np.transpose([[1, 2.5, 3.5, 8, 8.9, 9.9, 10.2],
                            [2, 3.1, 4.2, 0.5, 2.1, 2.6, 4],
                            [2.9, 3.8, 5.2, 0.6, 1.6, 3, 4]])
    exp = np.transpose([[0, 0, 0, 1, 1, 1, 1]])

    act = knn.classify(X_train, X_test, Y_train, k=3)
    assert np.all(act == exp)


def test_classify_point():
    X_train = np.transpose([[1, 2, 3, 7, 8, 9, 10],
                            [3, 4, 5, 1, 2, 3, 4],
                            [3, 4, 5, 1, 2, 3, 4]])
    X_test = np.transpose([[0, 0, 0, 1, 1, 1, 1]])

    test_1 = np.array([9, 3, 4])
    test_2 = np.array([3, 5, 5])
    test_3 = np.array([0, 1, 0])
    test_4 = np.array([6, 6, 6])
    test_5 = np.array([8, 1.5, 1.5])

    classifier = partial(knn.classify_point, X_train, X_test, k=3)
    assert classifier(test_1) == 1
    assert classifier(test_2) == 0
    assert classifier(test_3) == 0
    assert classifier(test_4) == 0
    assert classifier(test_5) == 1


def test_majority_concensus():
    test_1 = np.transpose([[0, 0, 0]])
    test_2 = np.transpose([[1, 1, 1]])
    test_3 = np.transpose([[0, 1, 0]])
    test_4 = np.transpose([[0, 1, 1, 2]])
    test_5 = np.transpose([[1, 0, 2]])

    assert knn.majority_concensus(test_1) == 0
    assert knn.majority_concensus(test_2) == 1
    assert knn.majority_concensus(test_3) == 0
    assert knn.majority_concensus(test_4) == 1
    assert knn.majority_concensus(test_5) == 1


ltr()
