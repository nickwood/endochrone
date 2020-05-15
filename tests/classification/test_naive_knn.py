# -*- coding: utf-8 -*-
import numpy as np

from endochrone.classification import naive_knn as knn
from endochrone.classification import KNearest
from endochrone.utils import lazy_test_runner as ltr

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_fit_and_predict():
    X_train = np.transpose([[1, 2, 3, 7, 8, 9, 10],
                            [3, 4, 5, 1, 2, 3, 4],
                            [3, 4, 5, 1, 2, 3, 4]])
    Y_train = np.transpose([0, 0, 0, 1, 1, 1, 1])

    X_test = np.transpose([[1, 2.5, 3.5, 8, 8.9, 9.9, 10.2],
                           [2, 3.1, 4.2, 0.5, 2.1, 2.6, 4],
                           [2.9, 3.8, 5.2, 0.6, 1.6, 3, 4]])
    Y_test = np.transpose([0, 0, 0, 1, 1, 1, 1])

    knn_model = KNearest()
    knn_model.fit(features=X_train, targets=Y_train)
    act = knn_model.predict(features=X_test)
    assert np.all(act == Y_test)


def test_classify_point():
    X_train = np.transpose([[1, 2, 3, 7, 8, 9, 10],
                            [3, 4, 5, 1, 2, 3, 4],
                            [3, 4, 5, 1, 2, 3, 4]])
    Y_train = np.transpose([[0, 0, 0, 1, 1, 1, 1]])

    knn_model = KNearest()
    knn_model.fit(features=X_train, targets=Y_train)

    test_1 = np.array([9, 3, 4])
    test_2 = np.array([3, 5, 5])
    test_3 = np.array([0, 1, 0])
    test_4 = np.array([6, 6, 6])
    test_5 = np.array([8, 1.5, 1.5])

    assert knn_model.classify_point(point=test_1) == 1
    assert knn_model.classify_point(point=test_2) == 0
    assert knn_model.classify_point(point=test_3) == 0
    assert knn_model.classify_point(point=test_4) == 0
    assert knn_model.classify_point(point=test_5) == 1


def test_majority():
    test_1 = np.transpose([[0, 0, 0]])
    test_2 = np.transpose([[1, 1, 1]])
    test_3 = np.transpose([[0, 1, 0]])
    test_4 = np.transpose([[0, 1, 1, 2]])
    test_5 = np.transpose([[1, 0, 2]])

    assert knn.majority(test_1) == 0
    assert knn.majority(test_2) == 1
    assert knn.majority(test_3) == 0
    assert knn.majority(test_4) == 1
    assert knn.majority(test_5) == 0


ltr()
