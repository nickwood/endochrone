# -*- coding: utf-8 -*-
import numpy as np

from endochrone import metrics

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_small_confusion_matrix():
    y_pred = np.transpose([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1]])
    y_test = np.transpose([[0, 1, 2, 2, 2, 2, 0, 2, 2, 0, 1, 1, 0, 1, 2, 0]])
    labels, comb, counts = metrics.confusion_matrix(y_test, y_pred)
    assert np.all(labels == np.array([0, 1, 2]))
    e_comb = np.array([[0, 0], [0, 2], [1, 0], [1, 1], [1, 2], [2, 1], [2, 2]])
    assert np.all(comb == e_comb)
    assert np.all(counts == np.array([4, 1, 1, 3, 2, 1, 4]))
    metrics.print_confusion_matrix(y_test, y_pred)


def test_large_confusion_matrix():
    y_pred = np.transpose([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7,
                            8, 9, 1, 2, 3, 4, 5, 5, 7, 8, 9, 1, 2, 3, 4, 5, 6,
                            7, 8, 9, 1, 2, 3, 5, 6, 7, 8, 9, 1, 2, 3, 1, 3]])
    y_test = np.transpose([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 2, 5, 6, 7,
                            8, 9, 1, 2, 5, 4, 5, 5, 7, 8, 9, 1, 2, 3, 4, 5, 6,
                            7, 8, 9, 1, 2, 3, 6, 6, 7, 8, 9, 1, 8, 3, 1, 3]])
    labels, comb, counts = metrics.confusion_matrix(y_test, y_pred)
    assert len(labels) == 10
    assert comb.shape == (14, 2)
    e_counts = np.array([1, 7, 5, 1, 6, 1, 1, 3, 5, 1, 4, 5, 5, 5])
    assert np.all(counts == e_counts)


def test_binary_metrics():
    ytest = np.transpose([[True, False, True, False, True, False, False, True,
                           False, False, False, True, False, False, True]])
    ypred = np.transpose([[False, True, False, False, True, False, True, True,
                           True, False, True, False, True, False, False]])
    assert metrics.true_positive(ytest, ypred) == pytest.approx(2/15)
    assert metrics.false_positive(ytest, ypred) == pytest.approx(5/15)
    assert metrics.true_negative(ytest, ypred) == pytest.approx(4/15)
    assert metrics.false_negative(ytest, ypred) == pytest.approx(4/15)
    assert metrics.precision(ytest, ypred) == pytest.approx(2/7)
    assert metrics.recall(ytest, ypred) == pytest.approx(2/6)
    assert metrics.f1_score(ytest, ypred) == pytest.approx(2/13)


def test_non_binary_metrics():
    # y_test = np.transpose([[0, 1, 2, 2, 2, 2, 0, 2, 2, 0, 1, 1, 0, 1, 2, 0]])
    # y_pred = np.transpose([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1]])
    # assert metrics.true_positive(y_test, y_pred) == pytest.approx(11/16)
    # TODO
    assert True


# test_small_confusion_matrix()
# test_large_confusion_matrix()
test_binary_metrics()
# test_non_binary_metrics()
