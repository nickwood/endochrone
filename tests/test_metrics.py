# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone.metrics import BinaryMetrics
from endochrone.metrics import MulticlassMetrics
from endochrone.misc import lazy_test_runner as ltr

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_small_confusion_matrix(capsys):
    ytest = np.transpose([[0, 1, 2, 2, 2, 2, 0, 2, 2, 0, 1, 1, 0, 1, 2, 0]])
    ypred = np.transpose([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1]])
    metrics = BinaryMetrics(ytest, ypred)
    comb, counts, labels = metrics.confusion_matrix
    assert np.all(labels == np.array([0, 1, 2]))
    e_comb = np.array([[0, 0], [0, 2], [1, 0], [1, 1], [1, 2], [2, 1], [2, 2]])
    e_comb = np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
    assert np.all(comb == e_comb)
    assert np.all(counts == np.array([4, 1, 3, 1, 1, 2, 4]))
    metrics.print_confusion_matrix()
    captured = capsys.readouterr()
    printed = captured.out
    assert 'pred\\act\t0\t1\t2\n0' in printed
    assert '\t4\t0\t1\n1' in printed
    assert '\t1\t3\t2\n2' in printed
    assert '\t0\t1\t4\n' in printed


def test_large_confusion_matrix():
    ytest = np.transpose([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 2, 5, 6, 7,
                           8, 9, 1, 2, 5, 4, 5, 5, 7, 8, 9, 1, 2, 3, 4, 5, 6,
                           7, 8, 9, 1, 2, 3, 6, 6, 7, 8, 9, 1, 8, 3, 1, 3]])
    ypred = np.transpose([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7,
                           8, 9, 1, 2, 3, 4, 5, 5, 7, 8, 9, 1, 2, 3, 4, 5, 6,
                           7, 8, 9, 1, 2, 3, 5, 6, 7, 8, 9, 1, 2, 3, 1, 3]])
    metrics = BinaryMetrics(ytest, ypred)
    comb, counts, labels = metrics.confusion_matrix
    assert len(labels) == 10
    assert comb.shape == (14, 2)
    e_counts = np.array([1, 7, 5, 1, 6, 3, 1, 5, 1, 4, 5, 1, 5, 5])
    assert np.all(counts == e_counts)


def test_binary_metrics():
    ytest = np.transpose([[True, False, True, False, True, False, False, True,
                           False, False, False, True, False, False, True]])
    ypred = np.transpose([[False, True, False, False, True, False, True, True,
                           True, False, True, False, True, False, False]])
    metrics = BinaryMetrics(ytest, ypred)
    assert metrics.true_positive == pytest.approx(2/15)
    assert metrics.false_positive == pytest.approx(5/15)
    assert metrics.true_negative == pytest.approx(4/15)
    assert metrics.false_negative == pytest.approx(4/15)
    assert metrics.precision == pytest.approx(2/7)
    assert metrics.recall == pytest.approx(2/6)
    assert metrics.f1_score == pytest.approx(4/13)


def test_multiclass_metrics():
    ytest = np.transpose([[0, 1, 2, 2, 2, 2, 0, 2, 2, 0, 1, 1, 0, 1, 2, 0]])
    ypred = np.transpose([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1]])
    metrics = MulticlassMetrics(ytest, ypred)
    assert np.all(metrics.n_predicted == np.array([5, 6, 5]))
    assert isinstance(metrics.n_predicted, np.ndarray)
    assert np.all(metrics.n_true == np.array([5, 4, 7]))
    assert isinstance(metrics.n_true, np.ndarray)
    assert np.all(metrics.n_true_positive == np.array([4, 3, 4]))
    assert isinstance(metrics.n_true_positive, np.ndarray)
    assert metrics.macro_precision == pytest.approx(7/10)
    assert metrics.macro_recall == pytest.approx(99/140)
    assert metrics.macro_f1_score == pytest.approx(693/985)


def test_char_labels():
    ytest = np.transpose([['a']*5 + ['b']*4 + ['c']*2 + ['a'] + ['b']*6])
    ypred = np.transpose([['a']*4 + ['b']*4 + ['c']*2 + ['a'] + ['b']*7])
    metrics = MulticlassMetrics(ytest, ypred)
    assert np.all(metrics.n_predicted == np.array([5, 11, 2]))
    assert isinstance(metrics.n_predicted, np.ndarray)
    assert np.all(metrics.n_true == np.array([6, 10, 2]))
    assert isinstance(metrics.n_true, np.ndarray)
    assert np.all(metrics.n_true_positive == np.array([4, 9, 1]))
    assert isinstance(metrics.n_true_positive, np.ndarray)
    assert metrics.macro_precision == pytest.approx(233/330)
    assert metrics.macro_recall == pytest.approx(31/45)
    assert metrics.macro_f1_score == pytest.approx(14446/20715)


def test_word_labels():
    ytest = np.transpose([['cat']*5 + ['dog']*4 + ['cow']*6])
    ypred = np.transpose([['cat']*4 + ['dog']*4 + ['cat']*2 + ['cow']*5])
    metrics = MulticlassMetrics(ytest, ypred)
    assert np.all(metrics.n_predicted == np.array([6, 5, 4]))
    assert isinstance(metrics.n_predicted, np.ndarray)
    assert np.all(metrics.n_true == np.array([5, 6, 4]))
    assert isinstance(metrics.n_true, np.ndarray)
    assert np.all(metrics.n_true_positive == np.array([4, 5, 3]))
    assert isinstance(metrics.n_true_positive, np.ndarray)
    assert metrics.macro_precision == pytest.approx(29/36)
    assert metrics.macro_recall == pytest.approx(143/180)
    assert metrics.macro_f1_score == pytest.approx(4147/5184)


ltr('test_metrics.py')
