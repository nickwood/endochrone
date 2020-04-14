# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone import binary_decision_tree as bdt
from endochrone.misc import lazy_test_runner as ltr

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_gini_score():
    assert bdt.gini_score([0, 0, 0, 1], [1, 1, 1]) == pytest.approx(17/112)
    p1 = np.array([0, 0, 0, 1])
    p2 = np.array([1, 1, 1])
    assert bdt.gini_score(p1, p2) == pytest.approx(17/112)
    assert bdt.gini_score([0, 0, 0, 0], [1, 1, 1]) == pytest.approx(0)
    assert bdt.gini_score([0, 0, 0, 1], [1, 0, 1]) == pytest.approx(401/1008)

    p3 = ['cat', 'dog', 'dog', 'cow']
    p4 = ['cat', 'cat', 'cow']
    exp = pytest.approx(667/1008)
    assert bdt.gini_score(p3, p4) == exp
    assert bdt.gini_score(np.array(p3), np.array(p4)) == exp


def test_partition():
    x_1 = np.transpose([[1, 2, 3, 7, 8, 9, 10],
                        [3, 4, 5, 1, 2, 3, 4],
                        [3, 1, 5, 1, 2, 3, 4]])
    y_1 = np.array([0, 0, 0, 1, 1, 1, 1])

    p_1, p_2 = bdt.partition(x_1, y_1, 0, 2.5)
    assert np.all(p_1 == [0, 0])
    assert np.all(p_2 == [0, 1, 1, 1, 1])
    assert isinstance(p_1, np.ndarray)
    assert isinstance(p_2, np.ndarray)

    p_1, p_2 = bdt.partition(x_1, y_1, 1, 1.5)
    assert np.all(p_1 == [1])
    assert np.all(p_2 == [0, 0, 0, 1, 1, 1])

    p_1, p_2 = bdt.partition(x_1, y_1, 2, 1.5)
    assert np.all(p_1 == [0, 1])
    assert np.all(p_2 == [0, 0, 1, 1, 1])

    y_2 = np.array([0, 2, 2, 0, 1, 1, 2])

    p_1, p_2 = bdt.partition(x_1, y_2, 0, 5.5)
    assert np.all(p_1 == [0, 2, 2])
    assert np.all(p_2 == [0, 1, 1, 2])
    assert isinstance(p_1, np.ndarray)
    assert isinstance(p_2, np.ndarray)

    p_1, p_2 = bdt.partition(x_1, y_2, 1, 1.5)
    assert np.all(p_1 == [0])
    assert np.all(p_2 == [0, 2, 2, 1, 1, 2])

    p_1, p_2 = bdt.partition(x_1, y_2, 2, 2.5)
    assert np.all(p_1 == [2, 0, 1])
    assert np.all(p_2 == [0, 2, 1, 2])

    y_3 = np.array([['cat']*3 + ['dog']*2 + ['cow']*2])

    p_1, p_2 = bdt.partition(x_1, y_3, 0, 5.5)
    assert np.all(p_1 == ['cat']*3)
    assert np.all(p_2 == ['dog']*2 + ['cow']*2)
    assert isinstance(p_1, np.ndarray)
    assert isinstance(p_2, np.ndarray)

    p_1, p_2 = bdt.partition(x_1, y_3, 1, 3.5)
    assert np.all(p_1 == ['cat', 'dog', 'dog', 'cow'])
    assert np.all(p_2 == ['cat', 'cat', 'cow'])

    p_1, p_2 = bdt.partition(x_1, y_3, 2, 2.5)
    assert np.all(p_1 == ['cat', 'dog', 'dog'])
    assert np.all(p_2 == ['cat', 'cat', 'cow', 'cow'])


ltr()
