# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone.utils import lazy_test_runner as ltr
from endochrone.utils.helpers import softmax

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_softmax():
    X = np.array([[1, 0.5, 0.2, 3],
                  [1,  -1,   7, 3],
                  [2,  12,  13, 3]])
    act1 = softmax(X)
    exp1 = np.array([[4.48309e-06, 2.71913e-06, 2.01438e-06, 3.31258e-05],
                     [4.48309e-06, 6.06720e-07, 1.80861e-03, 3.31258e-05],
                     [1.21863e-05, 2.68421e-01, 7.29644e-01, 3.31258e-05]])
    assert np.all(act1 == pytest.approx(exp1, rel=1e-5))

    act2 = softmax(X, axis=0)
    exp2 = np.array([[2.11942e-01, 1.01300e-05, 2.75394e-06, 3.33333e-01],
                     [2.11942e-01, 2.26030e-06, 2.47262e-03, 3.33333e-01],
                     [5.76117e-01, 9.99988e-01, 9.97525e-01, 3.33333e-01]])
    assert np.all(act2 == pytest.approx(exp2, rel=1e-5))
    assert np.all(exp2.sum(axis=0) == pytest.approx(1, rel=1e-5))

    act3 = softmax(X, axis=1)
    exp3 = np.array([[1.05877e-01, 6.42177e-02, 4.75736e-02, 7.82332e-01],
                     [2.42746e-03, 3.28521e-04, 9.79307e-01, 1.79366e-02],
                     [1.22094e-05, 2.68929e-01, 7.31025e-01, 3.31885e-05]])
    assert np.all(act3 == pytest.approx(exp3, rel=1e-5))
    assert np.all(exp3.sum(axis=1) == pytest.approx(1, rel=1e-5))


ltr()
