# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone.misc import lazy_test_runner as ltr
from endochrone import correlation

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


x = np.arange(0, 10, 0.1)
y_pos = np.arange(1, 21, 0.2)
y_neg = np.arange(21, 1, -0.2)
y_rand = np.random.randint(0, 10, 100)


def test_covariance():
    assert correlation.covariance([1, 2], [1, 2]) == pytest.approx(0.5)
    assert correlation.covariance(x, y_pos) == pytest.approx(1666.5)
    assert correlation.covariance(x, y_neg) == pytest.approx(-1666.5)


def test_pearson():
    assert correlation.pearson([1, 2], [1, 2]) == pytest.approx(1)
    assert correlation.pearson(x, y_pos) == pytest.approx(1)
    assert correlation.pearson(x, y_neg) == pytest.approx(-1)
    assert np.abs(correlation.pearson(x, y_rand)) < 0.2


ltr()
