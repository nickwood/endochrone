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
y_per = np.sin(x * np.pi)


def test_covariance():
    assert correlation.covariance([1, 2], [1, 2]) == pytest.approx(0.5)
    assert correlation.covariance(x, y_pos) == pytest.approx(1666.5)
    assert correlation.covariance(x, y_neg) == pytest.approx(-1666.5)


def test_pearson():
    assert correlation.pearson([1, 2], [1, 2]) == pytest.approx(1)
    assert correlation.pearson(x, y_pos) == pytest.approx(1)
    assert correlation.pearson(x, y_neg) == pytest.approx(-1)
    assert np.abs(correlation.pearson(x, y_rand)) < 0.26


def test_partial_correlation():
    x = np.array([2, 4, 15, 20])
    y = np.array([1, 2, 3, 4])
    z = np.array([0, 0, 1, 1])
    assert correlation.pearson(x, y) == pytest.approx(0.9695015519208121)

    assert correlation.partial_correlation(x, y, z) == pytest.approx(0.9191450)
    assert correlation.partial_correlation(x, x, z) == pytest.approx(1)
    assert correlation.partial_correlation(x, z, y) == pytest.approx(0.9128709)
    assert correlation.partial_correlation(y, z, x) == pytest.approx(-0.695222)


def test_singular_auto_correlation():
    assert correlation.auto_correlation(y_pos, 2) == pytest.approx(0.95919592)
    assert correlation.auto_correlation(y_pos, 7) == pytest.approx(0.85018502)
    assert correlation.auto_correlation(y_neg, 5) == pytest.approx(0.8949895)
    assert correlation.auto_correlation(y_neg, 13) == pytest.approx(0.70617062)
    assert correlation.auto_correlation(y_per, 0) == pytest.approx(1)
    assert correlation.auto_correlation(y_per, 6) == pytest.approx(-0.27787815)
    assert correlation.auto_correlation(y_per, 16) == pytest.approx(0.27417115)
    assert correlation.auto_correlation(y_per, 18) == pytest.approx(0.78695581)


def test_acf():
    acf_act = correlation.acf(x, lags=10)
    acf_exp = np.array([1., 0.97979798, 0.95919592, 0.93819382, 0.91679168,
                        0.8949895, 0.87278728, 0.85018502, 0.82718272,
                        0.80378038, 0.779978])
    assert np.all(acf_act == pytest.approx(acf_exp))


ltr()
