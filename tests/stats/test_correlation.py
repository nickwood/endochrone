# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone.utils import lazy_test_runner as ltr
from endochrone.stats import correlation

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


def test_auto_correlation():
    acf1 = correlation.acf(y_pos)
    assert acf1.auto_correlation(2) == pytest.approx(0.95919592)
    assert acf1.auto_correlation(7) == pytest.approx(0.85018502)

    acf2 = correlation.acf(y_neg, lags=10)
    assert acf2.auto_correlation(5) == pytest.approx(0.8949895)
    assert acf2.auto_correlation(13) == pytest.approx(0.70617062)

    acf3 = correlation.acf(y_per, lags=20)
    assert acf3.auto_correlation(0) == pytest.approx(1)
    assert acf3.auto_correlation(6) == pytest.approx(-0.27787815)
    assert acf3.auto_correlation(16) == pytest.approx(0.27417115)
    assert acf3.auto_correlation(18) == pytest.approx(0.78695581)


acf_exp = np.array([1., 0.97979798, 0.95919592, 0.93819382, 0.91679168,
                    0.8949895, 0.87278728, 0.85018502, 0.82718272, 0.80378038,
                    0.779978])


def test_acf():
    acf_act = correlation.acf(x, lags=10).values
    assert np.all(acf_act == pytest.approx(acf_exp))


def test_pacf():
    pacf_cls = correlation.pacf(x, lags=10)
    pacf_exp = np.array([1., 0.97979798, -0.0202061, -0.02042467, -0.0206562,
                         -0.02090123, -0.02116036, -0.02143422, -0.02172351,
                         -0.02202899, -0.02235147])

    assert np.all(pacf_cls.values == pytest.approx(pacf_exp))
    assert np.all(pacf_cls.acf_values == pytest.approx(acf_exp))


ltr()
