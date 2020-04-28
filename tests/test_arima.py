# -*- coding: utf-8 -*-
from functools import lru_cache
import numpy as np
import pytest

from endochrone.misc import lazy_test_runner as ltr
import endochrone.arima as arima

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_ar1_model():
    x = np.arange(0, 10, 0.2)

    AR1 = arima.ArModel(order=1, calculate_residuals=True)
    assert np.all(AR1.generate_lags(x) == x[:-1, np.newaxis])
    assert np.all(AR1.generate_lags(x, include_last=True) == x[:, np.newaxis])
    assert np.all(AR1.generate_targets(x) == x[1:])

    AR1.fit(x)
    assert AR1.coef_[0] == pytest.approx(1)
    assert AR1.intercept_ == pytest.approx(0.2)
    assert np.all(AR1.residuals_ == pytest.approx(0))

    x_to_pred = np.array([4, 7.4, 9])
    y_exp = np.array([4.2, 7.6, 9.2])
    assert np.all(AR1.predict(x_to_pred) == pytest.approx(y_exp))


def test_ar2_model():
    '''t_2 = t_1 + t_0 i.e. fibonacci'''
    x = np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
                  987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368])

    AR2 = arima.ArModel(order=2, calculate_residuals=True)
    lags = AR2.generate_lags(np.array(x))
    assert lags.shape == (len(x)-2, 2)
    assert np.all(lags[:, 0] == x[1:-1])
    assert np.all(lags[:, 1] == x[:-2])

    lags_wl = AR2.generate_lags(x, include_last=True)
    assert lags_wl.shape == (len(x)-1, 2)
    assert np.all(lags_wl[:, 0] == x[1:])
    assert np.all(lags_wl[:, 1] == x[:-1])
    assert np.all(AR2.generate_targets(x) == x[2:])

    AR2.fit(x)
    assert np.all(AR2.coef_ == pytest.approx(1))
    assert AR2.intercept_ == pytest.approx(0, abs=0.005)
    assert np.all(AR2.residuals_ == pytest.approx(0, abs=0.005))

    x_to_pred = np.array([4, 8, 12, 24])
    y_exp = np.array([12, 20, 36])
    assert np.all(AR2.predict(x_to_pred) == pytest.approx(y_exp, abs=0.005))


ltr()
