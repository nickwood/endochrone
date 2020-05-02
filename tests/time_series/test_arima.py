# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone.utils.misc import lazy_test_runner as ltr
import endochrone.time_series.arima as arima

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


def test_ma1_model():
    x = np.array([9, 10, 11, 12, 11, 10, 9, 8])
    MA1 = arima.MaModel(order=1)

    residuals = MA1.residuals(x, [np.mean(x), 0.5])
    exp_res = [-1.0, 0.5, 0.75, 1.625, 0.1875, -0.09375, -0.953125, -1.5234375]
    assert np.all(residuals == pytest.approx(exp_res))

    assert MA1.fit(x)
    exp_thetas = [9.6237, 0.7531]
    assert np.all(MA1.thetas_ == pytest.approx(exp_thetas, abs=0.0001))

    with pytest.raises(ValueError):
        preds = MA1.predict([])

    preds = MA1.predict([10, 11, 9, 10, 12])
    exp = [9.90709153, 10.44676937, 8.534137988, 10.72764068, 10.5819138]
    assert np.all(preds == pytest.approx(exp, abs=0.001))


def test_ma2_model():
    x = np.array([9, 10, 11, 12, 11, 10, 9, 8])
    MA2 = arima.MaModel(order=2)

    residuals = MA2.residuals(x, [np.mean(x), 0.5, 0.5])
    exp_res = [-1.0, 0.5, 1.25, 1.125, -0.1875, -0.46875, -0.671875, -1.429688]
    assert np.all(residuals == pytest.approx(exp_res))

    assert MA2.fit(x)
    assert np.sum(MA2.residuals_**2) == pytest.approx(1.7175879816717088)
    exp_thetas = [8.86689311, 1.38181157, 1.98175309]
    assert np.all(MA2.thetas_ == pytest.approx(exp_thetas))

    with pytest.raises(ValueError):
        preds = MA2.predict([10])

    preds = MA2.predict([10, 11, 9, 10, 12])
    exp = [11.89642503, 5.988960158, 8.669395113, 21.41805208, 15.46732964]
    assert np.all(preds == pytest.approx(exp, abs=0.001))


ltr()
