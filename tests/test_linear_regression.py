# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone.linear_regression import LinearRegression

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_2d_zero_intercept():
    X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])[:, np.newaxis]
    Y_train = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18])[:, np.newaxis]

    model = LinearRegression(X_train, Y_train)
    model.fit()

    # test fit
    assert model.coef_[0] == pytest.approx(2)
    assert model.intercept_ == pytest.approx(0)

    # test predictions
    X_test = np.array([1.5, 2.5, 3.5, 7.5])[:, np.newaxis]
    Y_test = np.array([3, 5, 7, 15])[:, np.newaxis]
    Y_pred = model.predict(X_test)
    assert np.all(Y_pred == pytest.approx(Y_test))


def test_2d_nonzero_intercept():
    X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])[:, np.newaxis]
    Y_train = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19])[:, np.newaxis]

    model = LinearRegression(X_train, Y_train)
    model.fit()

    # test fit
    assert model.coef_[0] == pytest.approx(2)
    assert model.intercept_ == pytest.approx(1)

    # test predictions
    X_test = np.array([1.5, 2.5, 3.5, 7.5])[:, np.newaxis]
    Y_test = np.array([4, 6, 8, 16])[:, np.newaxis]
    Y_pred = model.predict(X_test)
    assert np.all(Y_pred == pytest.approx(Y_test))


test_2d_zero_intercept()
test_2d_nonzero_intercept()
