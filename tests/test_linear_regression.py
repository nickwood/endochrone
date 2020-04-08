# -*- coding: utf-8 -*-
import numpy as np
import random
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


def test_nd_nonzero_intercept(n_samples=100, dim=4):
    X_train = np.random.uniform(100, size=(n_samples, dim))
    coefs = np.random.uniform(1, 10, size=dim)
    intercept = random.uniform(-10, 10)
    Y_train_values = [sum(point*coefs) + intercept + random.uniform(-0.2, 0.2)
                      for point in X_train]
    Y_train = np.array(Y_train_values)[:, np.newaxis]

    model = LinearRegression(X_train, Y_train)
    model.fit()

    # test fit
    assert model.intercept_ == pytest.approx(intercept, abs=0.2)
    assert np.all(model.coef_ == pytest.approx(coefs, abs=0.2))

    # test predictions
    X_test = np.random.uniform(100, size=(50, dim))
    Y_test = np.array([sum(point*coefs) + intercept
                       for point in X_test])[:, np.newaxis]
    Y_pred = model.predict(X_test)
    assert np.all(Y_pred == pytest.approx(Y_test, abs=0.2))
    assert model.score(X_test, Y_test) > 0.999


# test_2d_zero_intercept()
# test_2d_nonzero_intercept()
# test_nd_nonzero_intercept()
test_nd_nonzero_intercept(n_samples=100000, dim=200)
