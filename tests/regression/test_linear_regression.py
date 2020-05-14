# -*- coding: utf-8 -*-
import numpy as np
import pytest
import random

from endochrone.regression import LinearRegression
from endochrone.utils import lazy_test_runner as ltr

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_invalid_method():
    with pytest.raises(ValueError):
        _ = LinearRegression(method='bob')


def test_2d_zero_intercept():
    X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])[:, np.newaxis]
    Y_train = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18])

    model = LinearRegression(calculate_residuals=True)
    model.fit(X_train, Y_train)

    # test fit
    assert model.coef_[0] == pytest.approx(2)
    assert model.intercept_ == pytest.approx(0)

    assert model.residuals_.shape == (9,)
    assert np.all(model.residuals_ == pytest.approx(0))

    # test predictions
    X_test = np.array([1.5, 2.5, 3.5, 7.5])[:, np.newaxis]
    Y_test = np.array([3, 5, 7, 15])
    Y_pred = model.predict(X_test)
    assert np.all(Y_pred == pytest.approx(Y_test))


def test_2d_nonzero_intercept():
    X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])[:, np.newaxis]
    Y_train = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19])

    model = LinearRegression()
    model.fit(X_train, Y_train)

    # test fit
    assert model.coef_[0] == pytest.approx(2)
    assert model.intercept_ == pytest.approx(1)

    # test predictions
    X_test = np.array([1.5, 2.5, 3.5, 7.5])[:, np.newaxis]
    Y_test = np.array([4, 6, 8, 16])
    Y_pred = model.predict(X_test)
    assert np.all(Y_pred == pytest.approx(Y_test))
    assert model.residuals_ is None


def test_invalid_dimensions():
    X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])[:, np.newaxis]
    Y_train = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19])

    model = LinearRegression()
    with pytest.raises(ValueError):
        model.fit(X_train, Y_train[:, np.newaxis])
    with pytest.raises(ValueError):
        model.fit(X_train.ravel(), Y_train)


def test_nd_nonzero_intercept(n_samples=1000, dim=20):
    X_train = np.random.uniform(100, size=(n_samples, dim))
    coefs = np.random.uniform(1, 10, size=dim)
    intercept = random.uniform(-10, 10)
    Y_train_values = [sum(point*coefs) + intercept + random.uniform(-0.2, 0.2)
                      for point in X_train]
    Y_train = np.array(Y_train_values)

    model = LinearRegression()
    model.fit(X_train, Y_train)

    # test fit
    assert model.intercept_ == pytest.approx(intercept, abs=0.2)
    assert np.all(model.coef_ == pytest.approx(coefs, abs=0.2))

    # test predictions
    X_test = np.random.uniform(100, size=(50, dim))
    Y_test = np.array([sum(point*coefs) + intercept
                       for point in X_test])
    Y_pred = model.predict(X_test)
    assert np.all(Y_pred == pytest.approx(Y_test, abs=0.2))
    assert model.score(X_test, Y_test) > 0.999
    assert model.residuals_ is None


def test_2dimx_1dimy():
    X_train = np.transpose([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [1, 2, 3, 4, 6, 5, 7, 8, 9]])
    Y_train = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19])

    model = LinearRegression()
    model.fit(X_train, Y_train)

    assert model.coef_[0] == pytest.approx(2)
    assert model.intercept_ == pytest.approx(1)

    # test predictions
    X_test = np.transpose([[1.5, 2.5, 3.5, 7.5], [1.5, 2.5, 3.5, 7.5]])
    Y_test = np.array([4, 6, 8, 16])
    Y_pred = model.predict(X_test)
    assert np.all(Y_pred == pytest.approx(Y_test))


def test_gradient_desc():
    X_train = np.transpose([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [1, 2, 3, 4, 5, 6, 7, 8, 9]])
    Y_train = np.array([6, 10, 14, 18, 22, 26, 30, 34, 38])

    model_la = LinearRegression()
    with pytest.raises(np.linalg.LinAlgError):
        model_la.fit(X_train, Y_train)

    model_gd_no_params = LinearRegression(method='gradient_descent')
    with pytest.warns(RuntimeWarning):
        model_gd_no_params.fit(X_train, Y_train)

    gd_params = {'tol': 0.001, 'max_iter': 2000, 'learning_rate': 2}
    model_gd = LinearRegression(method='gradient_descent', params=gd_params)
    model_gd.fit(X_train, Y_train)

    assert np.sum(model_gd.coef_) == pytest.approx(4, abs=0.1)
    assert model_gd.intercept_ == pytest.approx(2, abs=0.2)

    # test predictions
    X_test = np.transpose([[1.5, 2.5, 3.5, 7.5], [1.5, 2.5, 3.5, 7.5]])
    Y_test = np.array([8, 12, 16, 32])
    Y_pred = model_gd.predict(X_test)
    assert np.all(Y_pred == pytest.approx(Y_test, abs=0.2))


ltr()
