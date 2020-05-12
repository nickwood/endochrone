# -*- coding: utf-8 -*-
from functools import partial
import numpy as np

from endochrone.optimisation import BatchGradientDescent

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class LinearRegression:
    def __init__(self, *, method='inv_covariance', calculate_residuals=False,
                 predict_vectors=False, params={}):
        '''
        supported methods:
        * 'inv_covariance' solves by finding the inverse of the covariance
        matrix.
        * 'gradient_descent' is more robust, but may take longer and be less
        precise.
        calculate_residuals specifies if you wish residuals to be calculated
        and stored as self.residuals_
        predict_vectors lets you specify whether the predict method
        returns a 1-d or 2-d column vector (True -> 2d vector). Default
        behaviour retains the dimensionality of the Y_train set
        '''
        self.calculate_residuals = calculate_residuals
        self.predict_vectors = predict_vectors
        if method not in ['inv_covariance', 'gradient_descent']:
            raise ValueError('Unknown method: %s' % method)
        self.method = method
        self.params = params

    def fit(self, X_train, Y_train):
        if self.predict_vectors is False:
            if Y_train.ndim == 1:
                self.predict_vectors = False
            else:
                self.predict_vectors = True

        if self.method == 'inv_covariance':
            self.fit_inv_covariance(X_train, Y_train)
        else:
            self.fit_gradient_desc(X_train, Y_train)

        if self.calculate_residuals:
            preds = self.predict(X_train)
            if Y_train.ndim < preds.ndim:
                self.residuals_ = Y_train[:, np.newaxis] - preds
            else:
                self.residuals_ = Y_train - preds
        else:
            self.residuals_ = None

    def fit_inv_covariance(self, X_train, Y_train):
        n_samples = X_train.shape[0]
        X = np.c_[np.ones(n_samples), X_train]
        X_T = X.transpose()

        beta_vector = np.linalg.inv(X_T@X)@X_T@Y_train
        if beta_vector.ndim == 1:
            beta_vector = beta_vector[:, np.newaxis]
        self.intercept_ = beta_vector[0, 0]
        self.coef_ = beta_vector[1:, 0]

    def fit_gradient_desc(self, X_train, Y_train):
        coefs = create_coef_dict(size=X_train.shape[1] + 1)
        squares = partial(sq_errors, X=X_train, Y=Y_train)
        gd = BatchGradientDescent(**self.params)
        if gd.fit(x0=coefs, func=squares):
            self.intercept_ = gd.min_args['x0']
            self.coef_ = list(gd.min_args.values())[1:]
        else:
            raise AttributeError("GD algorithm did not terminate")

    def predict(self, X_test):
        if X_test.ndim == 1:
            y = X_test * self.coef_ + self.intercept_
        else:
            y = np.sum(X_test * self.coef_, axis=1) + self.intercept_

        if self.predict_vectors:
            return y[:, np.newaxis]
        else:
            return y

    def score(self, X_true, Y_true):
        """Return the R^2 value for the prediction
        R^2 = 1- u/v with:
        u = ((Y_true - Y_pred) ** 2).sum()
        v = ((Y_true - Y_true.mean()) ** 2).sum()
        """
        Y_pred = self.predict(X_true)
        u = np.sum((Y_true - Y_pred)**2)
        v = np.sum((Y_true - np.mean(Y_true))**2)
        return 1 - u/v


def sq_errors(*, X, Y, **coefs):
    return 1/2 * np.mean((evaluate(X=X, **coefs) - Y)**2)


def evaluate(*, X, **coefs):
    '''return the regressed values of X given specified coeffiecients.
    NB coefs[0] is the intercept term'''
    multiples = np.fromiter(coefs.values(), dtype=float)[1:]
    return np.sum(X * multiples, axis=1) + coefs['x0']


def create_coef_dict(*, size, default=1):
    '''returns a dict of the form {'x0': 1, 'x1': 1, 'x2': 1}'''
    return dict(zip(['x%s' % i for i in range(size)], [default] * size))
