# -*- coding: utf-8 -*-
from functools import partial
import numpy as np
import warnings

from endochrone import Base
from endochrone.optimisation import BatchGradientDescent

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class LinearRegression(Base):
    def __init__(self, *, method='inv_covariance', calculate_residuals=False,
                 params={}):
        '''
        supported methods:
        * 'inv_covariance' solves by finding the inverse of the covariance
        matrix.
        * 'gradient_descent' is more robust, but may take longer and be less
        precise.
        calculate_residuals specifies if you wish residuals to be calculated
        and stored as self.residuals_
        '''
        self.calculate_residuals = calculate_residuals
        if method not in ['inv_covariance', 'gradient_descent']:
            raise ValueError('Unknown method: %s' % method)
        self.method = method
        self.params = params

    def fit(self, X_train, Y_train):
        self.validate_fit(features=X_train, targets=Y_train)

        if self.method == 'inv_covariance':
            self.fit_inv_covariance(X_train, Y_train)
        else:
            self.fit_gradient_desc(X_train, Y_train)

        self.residuals_ = None
        if self.calculate_residuals:
            self.residuals_ = Y_train - evaluate(X=X_train, **self.coef_dict_)

    def fit_inv_covariance(self, X_train, Y_train):
        n_samples = X_train.shape[0]
        X = np.c_[np.ones(n_samples), X_train]
        X_T = X.transpose()

        beta_vector = np.linalg.inv(X_T@X)@X_T@Y_train.ravel()
        N = beta_vector.shape[0]
        coef_dict = dict(zip(['x%s' % i for i in range(N)], beta_vector))

        self.coef_dict_ = coef_dict
        self.intercept_ = beta_vector[0]
        self.coef_ = beta_vector[1:]

    def fit_gradient_desc(self, X_train, Y_train):
        N = X_train.shape[1] + 1
        coefs = dict(zip(['x%s' % i for i in range(N)], [1.] * N))
        squares = partial(mean_sq_errors, X=X_train, Y=Y_train)
        gd = BatchGradientDescent(**self.params)
        if gd.fit(x0=coefs, func=squares):
            self.coef_dict_ = gd.min_args
            self.intercept_ = gd.min_args['x0']
            self.coef_ = list(gd.min_args.values())[1:]
            return True
        else:
            warnings.warn("GD algorithm did not terminate", RuntimeWarning)
            return False

    def predict(self, X_test):
        self.validate_predict(features=X_test)
        return evaluate(X=X_test, **self.coef_dict_)

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


def mean_sq_errors(*, X, Y, **coefs):
    return 1/2 * np.mean((evaluate(X=X, **coefs) - Y)**2)


def evaluate(*, X, **coefs):
    '''return the regressed values of X given specified coeffiecients.
    NB coefs[0] is the intercept term'''
    multiples = np.fromiter(coefs.values(), dtype=float)[1:]
    return np.sum(X * multiples, axis=1) + coefs['x0']
