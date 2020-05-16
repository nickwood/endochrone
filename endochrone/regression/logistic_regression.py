# -*- coding: utf-8 -*-
from functools import partial
import numpy as np
import warnings

from endochrone import Base
from endochrone.optimisation import BatchGradientDescent

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class LogisticRegression(Base):
    def __init__(self, *, gd_params={}, suppress_warnings=False):
        '''
        gd_params is a dict specifying parameters to be passewd to the gradient
        descent algorithm
        '''
        self.gd_params = gd_params
        self.suppress_warnings = suppress_warnings
        self.coef_dict_ = None
        super().__init__(properties={'binary_targets': True})

    def fit(self, X_train, Y_train):
        self.validate_fit(features=X_train, targets=Y_train)
        self.fit_gradient_desc(X_train, Y_train)

    def fit_gradient_desc(self, X_train, Y_train):
        N = X_train.shape[1] + 1
        coefs = dict(zip(['θ%s' % i for i in range(N)], [1.] * N))
        squares = partial(neg_log_likelihood, X=X_train, Y=Y_train)
        gd = BatchGradientDescent(**self.gd_params)
        if not gd.fit(x0=coefs, func=squares) and not self.suppress_warnings:
            warnings.warn("GD algorithm did not terminate", RuntimeWarning)
        self.coef_dict_ = gd.min_args

    def predict(self, X_test):
        self.validate_predict(features=X_test)
        h_x = evaluate(X=X_test, **self.coef_dict_)
        return np.round(h_x).astype(int)


def neg_log_likelihood(*, X, Y, **coefs):
    '''return the negative log likelihood of the logistic function defined by
    **coefs, given X and Y
    Note we return the negative log, and minimise using gradient descent'''
    # we add epsilon to our log function - to prevent infinities from log(0)
    eps = 1.0 * 10**-20
    hypothesis = evaluate(X=X, **coefs)
    y_0_contrib = -np.sum(np.log(1 - hypothesis[np.argwhere(Y == 0)] + eps))
    y_1_contrib = -np.sum(np.log(hypothesis[np.argwhere(Y == 1)] + eps))
    return y_0_contrib + y_1_contrib


def evaluate(*, X, **coefs):
    '''return the values h(x) = g(θ.x) where g is the sigmoid function.
    by convention x0 = 1'''
    multiples = np.fromiter(coefs.values(), dtype=float)[1:]
    theta_x = np.sum(X * multiples, axis=1) + coefs['θ0']
    np.seterr(over='ignore')
    hypothesis = 1/(1+np.exp(-theta_x))
    np.seterr(over='warn')
    return hypothesis
