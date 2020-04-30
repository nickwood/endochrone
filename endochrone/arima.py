# -*- coding: utf-8 -*-
from functools import partial
import numpy as np

from endochrone.linear_regression import LinearRegression

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


# TODO: refecator to use optimise least squares - when written
class ArModel(LinearRegression):
    def __init__(self, *args, order=1, **kwargs):
        self.order = order
        super().__init__(*args, **kwargs)

    def fit(self, x):
        features = self.generate_lags(x)
        targets = self.generate_targets(x)
        super().fit(features, targets)

    def predict(self, x_test):
        if x_test.ndim == 1:
            features = self.generate_lags(x_test, include_last=True)
        else:
            features = x_test
        return super().predict(features)

    def generate_lags(self, x, include_last=False):
        o = self.order
        if include_last:
            N = len(x)
        else:
            N = len(x) - 1
        lags = [x[o-l-1:N-l, np.newaxis] for l in range(self.order)]
        return np.hstack(lags)

    def generate_targets(self, x):
        return np.array(x[self.order:])


class MaModel:
    def __init__(self, order=1):
        self.order = order

    def residuals(self, x, coefs):
        res = []
        for t in range(0, len(x)):
            N = min(t, self.order)
            centre = x[t] - coefs[0]
            adj = np.sum([coefs[s+1]*res[t-s-1] for s in range(N)])
            res.append(centre - adj)
        return res

    def fit(self, x):
        # iniitalise guesses for mu and thetas
        initial = np.array([np.mean(x)] + [0.5]*self.order)

        # TODO write our own least squares module
        from scipy.optimize import least_squares
        optim_kwds = dict(ftol=1e-10)
        err_func = partial(self.residuals, x)
        optimum = least_squares(err_func, initial, **optim_kwds)

        self.thetas_ = optimum.x
        self.residuals_ = optimum.fun
        return optimum.success

    def predict(self, y):
        if len(y) < self.order:
            raise ValueError("Insufficient values provided")
        preds = []
        errs = []
        for t in range(0, len(y) + self.order):
            N = min(t, self.order)
            adj = np.sum([self.thetas_[s+1]*errs[t-s-1] for s in range(N)])
            prediction = self.thetas_[0] - adj
            preds.append(prediction)
            if t < len(y):
                errs.append(prediction-y[t])
            else:
                errs.append(0)
        return preds[self.order:]
