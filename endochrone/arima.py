# -*- coding: utf-8 -*-
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
        if order > 1:
            raise NotImplementedError("Only supports order 1 currently")
        self.order = order

    def residuals(self, coefs):
        res = [self.x[0] - coefs[0]]
        for t in range(self.order, len(self.x)):
            res.append((self.x[t] - coefs[0] - coefs[1]*res[t-1]))
        res = np.array(res)

        return res

    def fit(self, x):
        self.x = x
        # iniitalise guesses for mu and thetas
        initial = np.array([np.mean(x)] + [0.5]*self.order)

        # TODO write our own least squares module
        from scipy.optimize import least_squares
        optim_kwds = dict(ftol=1e-10)
        optimum = least_squares(self.residuals, initial, **optim_kwds)

        self.thetas_ = optimum.x
        self.residuals_ = optimum.fun
        print(optimum)
