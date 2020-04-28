# -*- coding: utf-8 -*-
import numpy as np

from endochrone.linear_regression import LinearRegression

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


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
