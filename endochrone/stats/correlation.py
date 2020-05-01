# -*- coding: utf-8 -*-
from functools import lru_cache
from math import sqrt
import numpy as np

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def covariance(x, y):
    return np.sum((x-np.mean(x))*(y-np.mean(y)))


def pearson(x, y):
    denominator = sqrt(covariance(x, x) * covariance(y, y))
    return covariance(x, y) / denominator


class acf:
    def __init__(self, x, lags=5):
        self.x = x
        self.mu = np.mean(x)
        self.cent_x = x - self.mu
        self.N = len(x)
        self.lags = lags

    @lru_cache(None)
    def auto_correlation(self, lag):
        cov_lagged = np.sum(self.cent_x[lag:] * self.cent_x[:self.N-lag])
        numerator = cov_lagged / (self.N - lag)
        denominator = covariance(self.x, self.x) / self.N
        return numerator / denominator

    @property
    def values(self):
        return np.array([self.auto_correlation(lag)
                         for lag in range(0, self.lags+1)])


class pacf(acf):
    '''Using formulae from http://pecar-uk.com/Autocorrelations.pdf §5.5'''

    @lru_cache(None)
    def p_val_(self, i, j):
        if i == j:
            return self.p_val_with_same_index(i)
        else:
            return self.reduced_p_val_(i, j)

    @lru_cache(None)
    def p_val_with_same_index(self, i):
        if i in [0, 1]:
            return self.auto_correlation(i)
        else:
            n_sum = np.sum([self.p_val_(i-1, j) * self.auto_correlation(i-j)
                            for j in range(1, i)])
            numerator = self.auto_correlation(i) - n_sum
            d_sum = np.sum([self.p_val_(i-1, j) * self.auto_correlation(j)
                            for j in range(1, i)])
            denominator = 1 - d_sum
            return numerator / denominator

    @lru_cache(None)
    def reduced_p_val_(self, i, j):
        return self.p_val_(i-1, j) - self.p_val_(i, i) * self.p_val_(i-1, i-j)

    @property
    def values(self):
        return np.array([self.p_val_(lag, lag)
                         for lag in range(0, self.lags+1)])

    @property
    def acf_values(self):
        return super().values
