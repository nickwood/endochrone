# -*- coding: utf-8 -*-
from math import sqrt
import numpy as np

from endochrone.linear_regression import LinearRegression

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def covariance(x, y):
    return np.sum((x-np.mean(x))*(y-np.mean(y)))


def pearson(x, y):
    denominator = sqrt(covariance(x, x) * covariance(y, y))
    return covariance(x, y) / denominator


def partial_correlation(x, y, others):
    '''x and y are the two series we wish to calculate PACF for. Others is the
    remaining variables whose residuals we wish to remove'''
    x_other = LinearRegression(calculate_residuals=True)
    x_other.fit(others, x)
    x_res = x_other.residuals_

    y_other = LinearRegression(calculate_residuals=True)
    y_other.fit(others, y)
    y_res = y_other.residuals_

    return pearson(x_res, y_res)


def auto_correlation(x, lag):
    mu = np.mean(x)
    N = x.shape[0]
    numerator = N * np.sum((x[(lag):]-mu) * (x[:N-(lag)]-mu))
    denominator = (N - lag) * covariance(x, x)
    return numerator / denominator


def acf(x, lags=3):
    return np.array([auto_correlation(x, i) for i in range(0, lags+1)])
