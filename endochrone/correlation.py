# -*- coding: utf-8 -*-
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
