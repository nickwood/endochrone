# -*- coding: utf-8 -*-
import numpy as np

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def min_max(data):
    col_min = np.min(data, axis=0)
    col_max = np.max(data, axis=0)
    return (data-col_min)/(col_max-col_min)


def mean_norm(data):
    col_min = np.min(data, axis=0)
    col_max = np.max(data, axis=0)
    col_mean = np.mean(data, axis=0)
    return (data-col_mean)/(col_max-col_min)
