# -*- coding: utf-8 -*-
import numpy as np

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def softmax(x, axis=None):
    divisors = np.sum(np.exp(x), axis=axis, keepdims=True)
    return np.exp(x) / divisors
