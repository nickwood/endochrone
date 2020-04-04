# -*- coding: utf-8 -*-
import numpy as np
import random

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def calculate(X, Y, k=3):

    return True


def initial_means(X, Y, k=3):
    if len(X) != len(Y):
        raise ValueError

    rand_idx = random.sample(range(len(X)), k)
    return np.transpose([(list(X)[i], list(Y)[i]) for i in rand_idx])
