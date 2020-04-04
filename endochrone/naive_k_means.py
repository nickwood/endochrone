# -*- coding: utf-8 -*-
import numpy as np
import random

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def calculate(X, Y, k=3):

    return True


def initial_means(X, Y, k=3):
    return np.array([random.sample(list(X), k=k), random.sample(list(Y), k=k)])
