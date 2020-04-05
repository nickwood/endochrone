# -*- coding: utf-8 -*-
import numpy as np
import random

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def calculate(data, k=3):
    means = initial_means(data, k)
    return means


def initial_means(data, k=3):
    if data.shape[1] != 2 or data.shape[0] < k:
        raise ValueError

    rand_idx = random.sample(range(data.shape[0]), k)
    return data[rand_idx, :]
