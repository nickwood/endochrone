# -*- coding: utf-8 -*-
from math import exp
import numpy as np
import random

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def find_minimum(min_x, max_x, f, n_steps=None):
    x_old = x_best = random.randrange(min_x, max_x)
    f_old = f_best = f(x_old)

    if n_steps is None:
        k = int(np.log10(max_x - min_x) * 30)
    else:
        k = n_steps

    steps = [(0, x_old, f_old)]
    for i in range(k):
        temp = 1/(i+1)
        x_new = choose_neighbor(x_old, temp, min_x, max_x)
        f_new = f(x_new)
        if f_new < f_best:
            x_best, f_best = x_new, f_new

        if transition(f_old, f_new, temp):
            x_old, f_old = x_new, f_new
            steps.append((i, x_old, f_old))
    return x_best, f_best


def find_maximum(min_x, max_x, f, n_steps=None):
    def g(*args):
        return -1 * f(*args)
    x, y = find_minimum(min_x, max_x, g, n_steps)
    return x, -y


def choose_neighbor(x, temp, min_x, max_x):
    """Use Cauchy CDF to generate a random jump, ensuring it lies in
    [min_x, max_x]"""
    r = random.uniform(-1/2, 1/2)
    jump = temp * np.tan(r * np.pi) * max_x
    width = max_x - min_x
    return int((x - min_x + jump) % width + min_x)


def transition(f_old, f_new, temp):
    """Always transition if we're going to a lower energy state. If f_new is
    a higher energy, only transition with small probability. Temp is a float
    in the range [0,1] which asymptotically approaches 0."""
    if f_new < f_old:
        return True
    if temp != 0 and random.uniform(0, 1) < exp(-(f_new - f_old)/(temp)):
        return True
    else:
        return False
