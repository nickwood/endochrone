# -*- coding: utf-8 -*-
import numpy as np
from typing import Iterable

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class GradientDescent:
    def __init__(self):
        raise NotImplementedError


def approx_jacobian(functions: Iterable[callable], point: dict, delta=0.00001):
    '''Calculates the approximate jacobian at a specified point. Note that we
    take an iterable of functions so we can generate the Jacobian for
    non-scalar functions
    e.g:
    def f(*, a, b):
        return (np.abs(a - 1.5) + np.abs(b - 1.5))
    point = {'a': 1, 'b': 1}
    approx_jacobian([f], point) ~> np.array([[-1, -1]])'''

    N = len(point.keys())
    M = len(functions)
    jac = np.zeros((M, N))

    f_initial = np.array([functions[j](**point) for j in range(M)])

    for i, k in enumerate(point.keys()):
        delta_point = dict(point, **{k: point[k] + delta})
        delta_f = np.array([functions[j](**delta_point) for j in range(M)])
        jac[:, i] = (delta_f - f_initial) / delta
    return jac
