# -*- coding: utf-8 -*-
import numpy as np
from typing import Iterable

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class BatchGradientDescent:
    def __init__(self, learning_rate=1, jac=None, tol=0.001,
                 max_iter=100):
        self.tol_ = tol
        self.max_iter = max_iter
        self.lr_ = learning_rate

    def fit(self, *, x0: dict,  funcs: Iterable[callable] = None, func=None):
        if func is not None:
            return self.fit_single(x0=x0, func=func)
        else:
            raise NotImplementedError

    def fit_single(self, *, x0: dict, func):
        t = 0
        x_prev = None
        x_next = x0
        rest = 0
        while diff_points(x_prev, x_next, self.tol_) and t < self.max_iter:
            x_prev = x_next
            t += 1
            jac_x = approx_jacobian([func], x_prev)[0]
            delta_x = -1 * self.lr_ * jac_x
            x_next_vals = np.fromiter(x_prev.values(), dtype=float) + delta_x
            x_try = dict(zip(x_prev.keys(), x_next_vals))
            if func(**x_try) > func(**x_prev):
                x_prev = None
                self.lr_ /= 2
            else:
                x_next = dict(zip(x_prev.keys(), x_next_vals))
                # once we're close we increase the lr to try and get closer
                if not diff_points(x_prev, x_next, self.tol_) and rest <= 3:
                    self.lr_ *= 10
                    rest += 1
                    x_prev = None

        self.minimum = func(**x_next)
        self.min_args = x_next
        self.n_steps = t
        if diff_points(x_prev, x_next, self.tol_):
            return False
        return True


def diff_points(x0, x1, tol):
    if x0 is None or x1 is None:
        return True
    diffs = np.abs(np.fromiter(x0.values(), dtype=float)
                   - np.fromiter(x1.values(), dtype=float))
    if np.any(diffs > tol):
        return True
    return False


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
