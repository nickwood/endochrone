# -*- coding: utf-8 -*-
import json
import numpy as np

from endochrone.optimisation import discrete_annealing as da
from endochrone.utils import lazy_test_runner as ltr

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


with open('tests/resources/discrete_annealing_data') as json_file:
    f_vals = json.load(json_file)


def test_find_minimum():
    def f(x):
        return f_vals[int(x)]

    # Since this is non-deterministic, we run it a bunch of times and
    # expect enough of them to pass
    n = 10
    y_s = np.array([da.find_minimum(0, len(f_vals)-1, f)[1] for _ in range(n)])
    assert np.count_nonzero(y_s < 2.1) >= 0.8 * n


def test_find_maximum():
    def g(x):
        return -1 * f_vals[int(x)]

    # Since this is non-deterministic, we run it a bunch of times and
    # expect enough of them to pass
    n = 10
    y_s = np.array([da.find_maximum(0, len(f_vals)-1, g)[1] for _ in range(n)])
    assert np.count_nonzero(y_s > -2.1) >= 0.8 * n


def test_specify_steps():

    def f(x):
        f.count += 1
        return f_vals[int(x)]

    f.count = 0

    # Since this is non-deterministic, we run it a bunch of times and
    # expect enough of them to pass
    da.find_maximum(0, len(f_vals)-1, f, n_steps=3)
    assert f.count == 4


def test_div_zero_in_transition():
    assert da.transition(0.0, -1.0, 0.1)
    assert da.transition(0.0, 11.0, 0.001) is False
    assert da.transition(0.0, 11.0, 0.0) is False


ltr()
