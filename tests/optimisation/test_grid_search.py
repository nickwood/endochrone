# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone.utils.misc import lazy_test_runner as ltr
from endochrone.optimisation import GridSearch

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_abs():
    def abs_function(*, a):
        return (np.abs(a-1.5))
    assert abs_function(a=1.5) == 0
    search_space = {'a': np.arange(0.5, 3.0, 0.05)}

    gs_test = GridSearch(find_minimum=True, find_maximum=False)
    gs_test.fit(search_space, abs_function)

    assert gs_test.minimum == pytest.approx(0.0)
    assert gs_test.min_args == {'a': 1.5}
    assert gs_test.maximum is None
    assert gs_test.max_args == {}


def test_polynomials():
    def polynomial1(*, a, b):
        return (2*a**3 - 9*a**2 + 12*a)*(3*b - b**2)

    search_space = {'a': np.arange(0.6, 2.5, 0.1), 'b': np.arange(0.1, 2, 0.1)}

    gs_test = GridSearch()
    gs_test.fit(search_space, polynomial1)
    assert gs_test.minimum == pytest.approx(1.16)
    assert gs_test.min_args == {'a': 2.0, 'b': 0.1}
    assert gs_test.maximum == pytest.approx(11.25)
    assert gs_test.max_args == {'a': 1.0, 'b': 1.5}

    def polynomial2(*, a, b, c):
        return 1.75 + (a - 0.5)**2 + (b-0.75)**2 + (c - 0.25)**2

    search_space = {'a': np.arange(0.0, 2.5, 0.05),
                    'b': np.arange(0.0, 2, 0.05),
                    'c': np.arange(0.0, 1, 0.05)}

    gs_test = GridSearch(precision=5, find_maximum=False)
    gs_test.fit(search_space, polynomial2)
    assert gs_test.minimum == 1.75
    assert gs_test.min_args == {'a': 0.5, 'b': 0.75, 'c': 0.25}
    assert gs_test.maximum is None


ltr()
