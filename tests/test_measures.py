# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone import measures
from endochrone.misc import lazy_test_runner as ltr

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_euclidian_dist():
    A = np.array([0, 0])
    B = np.array([3, 4])
    C = np.array([12, 10])
    D = np.array([0, 3])
    E = np.array([0.7, 2.8])

    assert measures.euclidean_dist(A, B)**2 == pytest.approx(25.0)
    assert measures.euclidean_dist(B, C)**2 == pytest.approx(117.0)
    assert measures.euclidean_dist(C, D)**2 == pytest.approx(193.0)
    assert measures.euclidean_dist(A, C)**2 == pytest.approx(244.0)
    assert measures.euclidean_dist(B, D)**2 == pytest.approx(10.0)
    assert measures.euclidean_dist(A, D)**2 == pytest.approx(9.0)
    assert measures.euclidean_dist(B, E)**2 == pytest.approx(6.73)


ltr('test_measures.py')
