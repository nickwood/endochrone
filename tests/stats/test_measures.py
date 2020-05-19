# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone.stats import measures
from endochrone.utils import lazy_test_runner as ltr

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


def test_euclidean_distances():
    X = np.arange(0, 10, 1).reshape(10, 1)
    exp = np.array([[5, 4, 3, 2, 1, 0, 1, 2, 3, 4]])
    assert np.all(measures.euclidean_distances(X=X, p=5) == exp)

    X2 = np.arange(0, 20, 1).reshape(10, 2)
    p2 = [3, 4]
    exp2 = np.array([4.24264069, 1.41421356, 1.41421356, 4.24264069,
                     7.07106781, 9.89949494, 12.72792206, 15.55634919,
                     18.38477631, 21.21320344])
    act2 = measures.euclidean_distances(X=X2, p=p2)
    assert np.all(act2 == pytest.approx(exp2))

    X3 = np.arange(0, 30, 1).reshape(10, 3)
    p3 = [3, 2, 4]
    exp3 = np.array([3.74165739, 2.23606798, 7.07106781, 12.20655562,
                     17.3781472, 22.56102835, 27.74887385, 32.93933818,
                     38.13135193, 43.32435804])
    act3 = measures.euclidean_distances(X=X3, p=p3)
    assert np.all(act3 == pytest.approx(exp3))


def test_arg_neighbours():
    X = np.arange(0, 1, 0.1)
    n1 = measures.arg_neighbours(X=X, p=np.array([0.3]), size=0.2)
    assert np.all(n1 == np.arange(1, 6, dtype=int))

    X = np.arange(0, 10, 1).reshape(10, 1)
    exp = np.array([2, 3, 4, 5])
    assert np.all(measures.arg_neighbours(X=X, p=3.5, size=2) == exp)
    exp = np.array([2, 3, 4, 5, 6, 7, 8])
    assert np.all(measures.arg_neighbours(X=X, p=5, size=3) == exp)

    X2 = np.arange(0, 20, 1).reshape(10, 2)
    p2 = [3.5, 2.3]
    exp2 = np.array([0, 1, 2])
    assert np.all(measures.arg_neighbours(X=X2, p=p2, size=4.5) == exp2)

    X3 = np.arange(0, 30, 1).reshape(10, 3)
    p3 = [3.5, 2.3, 4.2]
    exp3 = np.array([0, 1, 2])
    assert np.all(measures.arg_neighbours(X=X3, p=p3, size=8.5) == exp3)


def test_neighbours():
    X = np.arange(0, 10, 1).reshape(10, 1)
    exp = np.array([2, 3, 4, 5]).reshape(4, 1)
    assert np.all(measures.neighbours(X=X, p=3.5, size=2) == exp)
    exp = np.array([2, 3, 4, 5, 6, 7, 8]).reshape(7, 1)
    assert np.all(measures.neighbours(X=X, p=5, size=3) == exp)

    X2 = np.arange(0, 20, 1).reshape(10, 2)
    p2 = [3.5, 2.3]
    exp2 = np.array([[0, 1], [2, 3], [4, 5]])
    assert np.all(measures.neighbours(X=X2, p=p2, size=4.5) == exp2)

    X3 = np.arange(0, 30, 1).reshape(10, 3)
    p3 = [3.5, 2.3, 4.2]
    exp3 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert np.all(measures.neighbours(X=X3, p=p3, size=8.5) == exp3)


ltr()
