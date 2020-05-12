# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone.utils import lazy_test_runner as ltr
import endochrone.optimisation.gradient_descent as gd
from endochrone.optimisation import BatchGradientDescent

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_approx_jacobian_1x2():
    def abs_function(*, a, b):
        return (np.abs(a - 1.5) + np.abs(b - 1.5))

    p0 = {'a': 1, 'b': 1}
    jac0 = gd.approx_jacobian([abs_function], p0)
    assert jac0.shape == (1, 2)
    assert np.all(jac0[0] == pytest.approx(-1.))

    p1 = {'a': 1, 'b': 2}
    jac1 = gd.approx_jacobian([abs_function], p1)
    assert jac1.shape == (1, 2)
    assert jac1[0, 0] == pytest.approx(-1.)
    assert jac1[0, 1] == pytest.approx(1.)

    p2 = {'a': 2, 'b': 2}
    jac2 = gd.approx_jacobian([abs_function], p2)
    assert jac2.shape == (1, 2)
    assert jac2[0, 0] == pytest.approx(1.)
    assert jac2[0, 1] == pytest.approx(1.)


def test_approx_jacobian_2x2():
    def f1(*, x, y):
        return (x**2 * y)

    def f2(*, x, y):
        return (5*x + np.sin(y))

    funcs = [f1, f2]

    p0 = {'x': 1, 'y': 1}
    jac0 = gd.approx_jacobian(funcs, p0)
    assert jac0.shape == (2, 2)
    assert jac0[0, 0] == pytest.approx(2., abs=0.0001)
    assert jac0[0, 1] == pytest.approx(1., abs=0.0001)
    assert jac0[1, 0] == pytest.approx(5., abs=0.0001)
    assert jac0[1, 1] == pytest.approx(0.540302, abs=0.0001)


def test_approx_jacobian_3x4():
    # https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant#Example_4
    def f1(*, x, y, z):
        return x

    def f2(*, x, y, z):
        return 5 * z

    def f3(*, x, y, z):
        return 4 * y**2 - 2 * z

    def f4(*, x, y, z):
        return z * np.sin(x)

    funcs = [f1, f2, f3, f4]

    p0 = {'x': np.pi, 'y': 1, 'z': 2}
    jac = gd.approx_jacobian(funcs, p0)
    assert jac.shape == (4, 3)
    exp = np.array([[1., 0., 0.],
                    [0., 0., 5.],
                    [0., 8., -2.],
                    [-2., 0., 0.]])
    assert np.all(jac == pytest.approx(exp, abs=0.0001))


def test_gradient_descent():
    def poly_function(*, a, b):
        return 1.75 + (a - 1.5)**2 + 3*(b - 0.75)**2

    x0 = {'a': 20, 'b': 150}
    gs_test = BatchGradientDescent(tol=0.0001)
    gs_test.fit(func=poly_function, x0=x0)

    assert gs_test.minimum == pytest.approx(1.75, abs=0.0001)
    assert gs_test.min_args['a'] == pytest.approx(1.5, abs=0.0001)
    assert gs_test.min_args['b'] == pytest.approx(0.75, abs=0.0001)


def test_rosen_func():
    def rosen(*, x, y):
        return (1-x)**2 + (y-x**2)**2

    x0 = {'x': 1.3, 'y': 0.7}

    gs_test = BatchGradientDescent(learning_rate=0.2, tol=0.001)
    assert gs_test.fit(func=rosen, x0=x0)
    assert gs_test.min_args['x'] == pytest.approx(1., abs=0.01)
    assert gs_test.min_args['y'] == pytest.approx(1., abs=0.01)


def test_non_convergence():
    def abs_function(*, x, y):
        return (x+y)

    x0 = {'x': 1.3, 'y': 0.7}

    gs_test = BatchGradientDescent(learning_rate=0.2, tol=0.001)
    assert gs_test.fit(func=abs_function, x0=x0) is False


ltr()
