# -*- coding: utf-8 -*-

import pytest
from endochrone.skeleton import fib

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)


def test_fake():
    assert True
