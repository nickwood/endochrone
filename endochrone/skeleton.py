# -*- coding: utf-8 -*-
import logging

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def fib(n):
    """Fibonacci example function

    Args:
      n (int): integer

    Returns:
      int: n-th Fibonacci number
    """
    assert n > 0
    a, b = 1, 1
    for _ in range(n-1):
        a, b = b, a+b
    return a
