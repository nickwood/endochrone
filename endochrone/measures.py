# -*- coding: utf-8 -*-
import numpy as np

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def euclidean_dist(A, B):
    return np.sqrt(np.sum((A-B)**2))
