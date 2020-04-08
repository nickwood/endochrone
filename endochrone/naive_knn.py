# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter

from endochrone.measures import euclidean_dist

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def classify(x_train, y_train, x_test, k=3):
    return np.array([classify_point(x_train, y_train, point, k)
                     for point in x_test])[:, np.newaxis]


def classify_point(x_train, y_train, point, k):
    distances = [euclidean_dist(t_point, point) for t_point in x_train]
    return majority_concensus(y_train[np.argsort(distances)[:k]])


def majority_concensus(classifications):
    """provide the most common vote in the sequence, if there is a tie, uses
    the one that appears first"""
    return Counter(classifications[:, 0]).most_common(1)[0][0]
