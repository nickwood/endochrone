# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter

from endochrone.measures import euclidean_dist

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def classify(x_train, y_train, test_data, k=3):
    return np.transpose([[classify_point(x_train, y_train, point, k)
                        for point in test_data]])


def classify_point(x_train, y_train, point, k):
    distances = [euclidean_dist(t_point, point) for t_point in x_train]
    sort_order = np.argsort(distances)
    return majority_concensus(y_train[sort_order[:k]])


def majority_concensus(classifications):
    """provide the most common vote in the sequence, if there is a tie, uses
    the one that appears first"""
    return Counter(classifications[:, 0]).most_common(1)[0][0]
