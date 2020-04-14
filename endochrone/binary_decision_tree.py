# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def gini_score(p_1, p_2):
    """given two partitions of classification data, return the Gini score of
    the given split"""
    p1_counts, p2_counts = Counter(p_1), Counter(p_2)
    p1_sum, p2_sum = sum(p1_counts.values()), sum(p2_counts.values())
    tot_counts, n_samples = p1_counts + p2_counts, p1_sum + p2_sum

    g_i = np.array([1 - (p1_counts[k]/p1_sum)**2 - (p2_counts[k]/p2_sum)**2
                    for k in tot_counts.keys()])
    posterior_probs = np.array([tot_counts[k] / n_samples
                                for k in tot_counts.keys()])
    return np.sum((g_i * posterior_probs))


def partition(x, y, index, value):
    """return partitioned y corresponding to a split in the x at feature with
    index and value specified. e.g. 'partition(x, y, 0, 2.5)' will split the y
    values corresponding to x[0] being split at the value 2.5"""
    lte = x[:, index] <= value
    return np.extract(lte, y), np.extract(~lte, y)
