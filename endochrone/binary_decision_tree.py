# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def entropy(y):
    counts = Counter(y)
    n_samples = sum(counts.values())
    p_i = [counts[k]/n_samples for k in counts.keys()]
    return np.sum(p_i * np.log2(p_i) * -1)


def generate_partitions(x_feat, y):
    ordering = np.argsort(x_feat)
    sorted_y = (y[ordering])
    for i in range(1, len(sorted_y)):
        if x_feat[ordering[i-1]] != x_feat[ordering[i]]:
            yield sorted_y[:i], sorted_y[i:]


def weighted_partition_entropy(p_1, p_2):
    return (entropy(p_1)*len(p_1) + entropy(p_2)*len(p_2)) /\
           (len(p_1) + len(p_2))


def best_partition(x_feat, y):
    pop_ent = entropy(y)
    min_ent = None
    min_p_1 = None
    for p_1, p_2 in generate_partitions(x_feat, y):
        ent = weighted_partition_entropy(p_1, p_2)
        if min_ent is None or ent < min_ent:
            min_ent, min_p_1 = ent, len(p_1)
    division = np.mean(x_feat[np.argsort(x_feat)[min_p_1-1:min_p_1+1]])
    return (division, (pop_ent - min_ent))
