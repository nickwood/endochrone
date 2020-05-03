# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np

from endochrone.optimisation.discrete_annealing import find_minimum

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class BinaryDecisionTree:
    def __init__(self, depth=1, max_depth=None):
        self.max_depth = max_depth
        self.depth = depth
        self.result = None
        self.left = None
        self.right = None

    def fit(self, x, y):
        if len(set(y)) == 1:
            self.result = y[0]
            self.size = len(y)
            return None
        if self.depth == self.max_depth:
            self.size = len(y)
            self.result = Counter(y).most_common(1)[0][0]
            return None

        split = max([(i, best_partition(feature, y))
                     for i, feature in enumerate(x.T)], key=lambda x: x[1][0])
        feat_idx, (_, val) = split

        if val is None:
            self.size = len(y)
            self.result = Counter(y).most_common(1)[0][0]
            return None
        else:
            self.split_feature = feat_idx
            self.split_value = val

        self.left = BinaryDecisionTree(depth=self.depth+1,
                                       max_depth=self.max_depth)
        self.right = BinaryDecisionTree(depth=self.depth+1,
                                        max_depth=self.max_depth)
        lx, ly, rx, ry = partition(x, y, self.split_feature, self.split_value)
        self.left.fit(lx, ly)
        self.right.fit(rx, ry)

    def predict(self, x):
        return np.array([self.predict_point(p) for p in x])

    def predict_point(self, p):
        if self.result is None:
            if p[self.split_feature] <= self.split_value:
                return self.left.predict_point(p)
            else:
                return self.right.predict_point(p)
        else:
            return self.result

    def __repr__(self, level=0):
        if self.result is not None:
            desc = "ret: %s (%s)" % (self.result, self.size)
        else:
            desc = "split: %s, %s" % (self.split_feature, self.split_value)
        ret = "\t"*(level) + '|->\t' + desc + "\n"
        if self.left is not None:
            ret += self.left.__repr__(level+1)
        if self.right is not None:
            ret += self.right.__repr__(level+1)
        return ret


def partition(x, y, feature_index, split_value):
    """return partitioned x & y corresponding to a split in the x's at feature
    with index and value specified. e.g. 'partition(x, y, 0, 2.5)' will split
    x & y in corresponding to x[0] being split at the value 2.5"""
    lte_indexes = x[:, feature_index] <= split_value
    return x[lte_indexes], y[lte_indexes], x[~lte_indexes], y[~lte_indexes]


def entropy(y):
    _, counts_y = np.unique(y, return_counts=True)
    n_samples = np.sum(counts_y)
    p_i = counts_y / n_samples
    return np.sum(p_i * np.log2(p_i) * -1)


def sorted_x_y(x_feat, y):
    ordering = np.argsort(x_feat)
    return (x_feat[ordering], y[ordering])


def generate_partitions(x_feat, y):
    sorted_x, sorted_y = sorted_x_y(x_feat, y)
    for i in range(1, len(sorted_y)):
        if sorted_x[i-1] != sorted_x[i]:
            yield sorted_y[:i], sorted_y[i:]


def weighted_partition_entropy(p_1, p_2):
    return (entropy(p_1)*len(p_1) + entropy(p_2)*len(p_2)) /\
           (len(p_1) + len(p_2))


def best_partition(x_feat, y):
    """Determine where best to split a single feature in order to maximise the
    information gain. Returns a tuple (i_gain, division). With a small dataset
    we brute force, otherwise use simulated annealing"""
    if len(np.unique(x_feat)) < 100:
        (split_entropy, division) = best_partition_small(x_feat, y)
    else:
        (split_entropy, division) = best_partition_large(x_feat, y)

    if division is None:
        return (0, None)
    else:
        pop_ent = entropy(y)
        return (pop_ent - split_entropy, division)


def best_partition_small(x_feat, y):
    """Use brute force to find the best place top split the given feature"""
    min_ent = None
    min_p_1 = None
    for p_1, p_2 in generate_partitions(x_feat, y):
        ent = weighted_partition_entropy(p_1, p_2)
        if min_ent is None or ent < min_ent:
            min_ent, min_p_1 = ent, len(p_1)
    if min_ent is None:
        return (0, None)
    else:
        division = np.mean(x_feat[np.argsort(x_feat)[min_p_1-1:min_p_1+1]])
        return (min_ent, division)


def best_partition_large(x_feat, y):
    """Use simulated annealing to find a close to optimum partition"""
    sorted_x, sorted_y = sorted_x_y(x_feat, y)
    unique_x, counts_x = np.unique(x_feat, return_counts=True)
    num_partitions = len(unique_x) - 2

    def f(partition_number):
        split_point = np.sum(counts_x[:partition_number+1])
        p_1, p_2 = sorted_y[:split_point], sorted_y[split_point:]
        return weighted_partition_entropy(p_1, p_2)

    partition_number, min_ent = find_minimum(0, num_partitions, f)
    x_split = np.mean(unique_x[partition_number:partition_number+2])
    return (min_ent, x_split)
