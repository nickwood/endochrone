# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np

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
    """Determine where best to split a single feature in order to maximise the
    information gain. Returns a tuple (i_gain, division)"""
    pop_ent = entropy(y)
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
        return ((pop_ent - min_ent), division)
