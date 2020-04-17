# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np
import random

from endochrone.binary_decision_tree import BinaryDecisionTree

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class RandomForest:
    def __init__(self, n_trees, sample_size=None, feat_per_tree=None,
                 max_tree_depth=None):
        self.n_trees = n_trees
        self.samp_per_tree = sample_size
        self.feat_per_tree = feat_per_tree
        self.max_tree_depth = max_tree_depth
        self.trees = []

    def fit(self, x, y):
        n_samples = x.shape[0]
        if self.samp_per_tree is None:
            self.samp_per_tree = int(2 * n_samples / self.n_trees)

        self.trees = [BinaryDecisionTree(max_depth=self.max_tree_depth)
                      for i in range(self.n_trees)]

        for tree in self.trees:
            if self.feat_per_tree is None:
                x_feat = x
            else:
                x_feat = take_features(self.feat_per_tree, x, y)
            x_samp, y_samp = take_samples(self.samp_per_tree, x_feat, y)
            tree.fit(x_samp, y_samp)
        return self

    def predict(self, x):
        predictions = (np.transpose([t.predict(x) for t in self.trees]))
        return np.array([consensus(votes) for votes in predictions])


def consensus(votes):
    return Counter(votes).most_common(1)[0][0]
# TODO: something like :np.apply_along_axis(lambda x: np.bincount(x).argmax(),
# axis=0, arr=A)


def take_samples(sample_size, x, y):
    sample_indexes = random.choices(range(len(x)), k=sample_size)
    return x[sample_indexes], y[sample_indexes]


def take_features(n_features, x, y):
    sample_features = random.choices(range(len(x[0])), k=n_features)
    return x[:, sample_features]
