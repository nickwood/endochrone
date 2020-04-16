# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np
import random

from endochrone.binary_decision_tree import BinaryDecisionTree

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class RandomForest:
    def __init__(self, n_trees, sample_size=None, max_tree_depth=None):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.max_tree_depth = max_tree_depth
        self.trees = []

    def fit(self, x, y):
        if self.sample_size is None:
            self.sample_size = 2 / self.n_trees

        self.trees = [BinaryDecisionTree(max_depth=self.max_tree_depth)
                      for i in range(self.n_trees)]

        for tree in self.trees:
            x_samp, y_samp = take_sample(self.sample_size, x, y)
            tree.fit(x_samp, y_samp)
        return self

    def predict(self, x):
        predictions = (np.transpose([t.predict(x) for t in self.trees]))
        return np.array([consensus(votes) for votes in predictions])


def consensus(votes):
    return Counter(votes).most_common(1)[0][0]


def take_sample(sample_size, x, y):
    sample_indexes = random.choices(range(len(x)), k=sample_size)
    return x[sample_indexes], y[sample_indexes]
