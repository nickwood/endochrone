# -*- coding: utf-8 -*-
import numpy as np

from endochrone.stats.measures import euclidean_dist

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class KNearest:
    def __init__(self, *, k=3):
        self.k_ = k

    def fit(self, *, features, targets):
        #TODO check dimensions
        self.train_features_ = features
        self.train_targets_ = targets

    def predict(self, *, features):
        return np.array([self.classify_point(point=p) for p in features])

    def classify_point(self, *, point):
        distances = [euclidean_dist(t_point, point)
                     for t_point in self.train_features_]
        return majority(self.train_targets_[np.argsort(distances)[:self.k_]])


def majority(classifications):
    """provide the most common vote in the sequence, if there is a tie, uses
    the one that appears first alphabetically"""
    votes, counts = np.unique(classifications, return_counts=True)
    return votes[np.argmax(counts)]
