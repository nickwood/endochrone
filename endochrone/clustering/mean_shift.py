# -*- coding: utf-8 -*-
import numpy as np
from functools import partial

from endochrone import Base
from endochrone.stats.measures import euclidean_dist as dist
from endochrone.stats.measures import neighbours

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class MeanShift(Base):
    def __init__(self, bandwidth=None, kernel='flat'):
        self.bandwidth = bandwidth
        if kernel == 'flat':
            self.kernel = partial(flat)
        elif kernel == 'gaussian':
            self.kernel = partial(gaussian)
        else:
            raise ValueError("Unknown Kernel: %s" % kernel)
        super().__init__(properties={'requires_targets': False})

    def fit(self, X):
        self.validate_fit(features=X)
        points = np.copy(X).astype(float)
        prev = None
        bw = self.bandwidth
        while np.any(prev != points):
            prev = np.copy(points)
            for i, p in enumerate(prev):
                points[i] = self.kernel(prev, p, bw)
        self.centres_ = np.unique(points, axis=0)
        self.n_centres_ = len(self.centres_)
        self.labels_ = np.arange(0, self.n_centres_)
        return self.centres_, self.labels_

    def predict(self, X):
        self.validate_predict(features=X)
        return np.array([self.predict_point(p) for p in X])

    def predict_point(self, p):
        if len(self.labels_) == 1:
            return 0
        else:
            return np.argmin([dist(p, c) for c in self.centres_])


def flat(X, p, bandwidth):
    '''Return the centre of mass - i.e. mean - of points in X within
    neighbourhood = bandwidth'''
    neighbours_ = neighbours(X=X, p=p, size=bandwidth)
    return np.mean(neighbours_, axis=0, keepdims=True)


def gaussian(X, p, bandwidth):
    if X.ndim == 2:
        neighb_ = neighbours(X=X, p=p, size=bandwidth)
        sq_distances = np.sum((neighb_ - p)**2, axis=1)
        exponentials = np.exp((-1/2) * sq_distances / bandwidth**2)
        numerator = np.sum(neighb_ * exponentials[:, np.newaxis], axis=0)
        denominator = np.sum(exponentials)
        return numerator/denominator
    else:
        raise NotImplementedError("X dimension is too high, expected 2")
