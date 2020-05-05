# -*- coding: utf-8 -*-
import numpy as np
from functools import partial

from endochrone.stats.measures import euclidean_dist as dist

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class MeanShift:
    def __init__(self, bandwidth=None, kernel='flat'):
        self.bandwidth = bandwidth
        if kernel == 'flat':
            self.kernel = partial(flat)
        elif kernel == 'gaussian':
            self.kernel = partial(gaussian)
        else:
            raise ValueError("Unknown Kernel: %s" % kernel)

    def fit(self, X):
        points = np.copy(X).astype(float)
        prev = None
        bw = self.bandwidth
        while np.any(prev != points):
            prev = np.copy(points)
            for i, p in enumerate(prev):
                points[i] = self.kernel(prev, p, bw)
        self.centres_ = np.unique(points, axis=0)
        self.labels_ = np.arange(0, len(self.centres_))
        return self.centres_, self.labels_

    def predict(self):
        # TODO
        pass


def neighbours(X, p, bandwidth):
    '''Returns a list of points in "X" that are within a distance "bandwidth"
    of point "p"'''
    return X[np.array([dist(p, comp) for comp in X]) <= bandwidth]


def flat(X, p, bandwidth):
    '''Return the centre of mass - i.e. mean - of points in X within
    neighbourhood = bandwidth'''
    return np.mean(neighbours(X, p, bandwidth), axis=0, keepdims=True)


def gaussian(X, p, bandwidth):
    if X.ndim == 1:
        return gaussian_1d(X, p, bandwidth)
    elif X.ndim == 2:
        return gaussian_2d(X, p, bandwidth)
    else:
        raise NotImplementedError("X dimension is too high, expected 1 or 2")


def gaussian_1d(X, p, bandwidth):
    neighb = neighbours(X, p, bandwidth)
    sq_distances = (neighb-p)**2
    exponentials = np.exp((-1/2)*sq_distances/bandwidth**2)
    numerator = np.sum(neighb*exponentials, axis=0)
    denominator = np.sum(exponentials)
    return numerator/denominator


def gaussian_2d(X, p, bandwidth):
    neighb = neighbours(X, p, bandwidth)
    sq_distances = np.sum((neighb-p)**2, axis=1)
    exponentials = np.exp((-1/2)*sq_distances/bandwidth**2)
    numerator = np.sum(neighb*exponentials[:, np.newaxis], axis=0)
    denominator = np.sum(exponentials)
    return numerator/denominator
