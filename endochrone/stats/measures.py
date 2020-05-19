# -*- coding: utf-8 -*-
import numpy as np

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def euclidean_dist(A, B):
    return np.sqrt(np.sum((A-B)**2))


def euclidean_distances(*, X, p):
    '''return an array of distances of each X from p'''
    return np.array([euclidean_dist(p, com) for com in X])


def arg_neighbours(*, X, p, bandwidth):
    '''Return the indices of points in X which are within 'bandwidth' of p'''
    return np.nonzero(euclidean_distances(p=p, X=X) <= bandwidth)[0]


def neighbours(*, X, p, bandwidth):
    '''Returns a list of points in "X" that are within a distance "bandwidth"
    of point "p"'''
    return X[arg_neighbours(X=X, p=p, bandwidth=bandwidth)]
