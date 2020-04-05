# -*- coding: utf-8 -*-
import numpy as np
import random

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def calculate(data, k=3):
    means = initial_means(data, k)
    return means


def initial_means(data, k=3):
    if data.shape[1] != 2 or data.shape[0] < k:
        raise ValueError

    rand_idx = random.sample(range(data.shape[0]), k)
    return data[rand_idx, :]


def euclidean_dist(A, B):
    return np.sqrt(np.sum((A-B)**2))


def nearest_centroid(point, centroids):
    return np.argmin([euclidean_dist(point, c) for c in centroids])


def nearest_centroids(data, centroids):
    return np.transpose([[nearest_centroid(p, centroids) for p in data]])
