# -*- coding: utf-8 -*-
import numpy as np
import random

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def calculate(data, k=3):
    centroids = initial_centroids(data, k)
    return centroids


def initial_centroids(data, k=3):
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


def recalculate_centroids(data, assignments, k=3):
    centroids = np.zeros((k, 2), dtype=float)
    for i in range(k):
        centroids[i, :] = np.mean(data[assignments[:, 0] == i], axis=0)
    return centroids
