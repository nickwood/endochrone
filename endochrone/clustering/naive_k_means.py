# -*- coding: utf-8 -*-
import numpy as np

from endochrone import Base
from endochrone.stats.measures import euclidean_dist

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class KMeans(Base):
    def __init__(self, k=3):
        self.n_centroids_ = k
        super().__init__(properties={'requires_targets': False})

    def fit(self, *, features, initial_centroids=None):
        self.validate_fit(features=features)
        if initial_centroids is None:
            self.centroids = self.forgy_centroids_(features=features)
        else:
            self.centroids = initial_centroids

        old_centroids = np.zeros(self.centroids.shape)
        while np.any(old_centroids != self.centroids):
            old_centroids = self.centroids
            self.centroids = self.calculate_step(features=features)

    def forgy_centroids_(self, *, features):
        k = self.n_centroids_
        n_samples = features.shape[0]
        _indexes = np.random.choice(range(n_samples), k, replace=False)
        return features[_indexes, :]

    def calculate_step(self, *, features):
        n_c = self.nearest_centroids(features=features)
        return self.recalculate_centroids(features=features, assignments=n_c)

    def nearest_centroids(self, *, features):
        return np.array([self.nearest_centroid(point=p) for p in features])

    def nearest_centroid(self, *, point):
        return np.argmin([euclidean_dist(point, c) for c in self.centroids])

    def recalculate_centroids(self, *, features, assignments):
        return np.array([np.mean(features[assignments == i], axis=0)
                         for i in range(self.n_centroids_)])

    def predict(self, *, features):
        self.validate_predict(features=features)
        return self.nearest_centroids(features=features)
