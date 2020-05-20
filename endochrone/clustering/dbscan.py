# -*- coding: utf-8 -*-
import numpy as np

from endochrone import Base
from endochrone.stats.measures import arg_neighbours

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class DBSCAN(Base):
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        super().__init__(properties={'requires_targets': False})

    def fit(self, *, features, targets=None):
        self.validate_fit(features=features, targets=targets)
        if targets is not None:
            raise NotImplementedError('Providing targets is not yet supported')

        self.targets_ = np.full(features.shape[0], -1, dtype=int)
        self.features_ = features.copy()
        self.max_label = -1

        for i, point in enumerate(features):
            if self.targets_[i] == -1:
                self.explore_neighbourhood_(index=i)

        self.denoise_()

    def predict(self, *, features):
        self.validate_predict(features=features)
        return np.array([self.predict_single(p=p) for p in features])

    def predict_single(self, *, p):
        neighb = arg_neighbours(X=self.features_, p=p, size=self.eps)
        labels, counts = np.unique(self.targets_[neighb], return_counts=True)
        if counts:
            return labels[np.argmax(counts)]
        else:
            return -1

    def explore_neighbourhood_(self, *, index, in_cluster=False):
        if self.targets_[index] != -1:
            return None

        neighbours = arg_neighbours(X=self.features_, p=self.features_[index],
                                    size=self.eps)

        if in_cluster or len(neighbours) >= self.min_samples:
            if not in_cluster:
                self.max_label += 1
            self.targets_[index] = self.max_label
            for n in neighbours:
                self.explore_neighbourhood_(index=n, in_cluster=True)

    def denoise_(self):
        '''Once our model is fitted, we remove any features still marked as
        noise - these aren't needed and would just increase the number of
        comparisons needed when making predictions'''
        noisy_indexes = np.nonzero(self.targets_ == -1)[0]
        self.features_ = np.delete(self.features_, noisy_indexes, axis=0)
        self.targets_ = np.delete(self.targets_, noisy_indexes, axis=0)
