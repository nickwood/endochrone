# -*- coding: utf-8 -*-
from collections import defaultdict
import numpy as np

from endochrone import Base, Transformer
from endochrone.stats.measures import arg_nearest

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class TomekLinks(Base, Transformer):
    def __init__(self):
        super().__init__()

    def fit(self, *, features, targets):
        self.validate_fit(features=features, targets=targets)
        nn = defaultdict(int)
        nn_recip = defaultdict(set)
        n_samples = len(features)
        for i, f in enumerate(features):
            nearest = arg_nearest(X=features, p=f, n=n_samples)[1:]
            nn[i] = nearest
            nn_recip[nearest[0]].add(i)

        to_scan = set(range(n_samples))
        indexes_to_drop = set()
        while to_scan:
            i = to_scan.pop()
            j = nearest_unremoved_(nn[i], indexes_to_drop)

            if j in nn_recip[i] and targets[i] != targets[j]:
                to_scan.remove(j)
                # TODO: add logic to only drop certain classses
                dropped = [i, j]
                indexes_to_drop.update(dropped)

                '''We need to update nn_recip to remove references to removed
                features, we also recheck any features that had i or j as their
                nearest neigbour'''
                rescan = set()
                for k in dropped:
                    rescan |= nn_recip[k]
                    nn_recip.pop(k, None)
                rescan -= indexes_to_drop
                for k in rescan:
                    nn_recip[nearest_unremoved_(nn[k], indexes_to_drop)].add(k)
                to_scan |= rescan
        self.samples_to_drop_ = list(indexes_to_drop)

    def transform(self, *, features, targets):
        reduced_features = np.delete(features, self.samples_to_drop_, axis=0)
        reduced_targets = np.delete(targets, self.samples_to_drop_, axis=0)
        return reduced_features, reduced_targets


def nearest_unremoved_(ordered_nearest, removed):
    for i in ordered_nearest:
        if i not in removed:
            return i
