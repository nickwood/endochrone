# -*- coding: utf-8 -*-

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class Base:
    def validate_fit(self, *, features, targets):
        if targets.ndim > 1:
            raise ValueError("targets must be 1 dimensional")
        if features.ndim == 1:
            raise ValueError("features must be 2+ dimensional")
        # TODO: make this one binary_classifier specific
        # if np.any(np.unique(targets) != np.arange(0, 2)):
        #     raise ValueError("targets must contain only 0 or 1")
        if features.shape[0] != targets.shape[0]:
            raise ValueError("X and Y must have same number of samples")
        self.n_features_ = features.shape[1]

    def validate_predict(self, *, features):
        if not hasattr(self, 'n_features_'):
            raise RuntimeError("Is this model fitted?")
        if features.ndim == 1:
            raise ValueError("features must be 2+ dimensional")
        if features.shape[1] != self.n_features_:
            raise ValueError("wrong number of features")
