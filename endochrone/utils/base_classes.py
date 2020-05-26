# -*- coding: utf-8 -*-
import numpy as np

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


DEFAULT_PROPERTIES = {
    'binary_targets': False,
    'requires_targets': True
}


class Base:
    def __init__(self, properties: dict = {}):
        self.properties_ = DEFAULT_PROPERTIES.copy()
        for prop, val in properties.items():
            self.properties_[prop] = val

    def validate_fit(self, *, features, targets=None):
        if features.ndim == 1:
            raise ValueError("features must be 2+ dimensional")

        if self.properties_['requires_targets']:
            self.check_targets_(features=features, targets=targets)

        if self.properties_['binary_targets']:
            if np.any(np.unique(targets) != np.arange(0, 2)):
                raise ValueError("non-binary targets provided for binary "
                                 "estimator")

        self.n_features_ = features.shape[1]

    def validate_predict(self, *, features):
        if not hasattr(self, 'n_features_'):
            raise RuntimeError("Model is not fitted")
        if features.ndim == 1:
            raise ValueError("features must be 2+ dimensional")
        if features.shape[1] != self.n_features_:
            raise ValueError("wrong number of features")

    def check_targets_(self, *, features, targets=None):
        if targets is None:
            raise ValueError("no targets provided")
        if targets.ndim > 1:
            raise ValueError("targets must be 1 dimensional")
        if features.shape[0] != targets.shape[0]:
            raise ValueError("X and Y must have same number of samples")


class Transformer:
    def fit_and_transform(self, *, features, targets=None, **fit_params):
        self.fit(features=features, targets=targets, **fit_params)
        return self.transform(features=features, targets=targets)
