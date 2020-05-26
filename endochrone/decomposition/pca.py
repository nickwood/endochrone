# -*- coding: utf-8 -*-
import numpy as np

from endochrone import Base, Transformer

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class PCA(Base, Transformer):
    def __init__(self, n_components=None):
        self.n_components = n_components
        super().__init__(properties={'requires_targets': False})

    def fit(self, *, features):
        self.validate_fit(features=features)
        if self.n_components is None:
            self.n_components == features.shape[1]

        cov_matrix = np.cov(features.transpose())
        eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
        eig_vecs = eig_vecs.transpose()
        var_ratios = eig_vals / np.sum(eig_vals)

        self.n_samples_, self.n_features_ = features.shape
        self.mean_ = np.mean(features, axis=0)

        comp_indx = np.argsort(-np.abs(eig_vals))[:self.n_components]
        self.explained_variance_ = eig_vals[comp_indx]
        self.explained_variance_ratio_ = var_ratios[comp_indx]
        self.components_ = eig_vecs[comp_indx]
        self.n_components_ = self.components_.shape[0]
        return self

    def transform(self, *, features):
        self.validate_predict(features=features)
        return (features - self.mean_)@np.transpose(self.components_)

    def reverse(self, *, features):
        return features@self.components_ + self.mean_
