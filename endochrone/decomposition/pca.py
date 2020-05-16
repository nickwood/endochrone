# -*- coding: utf-8 -*-
import numpy as np

from endochrone import Base

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class PCA(Base):
    def __init__(self, n_components=None):
        self.n_components = n_components
        super().__init__(properties={'requires_targets': False})

    def fit(self, X_train):
        self.validate_fit(features=X_train)
        if self.n_components is None:
            self.n_components == X_train.shape[1]

        cov_matrix = np.cov(X_train.transpose())
        eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
        eig_vecs = eig_vecs.transpose()
        var_ratios = eig_vals / np.sum(eig_vals)

        self.n_samples_, self.n_features_ = X_train.shape
        self.mean_ = np.mean(X_train, axis=0)

        comp_indx = np.argsort(-np.abs(eig_vals))[:self.n_components]
        self.explained_variance_ = eig_vals[comp_indx]
        self.explained_variance_ratio_ = var_ratios[comp_indx]
        self.components_ = eig_vecs[comp_indx]
        self.n_components_ = self.components_.shape[0]
        return self

    def transform(self, X_data):
        self.validate_predict(features=X_data)
        return (X_data - self.mean_)@np.transpose(self.components_)

    def inverse_transform(self, X_trans):
        return X_trans@self.components_ + self.mean_
