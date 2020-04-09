# -*- coding: utf-8 -*-
import numpy as np

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X_train):
        cov_matrix = np.cov(X_train.transpose())
        eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
        eig_vecs = eig_vecs.transpose()

        comp_indx = np.argsort(-np.abs(eig_vals))[:self.n_components]
        self.eig_val_ = eig_vals[comp_indx]
        self.eig_vec_ = eig_vecs[comp_indx]
        return self
