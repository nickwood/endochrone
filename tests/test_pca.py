# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pytest

from endochrone import pca

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_2d_to_1d():
    min_x, max_x, gradient, intercept, n_samples = 10, 30, 1.5, 20, 300
    X_1 = np.random.uniform(min_x, max_x, n_samples)[:, np.newaxis]
    X_2_exact = gradient * X_1 + intercept
    flat_noise = np.random.standard_normal(size=(n_samples, 1))
    X_2_noise = flat_noise * (10 - X_1) * (X_1 - 30) / 40
    X_2 = X_2_exact + X_2_noise
    plt.scatter(X_1, X_2, color='green', s=1)
    X_train = np.concatenate([X_1, X_2], axis=1)

    # from sklearn.decomposition import PCA as skpca
    # sk_model = skpca(n_components=2)
    # sk_model.fit(X_train)

    pca_model = pca.PCA(n_components=1)
    pca_model.fit(X_train)

    assert pca_model.eig_val_.shape == (1,)
    assert pca_model.eig_val_[0] >= 80
    assert pca_model.eig_vec_.shape == (1, 2)
    principle = pca_model.eig_vec_[0]
    assert principle[1]/principle[0] == pytest.approx(1.5, abs=0.2)
    assert principle[1]**2 + principle[0]**2 == pytest.approx(1)

    center = np.mean(X_train, axis=0)
    for value, vector in zip(pca_model.eig_val_, pca_model.eig_vec_):
        points = np.vstack([center, center + (vector*value)/5])
        plt.plot(points[:, 0], points[:, 1])
    # plt.show()


test_2d_to_1d()
