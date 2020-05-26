# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone.decomposition import PCA
from endochrone.utils import lazy_test_runner as ltr

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
    X_train = np.concatenate([X_1, X_2], axis=1)

    pca_model = PCA(n_components=1)
    pca_model.fit(features=X_train)

    assert pca_model.n_components_ == 1
    assert pca_model.n_samples_ == n_samples
    assert pca_model.n_features_ == 2
    assert pca_model.explained_variance_.shape == (1,)
    assert pca_model.explained_variance_[0] >= 80
    assert pca_model.explained_variance_ratio_.shape == (1,)
    assert pca_model.explained_variance_ratio_[0] >= 0.98
    assert pca_model.components_.shape == (1, 2)

    principle = pca_model.components_[0]
    assert principle[1]/principle[0] == pytest.approx(gradient, abs=0.2)
    assert principle[1]**2 + principle[0]**2 == pytest.approx(1)

    X_transform = pca_model.transform(features=X_train)
    assert X_transform.shape == (n_samples, 1)
    assert np.mean(X_transform) == pytest.approx(0)


def test_zero_components_specified():
    min_x, max_x, gradient, intercept, n_samples = 10, 30, 4, 20, 300
    X_1 = np.random.uniform(min_x, max_x, n_samples)[:, np.newaxis]
    X_2_exact = gradient * X_1 + intercept
    flat_noise = np.random.standard_normal(size=(n_samples, 1))
    X_2_noise = flat_noise * (10 - X_1) * (X_1 - 30) / 40
    X_2 = X_2_exact + X_2_noise
    X_train = np.concatenate([X_1, X_2], axis=1)

    pca_model = PCA()
    pca_model.fit(features=X_train)

    assert pca_model.n_components_ == 2
    assert pca_model.n_samples_ == n_samples
    assert pca_model.n_features_ == 2
    assert pca_model.explained_variance_.shape == (2,)
    assert pca_model.explained_variance_ratio_.shape == (2,)
    assert sum(pca_model.explained_variance_ratio_) == pytest.approx(1)
    assert pca_model.components_.shape == (2, 2)


def test_6d_to_2d():
    n_samples = 300
    min_x, max_x, gradient_1, gradient_2, intercept = 10, 30, 1.5, 0.6, 20
    X_1 = np.random.uniform(min_x, max_x, n_samples)[:, np.newaxis]
    X_2 = np.random.uniform(min_x, max_x, n_samples)[:, np.newaxis]
    noise = 5 * np.random.standard_normal(size=(n_samples, 1))
    X_3 = gradient_1 * X_1 + intercept + noise
    X_4 = gradient_2 * X_2 + intercept + noise
    X_5 = X_3 + X_2
    X_6 = X_4 + X_1
    X_train = np.concatenate([X_1, X_2, X_3, X_4, X_5, X_6], axis=1)

    pca_model = PCA(n_components=2)
    pca_model.fit(features=X_train)

    assert pca_model.n_components_ == 2
    assert pca_model.n_samples_ == n_samples
    assert pca_model.n_features_ == 6
    assert pca_model.explained_variance_.shape == (2,)
    assert pca_model.explained_variance_ratio_.shape == (2,)
    assert np.abs(np.sum(pca_model.explained_variance_ratio_)) > 0.92
    assert pca_model.components_.shape == (2, 6)


def test_accuracy_of_inversion():
    n_features, n_samples = 6, 300
    X_train = np.random.rand(n_samples, n_features)

    pcam = PCA(n_components=n_features)
    pcam.fit(features=X_train)

    act = pcam.reverse(features=pcam.transform(features=X_train))
    assert np.all(act == pytest.approx(X_train))


ltr()
