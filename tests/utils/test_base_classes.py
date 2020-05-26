# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone.utils import lazy_test_runner as ltr
from endochrone import Base, Transformer

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_validate_fit_defaults():
    test_base = Base()
    X = np.array([0, 1, 2, 3])
    Y = np.array([0, 0, 1, 1])

    # invalid X dimensions
    with pytest.raises(ValueError):
        test_base.validate_fit(features=X, targets=Y)

    # invalid Y dimensions
    with pytest.raises(ValueError):
        targ = Y[:, np.newaxis]
        test_base.validate_fit(features=X[:, np.newaxis], targets=targ)

    # incompatible sizes
    with pytest.raises(ValueError):
        targ = np.array([0, 1, 1])
        test_base.validate_fit(features=X[:, np.newaxis], targets=targ)

    # no targets given
    with pytest.raises(ValueError):
        test_base.validate_fit(features=X[:, np.newaxis])

    test_base.validate_fit(features=X[:, np.newaxis], targets=Y)
    assert test_base.n_features_ == 1


def test_validate_predict_defaults():
    test_base = Base()
    X = np.array([0, 1, 2, 3])
    Y = np.array([0, 0, 1, 1])

    # model not fit
    with pytest.raises(RuntimeError):
        test_base.validate_predict(features=X)

    test_base.validate_fit(features=X[:, np.newaxis], targets=Y)

    # invalid features shape
    with pytest.raises(ValueError):
        test_base.validate_predict(features=np.array([[0, 1], [1, 0]]))
    with pytest.raises(ValueError):
        test_base.validate_predict(features=np.array([0, 1, 0]))

    # valid n_features
    assert test_base.validate_predict(features=X[:, np.newaxis]) is None


def test_no_targets_needed():
    test_base = Base(properties={'requires_targets': False})
    X = np.array([[0, 1, 2, 3]])

    test_base.validate_fit(features=X)


def test_binary_only():
    test_base = Base(properties={'binary_targets': True})
    X = np.array([0, 1, 2, 3]).reshape(4, 1)
    Y = np.array([0, 1, 2, 3])

    with pytest.raises(ValueError):
        test_base.validate_fit(features=X, targets=Y)

    Y_bin = np.array([0, 1, 1, 0])
    test_base.validate_fit(features=X, targets=Y_bin)


def test_transformer():
    class no_targets(Transformer):
        def __init__(self):
            self.fitted = False
            self.transformed = False

        def fit(self, *, features):
            self.fitted = True

        def transform(self, *, features):
            self.transformed = True

    test_instance = no_targets()
    assert test_instance.fitted is False
    assert test_instance.transformed is False

    test_instance.fit_and_transform(features=6)
    assert test_instance.fitted
    assert test_instance.transformed

    class with_targets(Transformer):
        def __init__(self):
            self.fitted = False
            self.transformed = False

        def fit(self, *, features, targets):
            self.fitted = True

        def transform(self, *, features, targets):
            self.transformed = True

    test_instance2 = with_targets()
    assert test_instance2.fitted is False
    assert test_instance2.transformed is False

    test_instance2.fit_and_transform(features=6, targets=5)
    assert test_instance2.fitted
    assert test_instance2.transformed


ltr()
