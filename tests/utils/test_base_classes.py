# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone.utils import lazy_test_runner as ltr
from endochrone import Base

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_validate_fit():
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

    test_base.validate_fit(features=X[:, np.newaxis], targets=Y)
    assert test_base.n_features_ == 1


def test_validate_predict():
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


ltr()
