# -*- coding: utf-8 -*-
import numpy as np
import pytest
from unittest.mock import Mock

from endochrone.ensemble import random_forest as rf
from endochrone.utils import lazy_test_runner as ltr

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_fit_and_predict():
    N, S = 10, 6
    x = np.transpose([[1, 2, 3, 7, 8, 9, 10],
                      [3, 4, 5, 1, 2, 3, 4],
                      [3, 1, 5, 1, 2, 3, 4]])
    y = np.array([0, 0, 0, 1, 1, 1, 1])

    test_rf = rf.RandomForest(n_trees=N, sample_size=S)
    test_rf.fit(x, y)
    assert len(test_rf.trees) == N

    y_pred = test_rf.predict(x)
    assert np.count_nonzero(y == y_pred) >= 5


def test_no_sample_size_given():
    N = 10
    x = np.transpose([[1, 2, 3, 7, 8, 9, 10],
                      [3, 4, 5, 1, 2, 3, 4],
                      [3, 1, 5, 1, 2, 3, 4]])
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    test_rf = rf.RandomForest(n_trees=N)
    assert test_rf.samp_per_tree is None
    test_rf.fit(x, y)
    assert test_rf.samp_per_tree is not None


def test_concensus():
    assert rf.consensus(np.array([1, 1, 1, 1, 1])) == 1
    assert rf.consensus(np.array([1, 1, 1, 0, 0])) == 1
    assert rf.consensus(np.array([1, 0, 2, 2, 1])) == 1
    assert rf.consensus(np.array([2, 1, 2, 2, 1])) == 2
    assert rf.consensus(np.array([0, 0, 0, 2, 1])) == 0


def test_take_sample():
    x = np.transpose([[1, 2, 3, 7, 8, 9, 10],
                      [3, 4, 5, 1, 2, 3, 4],
                      [3, 1, 5, 1, 2, 3, 4]])
    y = np.array([0, 0, 0, 1, 1, 1, 1])

    s3_x, s3_y = rf.take_samples(3, x, y)
    assert s3_x.shape == (3, 3)
    assert s3_y.shape == (3,)

    xy = np.column_stack([x, y])
    s3_xy = np.column_stack([s3_x, s3_y])
    for point in s3_xy:
        assert point in xy

    s10_x, s10_y = rf.take_samples(10, x, y)
    assert s10_x.shape == (10, 3)
    assert s10_y.shape == (10,)

    xy = np.column_stack([x, y])
    s10_xy = np.column_stack([s10_x, s10_y])
    for point in s10_xy:
        assert point in xy


def test_take_features():
    x = np.transpose([[1, 2, 3, 7, 8, 9, 10],
                      [3, 4, 5, 1, 2, 3, 4],
                      [0, 1, 6, 6, 3, 5, 6]])
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    f2_x = rf.take_features(2, x, y)

    assert np.all(y == np.array([0, 0, 0, 1, 1, 1, 1]))
    assert f2_x.shape == (7, 2)
    assert np.all(f2_x.T[0] != f2_x.T[1])
    for feat in f2_x.T:
        assert feat in x.T

    with pytest.raises(ValueError):
        _ = rf.take_features(4, x, y)


def test_feature_selection():
    N, f = 10, 1
    x = np.transpose([[1, 2, 3, 7, 8, 9, 10],
                      [3, 4, 5, 1, 2, 3, 4],
                      [3, 1, 5, 1, 2, 3, 4]])
    y = np.array([0, 0, 0, 1, 1, 1, 1])

    test_rf = rf.RandomForest(n_trees=N, feat_per_tree=f)
    rf.take_features = Mock(wraps=rf.take_features)
    test_rf.fit(x, y)

    rf.take_features.assert_called_with(f, x, y)


def test_debug_print_statements(capsys):
    N = 3
    x = np.transpose([[1, 2, 3, 7, 8, 9, 10],
                      [3, 4, 5, 1, 2, 3, 4],
                      [3, 1, 5, 1, 2, 3, 4]])
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    test_rf = rf.RandomForest(n_trees=N)
    test_rf.fit(x, y, debug=True)
    captured = capsys.readouterr()
    printed = captured.out
    assert printed.startswith('tree fitted in ')


ltr()
