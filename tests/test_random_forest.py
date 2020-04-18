# -*- coding: utf-8 -*-
import numpy as np

from endochrone import random_forest as rf
from endochrone.misc import lazy_test_runner as ltr

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


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

# TODO need more comprehensive tests


ltr()
