# -*- coding: utf-8 -*-
import numpy as np
import pytest
from unittest.mock import Mock

from endochrone.utils import lazy_test_runner as ltr
from endochrone.clustering import mean_shift as ms

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_flat_kernel():
    X = np.arange(0, 10)
    p = np.array([4.5])
    assert ms.flat(X, p, 2) == [4.5]

    X2d = np.arange(0, 20).reshape(10, 2)
    p = np.array([9, 9])
    assert np.all(ms.flat(X2d, p, 3) == np.array([9, 10]))

    X3d = np.arange(0, 30).reshape(10, 3)
    p = np.array([22, 23, 24])
    assert np.all(ms.flat(X3d, p, 4) == np.array([22.5, 23.5, 24.5]))

    Xnd = np.arange(0, 75).reshape(5, 5, 3)
    p = np.arange(31, 46).reshape(5, 3)
    exp = np.arange(37.5, 52, 1).reshape(5, 3)
    assert np.all(ms.flat(Xnd, p, 55) == exp)


def test_gaussian_kernel():
    X2d = np.arange(0, 20).reshape(10, 2)
    p = np.array([9, 9])
    exp = np.array([8.88934389, 9.88934389])
    assert np.all(ms.gaussian(X2d, p, 3) == pytest.approx(exp))
    assert np.all(ms.gaussian(X2d, p, 3) == pytest.approx(exp))
    exp = np.array([8.15192519, 9.15192519])
    assert np.all(ms.gaussian(X2d, p, 4) == pytest.approx(exp))
    assert np.all(ms.gaussian(X2d, p, 4) == pytest.approx(exp))

    with pytest.raises(NotImplementedError):
        X3d = np.arange(0, 8).reshape(2, 2, 2)
        p = np.array([[1, 1], [1, 1]])
        ms.gaussian(X3d, p, 4)


def test_1d_flat_fit_and_predict():
    X1 = np.arange(0, 6, 1).reshape(6, 1)
    ms.flat = Mock(wraps=ms.flat)
    clusters = ms.MeanShift(bandwidth=2.5)
    centres1, labels1 = clusters.fit(X1)
    ms.flat.assert_called()
    assert centres1 == np.array([2.5])
    assert labels1 == np.array([0])
    assert clusters.predict_point(np.array(3)) == 0
    assert clusters.predict_point(np.array(14)) == 0

    X2 = np.hstack([np.arange(0, 3, 1), np.arange(6, 9, 1)]).reshape(6, 1)
    clusters = ms.MeanShift(bandwidth=2, kernel='flat')
    centres2, labels2 = clusters.fit(X2)
    assert np.all(centres2 == np.array([1, 7]).reshape(2, 1))
    assert np.all(labels2 == np.array([0, 1]))
    assert clusters.predict_point(np.array(3)) == 0
    assert clusters.predict_point(np.array(5)) == 1


def test_2d_flat_fit_and_predict():
    X = np.arange(0, 60).reshape(30, 2)
    ms.flat = Mock(wraps=ms.flat)
    clusters = ms.MeanShift(bandwidth=20)
    centres, labels = clusters.fit(X)
    ms.flat.assert_called()
    exp = np.arange(29., 31.).reshape(1, 2)
    assert np.all(centres == pytest.approx(exp))
    assert labels == np.arange(0, 1)

    clusters2 = ms.MeanShift(bandwidth=14, kernel='flat')
    centres, labels = clusters2.fit(X)
    c1 = np.arange(14.11844117, 16)
    c2 = np.arange(29., 31.)
    c3 = np.arange(43.88155883, 45)
    exp = np.array([c1, c2, c3])
    assert np.all(centres == pytest.approx(exp))
    assert np.all(labels == np.arange(0, 3))

    to_predict = np.array([[11, 15], [24, 34], [37, 39]])
    assert np.all(clusters.predict(to_predict) == 0)
    assert np.all(clusters2.predict(to_predict) == np.array([0, 1, 2]))


def test_nd_flat_fit_and_predict():
    X = np.arange(0, 120).reshape(30, 2, 2)
    c = np.arange(58, 62).reshape(2, 2)
    ms.flat = Mock(wraps=ms.flat)
    clusters = ms.MeanShift(bandwidth=70)
    centres, labels = clusters.fit(X)
    ms.flat.assert_called()
    assert np.all(centres == c)
    assert labels == np.arange(0, 1)

    clusters2 = ms.MeanShift(bandwidth=43, kernel='flat')
    centres, labels = clusters2.fit(X)

    c1 = np.arange(35.43210209, 39).reshape(2, 2)
    c2 = np.arange(80.56789791, 84).reshape(2, 2)
    exp = np.array([c1, c2])

    assert np.all(centres == pytest.approx(exp))
    assert np.all(labels == np.arange(0, 2))

    to_predict = np.array([[[56, 56], [56, 56]], [[60, 60], [60, 60]]])
    assert np.all(clusters.predict(to_predict) == 0)
    assert np.all(clusters2.predict(to_predict) == np.array([0, 1]))


def test_1d_gaussian_fit_and_predict():
    X1 = np.arange(0, 6, 1).reshape(6, 1)
    ms.gaussian = Mock(wraps=ms.gaussian)
    clusters = ms.MeanShift(bandwidth=2, kernel='gaussian')
    centres1, labels1 = clusters.fit(X1)
    ms.gaussian.assert_called()
    assert centres1 == np.array([2.5])
    assert labels1 == np.array([0])

    X2 = np.hstack([np.arange(0, 3, 1), np.arange(6, 9, 1)]).reshape(6, 1)
    clusters2 = ms.MeanShift(bandwidth=2, kernel='gaussian')
    centres2, labels2 = clusters2.fit(X2)
    assert np.all(centres2 == pytest.approx(np.array([1, 7]).reshape(2, 1)))
    assert np.all(labels2 == np.array([0, 1]))

    to_predict = np.arange(3, 7).reshape(4, 1)
    assert np.all(clusters.predict(to_predict) == 0)
    exp = np.array([0, 0, 1, 1])
    assert np.all(clusters2.predict(to_predict) == exp)


def test_2d_gaussian_fit_and_predict():
    X = np.arange(0, 60).reshape(30, 2)
    ms.gaussian = Mock(wraps=ms.gaussian)
    clusters = ms.MeanShift(bandwidth=23, kernel='gaussian')
    centres, labels = clusters.fit(X)
    ms.gaussian.assert_called()
    exp = np.arange(29., 31.).reshape(1, 2)
    assert np.all(centres == pytest.approx(exp))
    assert labels == np.arange(0, 1)

    clusters2 = ms.MeanShift(bandwidth=14, kernel='gaussian')
    centres, labels = clusters2.fit(X)
    c1 = np.arange(13.60007775, 15)
    c2 = np.arange(29., 31.)
    c3 = np.arange(44.39992225, 46)
    exp = np.array([c1, c2, c3])
    assert np.all(centres == pytest.approx(exp))
    assert np.all(labels == np.arange(0, 3))

    to_predict = np.array([[11, 15], [24, 34], [37, 39]])
    assert np.all(clusters.predict(to_predict) == 0)
    assert np.all(clusters2.predict(to_predict) == np.array([0, 1, 2]))


def test_invalid_kernel():
    with pytest.raises(ValueError):
        _ = ms.MeanShift(bandwidth=23, kernel='test')


ltr()
