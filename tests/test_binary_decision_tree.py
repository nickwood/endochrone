# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.datasets import make_blobs

from endochrone import binary_decision_tree as bdt
from endochrone.misc import lazy_test_runner as ltr

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_entropy():
    assert bdt.entropy([0]*14 + [1]*16) == pytest.approx(0.9967916319816366)
    assert bdt.entropy([0]*16 + [1]*14) == pytest.approx(0.9967916319816366)
    assert bdt.entropy([0]*12 + [1]*1) == pytest.approx(0.39124356362925566)
    assert bdt.entropy([0]*1 + [1]*12) == pytest.approx(0.39124356362925566)
    assert bdt.entropy([0]*4 + [1]*13) == pytest.approx(0.7871265862012691)
    assert bdt.entropy([0]*13 + [1]*4) == pytest.approx(0.7871265862012691)
    assert bdt.entropy([0]*8 + [1]*8) == pytest.approx(1.0)
    assert bdt.entropy([0]*2 + [1]*3 + [3]*3) == pytest.approx(1.5612781244591)
    assert bdt.entropy([0]*6 + [1]*8 + [3]*8) == pytest.approx(1.5726236638951)
    assert bdt.entropy([0]*1 + [1]*2 + [3]*5) == pytest.approx(1.2987949406953)
    assert bdt.entropy(['ca']*9 + ['bd']*4) == pytest.approx(0.890491640219491)
    assert bdt.entropy(['ca']*4 + ['bd']*4) == pytest.approx(1.0)
    assert bdt.entropy(['ca']*2 + ['bd']*9) == pytest.approx(0.684038435639041)


def test_generate_partitions():
    x = np.transpose([[1, 2, 3, 7, 8, 9, 10],
                      [3, 4, 5, 1, 2, 3, 4],
                      [3, 1, 5, 1, 2, 3, 4]])
    y_1 = np.array([0, 0, 0, 1, 1, 1, 1])

    x_parts_0 = list(bdt.generate_partitions(x[:, 0], y_1))
    assert len(x_parts_0) == 6
    assert np.all(x_parts_0[0][0] == np.array([0]))
    assert np.all(x_parts_0[0][1] == np.array([0, 0, 1, 1, 1, 1]))
    assert np.all(x_parts_0[2][0] == np.array([0, 0, 0]))
    assert np.all(x_parts_0[2][1] == np.array([1, 1, 1, 1]))
    assert np.all(x_parts_0[4][0] == np.array([0, 0, 0, 1, 1]))
    assert np.all(x_parts_0[4][1] == np.array([1, 1]))

    x_parts_1 = list(bdt.generate_partitions(x[:, 1], y_1))
    assert len(x_parts_1) == 4
    assert np.all(x_parts_1[0][0] == np.array([1]))
    assert np.all(x_parts_1[0][1] == np.array([1, 0, 1, 0, 1, 0]))
    assert np.all(x_parts_1[2][0] == np.array([1, 1, 0, 1]))
    assert np.all(x_parts_1[2][1] == np.array([0, 1, 0]))
    assert np.all(x_parts_1[3][0] == np.array([1, 1, 0, 1, 0, 1]))
    assert np.all(x_parts_1[3][1] == np.array([0]))

    x_parts_2 = list(bdt.generate_partitions(x[:, 2], y_1))
    assert len(x_parts_2) == 4
    assert np.all(x_parts_2[0][0] == np.array([0, 1]))
    assert np.all(x_parts_2[0][1] == np.array([1, 0, 1, 1, 0]))
    assert np.all(x_parts_2[1][0] == np.array([0, 1, 1]))
    assert np.all(x_parts_2[1][1] == np.array([0, 1, 1, 0]))
    assert np.all(x_parts_2[3][0] == np.array([0, 1, 1, 0, 1, 1]))
    assert np.all(x_parts_2[3][1] == np.array([0]))

    y_2 = np.array([0, 0, 1, 1, 2, 2, 2])
    x_parts_0 = list(bdt.generate_partitions(x[:, 0], y_2))
    assert np.all(x_parts_0[0][0] == np.array([0]))
    assert np.all(x_parts_0[0][1] == np.array([0, 1, 1, 2, 2, 2]))
    x_parts_1 = list(bdt.generate_partitions(x[:, 1], y_2))
    assert len(x_parts_1) == 4
    assert np.all(x_parts_1[0][0] == np.array([1]))
    assert np.all(x_parts_1[2][0] == np.array([1, 2, 0, 2]))
    x_parts_2 = list(bdt.generate_partitions(x[:, 2], y_2))
    assert len(x_parts_2) == 4
    assert np.all(x_parts_2[0][0] == np.array([0, 1]))
    assert np.all(x_parts_2[2][0] == np.array([0, 1, 2, 0, 2]))

    y_3 = np.array(['cat']*3 + ['dog']*2 + ['cow']*2)
    x_parts_0 = list(bdt.generate_partitions(x[:, 0], y_3))
    assert np.all(x_parts_0[0][0] == np.array(['cat']))
    assert np.all(x_parts_0[2][0] == np.array(['cat']*3))
    x_parts_1 = list(bdt.generate_partitions(x[:, 1], y_3))
    assert len(x_parts_1) == 4
    assert np.all(x_parts_1[0][0] == np.array(['dog']))
    assert np.all(x_parts_1[2][0] == np.array(['dog', 'dog', 'cat', 'cow']))
    x_parts_2 = list(bdt.generate_partitions(x[:, 2], y_3))
    assert len(x_parts_2) == 4
    assert np.all(x_parts_2[0][0] == np.array(['cat', 'dog']))
    assert np.all(x_parts_2[2][1] == np.array(['cow', 'cat']))


def test_weighted_partition_entropy():
    p_1_1 = [0]*12 + [1]*1
    p_1_2 = [0]*4 + [1]*13
    e_1 = 0.6155772764200633
    assert bdt.weighted_partition_entropy(p_1_1, p_1_2) == pytest.approx(e_1)

    p_2_1 = [0]*12 + [1]*3
    p_2_2 = [0]*2 + [1]*13
    e_2 = 0.6442188007201338
    assert bdt.weighted_partition_entropy(p_2_1, p_2_2) == pytest.approx(e_2)

    p_3_1 = [0]*12
    p_3_2 = [1]*13
    e_3 = 0
    assert bdt.weighted_partition_entropy(p_3_1, p_3_2) == pytest.approx(e_3)

    p_m_1 = [0]*10 + [1]*4 + [2]*1
    p_m_2 = [0]*1 + [1]*4 + [2]*9
    e_m = 1.177852478657924
    assert bdt.weighted_partition_entropy(p_m_1, p_m_2) == pytest.approx(e_m)

    p_t1_1 = ['cat']*12 + ['cow']*1
    p_t1_2 = ['cat']*4 + ['cow']*13
    assert bdt.weighted_partition_entropy(p_t1_1, p_t1_2) == pytest.approx(e_1)

    p_mt_1 = ['cat']*10 + ['dog']*4 + ['cow']*1
    p_mt_2 = ['cat']*1 + ['dog']*4 + ['cow']*9
    assert bdt.weighted_partition_entropy(p_mt_1, p_mt_2) == pytest.approx(e_m)


def test_best_partition():
    x = np.transpose([[1, 2, 3, 7, 8, 9, 10],
                      [3, 4, 5, 1, 2, 3, 4],
                      [3, 1, 5, 1, 2, 3, 4]])
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    part, i_gain = bdt.best_partition(x[:, 0], y)
    assert 3 < part < 7
    assert i_gain == pytest.approx(bdt.entropy(y))

    part, i_gain = bdt.best_partition(x[:, 1], y)
    assert 2 < part < 3
    assert i_gain == pytest.approx(0.2916919971380596)

    part, i_gain = bdt.best_partition(x[:, 2], y)
    assert 4 < part < 5
    assert i_gain == pytest.approx(0.19811742113040332)


def plot_blobs():
    x, y = make_blobs(n_samples=1500, n_features=2, random_state=123,
                      cluster_std=2, centers=2)

    x_0, x_1 = x[:, 0], x[:, 1]

    part_0, i_gain_0 = bdt.best_partition(x_0, y)
    print(part_0, i_gain_0)
    part_1, i_gain_1 = bdt.best_partition(x_1, y)
    print(part_1, i_gain_1)
    plt.vlines(part_0, -10, 10)
    plt.hlines(part_1, -10, 10)
    plt.scatter(x_0, x_1, c=y)
    plt.show()


# plot_blobs()
ltr()
