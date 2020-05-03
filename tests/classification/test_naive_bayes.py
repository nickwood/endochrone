# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone.utils.misc import lazy_test_runner as ltr
import endochrone.classification.naive_bayes as nb

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_2d_fit_and_predict():
    X = np.transpose([[5.92, 5.58, 5.92, 6, 5, 5.5, 5.42, 5.75],
                      [180, 190, 170, 165, 100, 150, 130, 150],
                      [12, 11, 12, 10, 6, 8, 7, 9]])
    Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    classifier = nb.NaiveBayes()
    classifier.fit(X, Y)

    classes = np.array([0, 1])
    means = np.array([[5.855, 176.25, 11.25], [5.4175, 132.5, 7.5]])
    variances = np.array([[0.03503333, 122.91666, 0.9166667],
                          [0.097225, 558.333333, 1.666667]])

    assert classifier.n_classes_ == 2
    assert classifier.n_samples_ == 8
    assert classifier.n_feat_ == 3
    assert np.all(classifier.classes_ == classes)
    assert np.all(classifier.means_ == pytest.approx(means))
    assert np.all(classifier.variances_ == pytest.approx(variances))
    assert np.all(classifier.priors_ == 0.5)

    assert classifier.p_f_given_cl(0, 0, 6) == pytest.approx(1.57888318)
    assert classifier.p_f_given_cl(0, 1, 6) == pytest.approx(0.22345872)
    assert classifier.p_f_given_cl(1, 0, 130) == pytest.approx(5.986743e-06)
    assert classifier.p_f_given_cl(1, 1, 130) == pytest.approx(0.01678929)
    assert classifier.p_f_given_cl(2, 0, 8) == pytest.approx(0.001311221)
    assert classifier.p_f_given_cl(2, 1, 8) == pytest.approx(0.2866907)

    samp = np.array([6, 130, 8])
    assert classifier.posterior_numerator(0, samp) ==\
        pytest.approx(6.197071843878083e-09)
    assert classifier.posterior_numerator(1, samp) ==\
        pytest.approx(0.0005377909183630023)

    assert classifier.predict_single(samp) == 1
    assert classifier.predict(samp) == 1

    samps = np.array([[6, 130, 8], [5.9, 180, 11]])
    assert np.all(classifier.predict(samps) == np.array([1, 0]))


def test_1d_fit_and_predict():
    X = np.array([5.92, 5.58, 5.92, 6, 5, 5.5, 5.42, 5.75])
    Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    classifier = nb.NaiveBayes()
    classifier.fit(X, Y)

    classes = np.array([0, 1])
    means = np.array([5.855, 5.4175])[:, np.newaxis]
    variances = np.array([0.03503333333333331, 0.097225])[:, np.newaxis]

    assert classifier.n_classes_ == 2
    assert classifier.n_samples_ == 8
    assert classifier.n_feat_ == 1
    assert np.all(classifier.classes_ == classes)
    assert np.all(classifier.means_ == pytest.approx(means))
    assert np.all(classifier.variances_ == pytest.approx(variances))
    assert np.all(classifier.priors_ == 0.5)

    samp = np.array([5.4])

    assert classifier.predict_single(samp) == 1
    assert classifier.predict(samp) == 1

    samps = np.array([5.4, 6])
    assert np.all(classifier.predict(samps) == np.array([1, 0]))


def test_known_priors():
    X = np.transpose([[5.92, 5.58, 5.92, 6, 5, 5.5, 5.42, 5.75],
                      [180, 190, 170, 165, 100, 150, 130, 150],
                      [12, 11, 12, 10, 6, 8, 7, 9]])
    Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    classifier = nb.NaiveBayes()
    classifier.fit(X, Y, pop_priors=(0.3, 0.7))

    assert np.all(classifier.priors_ == np.array([0.3, 0.7]))

    samp = np.array([6, 130, 8])
    assert classifier.posterior_numerator(0, samp) ==\
        pytest.approx(3.7182431063e-09)
    assert classifier.posterior_numerator(1, samp) ==\
        pytest.approx(0.0007529072857)


ltr()