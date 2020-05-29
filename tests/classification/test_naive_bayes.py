# -*- coding: utf-8 -*-
import numpy as np
import pytest

from endochrone.utils import lazy_test_runner as ltr
from endochrone.classification import NaiveBayes

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def test_2d_gaussian():
    X = np.transpose([[5.92, 5.58, 5.92, 6, 5, 5.5, 5.42, 5.75],
                      [180, 190, 170, 165, 100, 150, 130, 150],
                      [12, 11, 12, 10, 6, 8, 7, 9]])
    Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    classifier = NaiveBayes()
    classifier.fit(features=X, targets=Y)

    classes = np.array([0, 1])
    means = {0: [5.855, 176.25, 11.25], 1: [5.4175, 132.5, 7.5]}
    varians = {0: [0.03503333, 122.91666, 0.9166667],
               1: [0.097225, 558.333333, 1.666667]}

    assert classifier.n_classes_ == 2
    assert classifier.n_samples_ == 8
    assert classifier.n_features_ == 3
    assert np.all(classifier.classes_ == classes)
    for cl in classes:
        assert classifier.priors_[cl] == 0.5
        assert np.all(classifier.means_[cl] == pytest.approx(means[cl]))
        assert np.all(classifier.variances_[cl] == pytest.approx(varians[cl]))

    assert classifier.p_f_given_cl_(0, 0, 6) == pytest.approx(1.57888318)
    assert classifier.p_f_given_cl_(0, 1, 6) == pytest.approx(0.22345872)
    assert classifier.p_f_given_cl_(1, 0, 130) == pytest.approx(5.986743e-06)
    assert classifier.p_f_given_cl_(1, 1, 130) == pytest.approx(0.01678929)
    assert classifier.p_f_given_cl_(2, 0, 8) == pytest.approx(0.001311221)
    assert classifier.p_f_given_cl_(2, 1, 8) == pytest.approx(0.2866907)

    samp = np.array([6, 130, 8])
    assert classifier.posterior_numerator_(0, samp) ==\
        pytest.approx(6.197071843878083e-09)
    assert classifier.posterior_numerator_(1, samp) ==\
        pytest.approx(0.0005377909183630023)

    assert classifier.predict_single_(samp) == 1
    assert classifier.predict(samp.reshape(1, 3)) == 1

    samps = np.array([[6, 130, 8], [5.9, 180, 11]])
    assert np.all(classifier.predict(samps) == np.array([1, 0]))


def test_1d_gaussian():
    X = np.array([5.92, 5.58, 5.92, 6, 5, 5.5, 5.42, 5.75])[:, np.newaxis]
    Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    classifier = NaiveBayes()
    classifier.fit(features=X, targets=Y)

    classes = np.array([0, 1])
    means = {0: 5.855, 1: 5.4175}
    varians = {0: 0.03503333333333331, 1: 0.097225}

    assert classifier.n_classes_ == 2
    assert classifier.n_samples_ == 8
    assert classifier.n_features_ == 1
    assert np.all(classifier.classes_ == classes)

    for cl in classes:
        assert classifier.priors_[cl] == 0.5
        assert np.all(classifier.means_[cl] == pytest.approx(means[cl]))
        assert np.all(classifier.variances_[cl] == pytest.approx(varians[cl]))

    samp = np.array([5.4])

    assert classifier.predict_single_(samp) == 1
    assert classifier.predict(samp.reshape(1, 1)) == 1

    samps = np.array([5.4, 6]).reshape(2, 1)
    assert np.all(classifier.predict(samps) == np.array([1, 0]))


def test_three_classes():
    X = np.array([5.92, 5.88, 5.93, 5.2, 5, 5.1, 4.4, 4.6, 4.5])[:, np.newaxis]
    Y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    classifier = NaiveBayes()
    priors = {0: 0.45, 1: 0.45, 2: 0.1}
    classifier.fit(features=X, targets=Y, pop_priors=priors)

    classes = np.array([0, 1, 2])
    means = {0: 5.91, 1: 5.1, 2: 4.5}
    varians = {0: 0.0007, 1: 0.01, 2: 0.01}

    assert classifier.n_classes_ == 3
    assert classifier.n_samples_ == 9
    assert classifier.n_features_ == 1
    assert np.all(classifier.classes_ == classes)
    for cl in classes:
        assert classifier.priors_[cl] == priors[cl]
        assert np.all(classifier.means_[cl] == pytest.approx(means[cl]))
        assert np.all(classifier.variances_[cl] == pytest.approx(varians[cl]))

    samps = np.array([5.9, 5.0, 4.8, 4.4])[:, np.newaxis]
    assert np.all(classifier.predict(samps) == np.array([0, 1, 1, 2]))


def test_lambda_smoothing():
    X = np.array([5.92, 5.93, 6.2, 5, 5.1, 4.4, 4.6, 2.5, 2.4])[:, np.newaxis]
    Y = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2])

    classifier = NaiveBayes(laplace_coefficient=1)
    classifier.fit(features=X, targets=Y)

    classes = np.array([0, 1, 2])
    means = {0: 6.016666667, 1: 4.775, 2: 2.45}
    varians = {0: 0.02523333, 1: 0.10916667, 2: 0.005}
    priors = {0: 4/12, 1: 5/12, 2: 3/12}

    assert classifier.n_classes_ == 3
    assert classifier.n_samples_ == 9
    assert classifier.n_features_ == 1
    for cl in classes:
        assert classifier.priors_[cl] == priors[cl]
        assert np.all(classifier.means_[cl] == pytest.approx(means[cl]))
        assert np.all(classifier.variances_[cl] == pytest.approx(varians[cl]))

    samps = np.array([5.9, 5.0, 4.8, 2.5])[:, np.newaxis]
    assert np.all(classifier.predict(samps) == np.array([0, 1, 1, 2]))


def test_known_priors():
    X = np.transpose([[5.92, 5.58, 5.92, 6, 5, 5.5, 5.42, 5.75],
                      [180, 190, 170, 165, 100, 150, 130, 150],
                      [12, 11, 12, 10, 6, 8, 7, 9]])
    Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    classifier = NaiveBayes()
    classifier.fit(features=X, targets=Y, pop_priors={0: 0.3, 1: 0.7})

    assert classifier.priors_[0] == 0.3
    assert classifier.priors_[1] == 0.7

    samp = np.array([6, 130, 8])
    assert classifier.posterior_numerator_(0, samp) ==\
        pytest.approx(3.7182431063e-09)
    assert classifier.posterior_numerator_(1, samp) ==\
        pytest.approx(0.0007529072857)


def test_non_zero_indexed_classnames():
    X = np.transpose([[5.92, 5.58, 5.92, 6, 5, 5.5, 5.42, 5.75],
                      [180, 190, 170, 165, 100, 150, 130, 150],
                      [12, 11, 12, 10, 6, 8, 7, 9]])
    Y = np.array([2, 2, 2, 2, 1, 1, 1, 1])

    classifier = NaiveBayes()
    priors = {1: 0.49, 2: 0.51}
    classifier.fit(features=X, targets=Y, pop_priors=priors)

    for cl in classifier.classes_:
        assert classifier.priors_[cl] == priors[cl]

    samp = np.array([6, 130, 8])
    assert classifier.predict_single_(samp) == 1
    assert classifier.predict(samp.reshape(1, 3)) == 1

    samps = np.array([[6, 130, 8], [5.9, 180, 11]])
    assert np.all(classifier.predict(samps) == np.array([1, 2]))


def test_text_classes():
    X = np.transpose([[5.92, 5.58, 5.92, 6, 5, 5.5, 5.42, 5.75],
                      [180, 190, 170, 165, 100, 150, 130, 150],
                      [12, 11, 12, 10, 6, 8, 7, 9]])
    Y = np.array(['male']*4 + ['female']*4)

    classifier = NaiveBayes()
    priors = {'male': 0.49, 'female': 0.51}
    classifier.fit(features=X, targets=Y, pop_priors=priors)

    samp = np.array([6, 130, 8])

    assert classifier.predict_single_(samp) == 'female'
    assert classifier.predict(samp.reshape(1, 3)) == 'female'

    for cl in classifier.classes_:
        assert classifier.priors_[cl] == priors[cl]

    samps = np.array([[6, 130, 8], [5.9, 180, 11]])
    assert np.all(classifier.predict(samps) == np.array(['female', 'male']))


def test_bernoilli():
    # from https://towardsdatascience.com/introduction-to-na%C3%AFve-bayes-classifier-fa59e3e24aaf  # noqa: E501
    X = np.transpose([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                      ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']])  # noqa: E501
    Y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    classifier = NaiveBayes(method='bernoulli')
    classifier.fit(features=X, targets=Y)

    assert classifier.priors_[0] == pytest.approx(0.4)
    assert classifier.priors_[1] == pytest.approx(0.6)
    assert classifier.cond_probs_[0][0]['0'] == pytest.approx(0.5)
    assert classifier.cond_probs_[1][0]['0'] == pytest.approx(2/9)
    assert classifier.cond_probs_[0][0]['1'] == pytest.approx(1/3)
    assert classifier.cond_probs_[1][0]['1'] == pytest.approx(1/3)
    assert classifier.cond_probs_[0][0]['2'] == pytest.approx(1/6)
    assert classifier.cond_probs_[1][0]['2'] == pytest.approx(4/9)
    assert classifier.cond_probs_[0][1]['S'] == pytest.approx(0.5)
    assert classifier.cond_probs_[1][1]['S'] == pytest.approx(1/9)
    assert classifier.cond_probs_[0][1]['M'] == pytest.approx(1/3)
    assert classifier.cond_probs_[1][1]['M'] == pytest.approx(4/9)
    assert classifier.cond_probs_[0][1]['L'] == pytest.approx(1/6)
    assert classifier.cond_probs_[1][1]['L'] == pytest.approx(4/9)

    samp = np.array([1, 'S'])
    assert classifier.posterior_numerator_(0, samp) == pytest.approx(1/15)
    assert classifier.posterior_numerator_(1, samp) == pytest.approx(1/45)
    assert classifier.predict_single_(samp) == 0
    assert classifier.predict(samp.reshape(1, 2)) == [0]


def test_bernoilli_with_laplace():
    X = np.transpose([['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C'],  # noqa: E501
                      ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']])  # noqa: E501
    Y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    classifier = NaiveBayes(method='bernoulli', laplace_coefficient=1)
    classifier.fit(features=X, targets=Y)

    assert classifier.priors_[0] == pytest.approx(7/17)
    assert classifier.priors_[1] == pytest.approx(10/17)

    assert classifier.cond_probs_[0][0]['A'] == pytest.approx(4/9)
    assert classifier.cond_probs_[1][0]['A'] == pytest.approx(1/4)
    assert classifier.cond_probs_[0][0]['B'] == pytest.approx(1/3)
    assert classifier.cond_probs_[1][0]['B'] == pytest.approx(1/3)
    assert classifier.cond_probs_[0][0]['C'] == pytest.approx(2/9)
    assert classifier.cond_probs_[1][0]['C'] == pytest.approx(5/12)
    assert classifier.cond_probs_[0][1]['S'] == pytest.approx(4/9)
    assert classifier.cond_probs_[1][1]['S'] == pytest.approx(1/6)
    assert classifier.cond_probs_[0][1]['M'] == pytest.approx(1/3)
    assert classifier.cond_probs_[1][1]['M'] == pytest.approx(5/12)
    assert classifier.cond_probs_[0][1]['L'] == pytest.approx(2/9)
    assert classifier.cond_probs_[1][1]['L'] == pytest.approx(5/12)


def test_invalid_priors():
    X = np.array([5.92, 5.75]).reshape(2, 1)
    Y = np.array([0, 1])

    classifier = NaiveBayes()
    priors = {1: 0.49, 0: 0.50}

    with pytest.raises(ValueError):
        classifier.fit(features=X, targets=Y, pop_priors=priors)


def test_invalid_method():
    with pytest.raises(ValueError):
        _ = NaiveBayes(method='wimble')


ltr()
