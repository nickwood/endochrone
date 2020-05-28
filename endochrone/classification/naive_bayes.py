# -*- coding: utf-8 -*-
import numpy as np

from endochrone import Base

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class NaiveBayes(Base):
    def __init__(self, *, laplace_coefficient=0, method='gaussian'):
        self.lambda_ = laplace_coefficient
        if method not in ['gaussian', 'bernoulli']:
            raise ValueError('Unknown method: %s' % method)
        self.method = method
        super().__init__()

    def fit(self, *, features, targets, pop_priors=None):
        self.validate_fit(features=features, targets=targets)

        self.classes_ = dict(enumerate(np.unique(targets)))
        self.n_classes_ = len(self.classes_)
        self.n_samples_ = len(targets)

        if pop_priors is not None and sum(pop_priors.values()) != 1:
            raise ValueError("Sum of population priors doesn't sum to 1")

        self.fit_gaussian(features=features, targets=targets,
                          pop_priors=pop_priors)

    def fit_gaussian(self, *, features, targets, pop_priors):
        means = []
        variances = []
        priors = []

        for cl_ind in self.classes_.keys():
            subset = features[targets == self.classes_[cl_ind]]
            means.append(np.mean(subset, axis=0))
            variances.append(np.var(subset, ddof=1, axis=0))
            if pop_priors is None:
                pri_numer = len(subset) + self.lambda_
                pri_denom = self.n_samples_ + self.n_classes_ * self.lambda_
                priors.append(pri_numer / pri_denom)
            else:
                priors.append(pop_priors[self.classes_[cl_ind]])

        dimensionality = (self.n_classes_, self.n_features_)
        self.means_ = np.reshape(means, dimensionality)
        self.variances_ = np.reshape(variances, dimensionality)
        self.priors_ = np.array(priors)

    def predict(self, features):
        self.validate_predict(features=features)
        if features.shape == (self.n_features_, 1):
            return self.predict_single_(features)
        else:
            return np.array([self.predict_single_(obs) for obs in features])

    def predict_single_(self, obs):
        numers = [self.posterior_numerator_(cl, obs) for cl in self.classes_]
        return self.classes_[np.argmax(numers)]

    def posterior_numerator_(self, cl, obs):
        if self.n_features_ == 1:
            pds = self.p_f_given_cl_(0, cl, obs)
        else:
            pds = [self.p_f_given_cl_(f, cl, obs[f])
                   for f in range(self.n_features_)]
        return self.priors_[cl] * np.product(pds) + self.lambda_

    def p_f_given_cl_(self, f, cl, obs):
        var = self.variances_[cl, f]
        mean = self.means_[cl, f]
        coef = 1/np.sqrt(2 * np.pi * var)
        exponent = -(obs - mean)**2/(2 * var)
        return coef * np.exp(exponent)
