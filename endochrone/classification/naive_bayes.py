# -*- coding: utf-8 -*-
import numpy as np

from endochrone import Base

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class NaiveBayes(Base):
    def __init__(self):
        super().__init__()

    def fit(self, x, y, pop_priors=None):
        self.validate_fit(features=x, targets=y)

        self.classes_ = dict(enumerate(np.unique(y)))
        self.n_classes_ = len(self.classes_)
        self.n_samples_ = len(y)

        if pop_priors is not None and sum(pop_priors.values()) != 1:
            raise ValueError("Sum of population priors don't sum to 1")

        means = []
        variances = []
        priors = []

        for cl_ind in self.classes_.keys():
            targets = y == self.classes_[cl_ind]
            subset = x[targets]
            means.append(np.mean(subset, axis=0))
            variances.append(np.var(subset, ddof=1, axis=0))
            if pop_priors is None:
                priors.append(len(subset) / self.n_samples_)
            else:
                priors.append(pop_priors[self.classes_[cl_ind]])

        dimensionality = (self.n_classes_, self.n_features_)
        self.means_ = np.reshape(means, dimensionality)
        self.variances_ = np.reshape(variances, dimensionality)
        self.priors_ = np.array(priors)

    def predict(self, X):
        self.validate_predict(features=X)
        if X.shape == (self.n_features_, 1):
            return self.predict_single(X)
        else:
            return np.array([self.predict_single(obs) for obs in X])

    def predict_single(self, obs):
        numers = [self.posterior_numerator(cl, obs) for cl in self.classes_]
        return self.classes_[np.argmax(numers)]

    def posterior_numerator(self, cl, obs):
        if self.n_features_ == 1:
            pds = self.p_f_given_cl(0, cl, obs)
        else:
            pds = [self.p_f_given_cl(f, cl, obs[f])
                   for f in range(self.n_features_)]
        return self.priors_[cl] * np.product(pds)

    def p_f_given_cl(self, f, cl, obs):
        var = self.variances_[cl, f]
        mean = self.means_[cl, f]
        coef = 1/np.sqrt(2 * np.pi * var)
        exponent = -(obs - mean)**2/(2 * var)
        return coef * np.exp(exponent)
