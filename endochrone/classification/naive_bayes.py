# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict

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

        self.classes_ = np.unique(targets)
        self.n_classes_ = len(self.classes_)
        self.n_samples_ = len(targets)

        if pop_priors is None:
            self.estimate_priors_(features=features, targets=targets)
        elif sum(pop_priors.values()) != 1:
            raise ValueError("Population priors don't sum to 1")
        else:
            self.priors_ = pop_priors

        if self.method == 'gaussian':
            self.fit_gaussian(features=features, targets=targets)
        else:
            self.fit_bernoulli(features=features, targets=targets)

    def estimate_priors_(self, *, features, targets):
        self.priors_ = {}
        for cl in self.classes_:
            subset = features[targets == cl]
            numer = len(subset) + self.lambda_
            denom = self.n_samples_ + self.n_classes_ * self.lambda_
            self.priors_[cl] = (numer / denom)

    def fit_gaussian(self, *, features, targets):
        self.means_ = {}
        self.variances_ = {}

        for cl in self.classes_:
            subset = features[targets == cl]
            self.means_[cl] = np.mean(subset, axis=0)
            self.variances_[cl] = np.var(subset, ddof=1, axis=0)

    def fit_bernoulli(self, *, features, targets):
        '''populate the cond_probs_ dictionary which has structure as follows:
        cond_probs_[target][feature_index][feature_value] == P(feat|targ)
        NB. feature values are cast to strings
        '''
        self.cond_probs_ = {}
        lam = self.lambda_
        cl_masks = {}
        cl_totals = {}

        for cl in self.classes_:
            self.cond_probs_[cl] = defaultdict(dict)
            cl_mask = targets == cl
            cl_masks[cl] = cl_mask
            cl_totals[cl] = np.count_nonzero(cl_mask)

        for i, feat in enumerate(features.T):
            feat = feat.astype(str)
            feature_values = np.unique(feat)
            n_f_vals = len(feature_values)

            for f_val in feature_values:
                f_mask = feat == f_val

                for cl in self.classes_:
                    numer = np.count_nonzero(f_mask & cl_masks[cl]) + lam
                    denom = cl_totals[cl] + (lam * n_f_vals)
                    self.cond_probs_[cl][i][f_val] = numer / denom

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
        if self.method == 'gaussian':
            var = self.variances_[cl][f]
            mean = self.means_[cl][f]
            coef = 1/np.sqrt(2 * np.pi * var)
            exponent = -(obs - mean)**2/(2 * var)
            return coef * np.exp(exponent)
        else:
            return self.cond_probs_[cl][f][obs]
