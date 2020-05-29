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

        if pop_priors is None:
            self.estimate_priors_(features=features, targets=targets)
        elif sum(pop_priors.values()) != 1:
            raise ValueError("Population priors don't sum to 1")
        else:
            self.priors_ = np.array([pop_priors[self.classes_[cl_ind]]
                                     for cl_ind in self.classes_.keys()])

        if self.method == 'gaussian':
            self.fit_gaussian(features=features, targets=targets)
        else:
            self.fit_bernoulli(features=features, targets=targets)

    def estimate_priors_(self, *, features, targets):
        priors = []
        for cl_ind in self.classes_.keys():
            subset = features[targets == self.classes_[cl_ind]]
            numer = len(subset) + self.lambda_
            denom = self.n_samples_ + self.n_classes_ * self.lambda_
            priors.append(numer / denom)
        self.priors_ = np.array(priors)

    def fit_gaussian(self, *, features, targets):
        means = []
        variances = []

        for cl_ind in self.classes_.keys():
            subset = features[targets == self.classes_[cl_ind]]
            means.append(np.mean(subset, axis=0))
            variances.append(np.var(subset, ddof=1, axis=0))

        dimensionality = (self.n_classes_, self.n_features_)
        self.means_ = np.reshape(means, dimensionality)
        self.variances_ = np.reshape(variances, dimensionality)

    def fit_bernoulli(self, *, features, targets):
        '''populate the cond_probs_ dictionary which has structure as follows:
        cond_probs_[feature_index][feature_value][target_index] == P(feat|targ)
        NB. feature values are cast to strings
        '''
        self.cond_probs_ = {}
        lam = self.lambda_

        t_masks = np.array([targets == self.classes_[cl_ind]
                            for cl_ind in self.classes_.keys()])
        label_totals = np.count_nonzero(t_masks, axis=1)

        for i, feat in enumerate(features.T):
            self.cond_probs_[i] = {}
            feat = feat.astype(str)
            feature_values = np.unique(feat)
            n_f_vals = len(feature_values)

            for f_val in feature_values:
                probs = {}
                f_mask = feat == f_val

                for cl_ind in self.classes_.keys():
                    numer = np.count_nonzero(f_mask & t_masks[cl_ind]) + lam
                    denom = label_totals[cl_ind] + (lam * n_f_vals)
                    probs[cl_ind] = numer / denom
                self.cond_probs_[i][f_val] = probs

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
            var = self.variances_[cl, f]
            mean = self.means_[cl, f]
            coef = 1/np.sqrt(2 * np.pi * var)
            exponent = -(obs - mean)**2/(2 * var)
            return coef * np.exp(exponent)
        else:
            return self.cond_probs_[f][obs][cl]
