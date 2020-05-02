# -*- coding: utf-8 -*-
import numpy as np

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


# TODO support non numeric classnames, and classnames not zero indexed
class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, x, y, pop_priors=None):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_samples_ = len(y)

        if x.ndim == 1:
            self.n_feat_ = 1
        else:
            self.n_feat_ = x.shape[1]

        means = []
        varis = []
        priors = []

        for cl in self.classes_:
            targets = y == cl
            subset = x[targets]
            means.append(np.mean(subset, axis=0))
            varis.append(np.var(subset, ddof=1, axis=0))
            priors.append(len(subset) / self.n_samples_)

        self.means_ = np.reshape(means, (self.n_classes_, self.n_feat_))
        self.variances_ = np.reshape(varis, (self.n_classes_, self.n_feat_))
        if pop_priors is None:
            self.priors_ = np.array(priors)
        else:
            self.priors_ = np.array(pop_priors)

    def predict(self, X):
        if X.shape == (self.n_feat_, ) or X.shape == (self.n_feat_, 1):
            return self.predict_single(X)
        else:
            return np.array([self.predict_single(obs) for obs in X])

    def predict_single(self, obs):
        numers = [self.posterior_numerator(cl, obs) for cl in self.classes_]
        return np.argmax(numers)

    def posterior_numerator(self, cl, obs):
        if self.n_feat_ == 1:
            pds = self.p_f_given_cl(0, cl, obs)
        else:
            pds = [self.p_f_given_cl(f, cl, obs[f])
                   for f in range(self.n_feat_)]
        return self.priors_[cl] * np.product(pds)

    def p_f_given_cl(self, f, cl, obs):
        var = self.variances_[cl, f]
        mean = self.means_[cl, f]
        coef = 1/np.sqrt(2 * np.pi * var)
        exponent = -(obs - mean)**2/(2 * var)
        return coef * np.exp(exponent)
