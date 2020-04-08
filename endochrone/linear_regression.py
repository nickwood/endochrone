# -*- coding: utf-8 -*-
import numpy as np

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class LinearRegression:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def fit(self):
        n_samples = self.X_train.shape[0]
        X = np.c_[np.ones(n_samples), self.X_train]
        X_T = X.transpose()
        XTX_inv = np.linalg.inv(X_T@X)
        beta_vector = (XTX_inv@X_T@self.y_train)
        self.intercept_ = int(beta_vector[0])
        self.coef_ = beta_vector[1:]

    def predict(self, X_test):
        y = [np.sum(point*self.coef_) + self.intercept_ for point in X_test]
        return np.array(y)[:, np.newaxis]
