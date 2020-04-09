# -*- coding: utf-8 -*-
import numpy as np

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class LinearRegression:
    def __init__(self):
        pass

    def fit(self, X_train, Y_train):
        n_samples = X_train.shape[0]
        X = np.c_[np.ones(n_samples), X_train]
        X_T = X.transpose()
        beta_vector = np.linalg.inv(X_T@X)@X_T@Y_train
        self.intercept_ = beta_vector[0, 0]
        self.coef_ = beta_vector[1:, 0]

    def predict(self, X_test):
        y = np.sum(X_test*self.coef_, axis=1) + self.intercept_
        return y[:, np.newaxis]

    def score(self, X_true, Y_true):
        """Return the R^2 value for the prediction
        R^2 = 1- u/v with:
        u = ((Y_true - Y_pred) ** 2).sum()
        v = ((Y_true - Y_true.mean()) ** 2).sum()
        """
        Y_pred = self.predict(X_true)
        u = np.sum((Y_true - Y_pred)**2)
        v = np.sum((Y_true - np.mean(Y_true))**2)
        return 1 - u/v
