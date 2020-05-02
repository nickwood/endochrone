# -*- coding: utf-8 -*-
import numpy as np

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class LinearRegression:
    def __init__(self, calculate_residuals=False, predict_vectors=False):
        '''calculate_residuals specifies if you wish residuals to be calculated
        and stored as self.residuals_
        predict_vectors lets you specify whether the predict method
        returns a 1-d or 2-d column vector (True -> 2d vector). Default
        behaviour retains the dimensionality of the Y_train set
        '''
        self.calculate_residuals = calculate_residuals
        self.predict_vectors = predict_vectors

    # TODO: refactor to use standalone least_squares module
    def fit(self, X_train, Y_train):
        n_samples = X_train.shape[0]
        X = np.c_[np.ones(n_samples), X_train]
        X_T = X.transpose()

        if self.predict_vectors is False:
            if Y_train.ndim == 1:
                self.predict_vectors = False
            else:
                self.predict_vectors = True

        beta_vector = np.linalg.inv(X_T@X)@X_T@Y_train
        if beta_vector.ndim == 1:
            beta_vector = beta_vector[:, np.newaxis]
        self.intercept_ = beta_vector[0, 0]
        self.coef_ = beta_vector[1:, 0]
        if self.calculate_residuals:
            # TODO: Add tests for this
            self.residuals_ = Y_train - self.predict(X_train)
        else:
            self.residuals_ = None

    def predict(self, X_test):
        if X_test.ndim == 1:
            y = X_test * self.coef_ + self.intercept_
        else:
            y = np.sum(X_test * self.coef_, axis=1) + self.intercept_

        if self.predict_vectors:
            return y[:, np.newaxis]
        else:
            return y

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
