# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt

from endochrone.linear_regression import LinearRegression

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def linear():
    # First invent some data
    n_samples = 1000
    n_dim = 1
    X_train = np.random.uniform(1000, size=(n_samples, n_dim))
    coefs = np.random.uniform(1, 10, size=n_dim)
    intercept = random.uniform(-100, 100)
    Y_train_values = [sum(point*coefs)*random.uniform(0.9, 1.1) +
                      intercept*random.uniform(0.9, 1.1)
                      for point in X_train]
    Y_train = np.array(Y_train_values)[:, np.newaxis]

    # Plot our samples
    plt.figure(facecolor="w", figsize=(15, 10))
    plt.scatter(X_train, Y_train, s=1)

    # Now train a regression model
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Add line of best fit to graph
    X_fit = np.linspace(0, 1000, 20)[:, np.newaxis]
    Y_fit = model.predict(X_fit)
    plt.plot(X_fit, Y_fit, color='green')

    print(model.score(X_train, Y_train))
    plt.show()


def hacky_polynomial():
    X_show = np.linspace(0, 2.5, 2000)[:, np.newaxis]
    Y_show = (X_show**10 - 26.9*X_show**9 + 308*X_show**8 - 1972*X_show**7 +
              7744*X_show**6 - 19152*X_show**5 + 29505*X_show**4 -
              26928*X_show**3 + 12984*X_show**2 - 2462*X_show - 1.5)
    plt.plot(X_show, Y_show, color='green')

    n_samples = 2000
    n_dim = 1
    X_train = np.random.uniform(0, 2.5, size=(n_samples, n_dim))
    Y_train_values = (X_train**10 - 26.9*X_train**9 + 308*X_train**8 -
                      1972*X_train**7 + 7744*X_train**6 - 19152*X_train**5 +
                      29505*X_train**4 - 26928*X_train**3 + 12984*X_train**2 -
                      2462*X_train - 1.5)
    Y_noise = 2*np.random.standard_normal(size=n_samples)[:, np.newaxis]
    Y_train = Y_train_values + Y_noise
    plt.scatter(X_train, Y_train, s=1)

    max_pow = 10
    powers = np.array(range(1, max_pow + 1))
    X_train_pow = np.power.outer(X_train[:, 0], powers)
    print(X_train_pow.shape)

    # Now train a regression model
    model = LinearRegression()
    model.fit(X_train_pow, Y_train)
    print(model.coef_)
    print(model.intercept_)

    # Plot the result
    X_show_powers = np.power.outer(X_show[:, 0], powers)
    y_pred = model.predict(X_show_powers)
    plt.plot(X_show, y_pred, c='red')

    print(model.score(X_train_pow, Y_train))
    print(model.score(X_show_powers, Y_show))
    plt.show()


# linear()
hacky_polynomial()
