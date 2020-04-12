# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

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


def PolynomialRegression(degree=3, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         skLinearRegression(**kwargs))


def poly_func(X):
    """This polynomial was chosen because all of its zeroes (-2, 4, 8) lie in
    the x-range we're looking at, so it is quite wiggly"""
    return X**3 - 10*X**2 + 8*X + 64


def hacky_polynomial():
    """As we only have a single feature, x, we perform polynomial regression by
    adding features for x^n up to the desired power. It's slightly hacky in
    that it doesn't give us a nice way to produce predictions other than
    performing the same power calculations for the x_test features.
    NB. This starts to diverge from the sklearn version for large powers (~15)
    I haven't figured out why but I suspect it is to do with f.p. precision"""
    min_x, max_x = -5, 10

    # Plot the true function with a green line
    X_show = np.linspace(min_x, max_x, 200)[:, np.newaxis]
    Y_show = poly_func(X_show)
    plt.plot(X_show, Y_show, color='green')

    # Define our training set, we add noise to y_values to make it interesting
    n_samples = 500
    X_train = np.random.uniform(min_x, max_x, size=(n_samples, 1))
    Y_exact = poly_func(X_train)
    Y_noise = 5 * np.random.standard_normal(size=n_samples)[:, np.newaxis]
    Y_train = Y_exact + Y_noise

    max_pow = 16
    powers = np.array(range(1, max_pow + 1))
    X_train_pow = np.power.outer(X_train[:, 0], powers)

    # Now train a regression model
    model = LinearRegression()
    model.fit(X_train_pow, Y_train)

    # compare with SKL
    skl_model = PolynomialRegression(max_pow)
    skl_model.fit(X_train, Y_train)

    # Plot the results
    X_show_pow = np.power.outer(X_show[:, 0], powers)
    plt.scatter(X_train, Y_train, s=1)
    plt.plot(X_show, model.predict(X_show_pow), c='red')
    plt.plot(X_show, skl_model.predict(X_show), c='blue')

    plt.show()


# linear()
# hacky_polynomial()
