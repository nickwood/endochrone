# -*- coding: utf-8 -*-
import numpy as np
from functools import cached_property

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


# TODO should really use 1d version instead of column vector
class Metrics:
    def __init__(self, y_true, y_predicted):
        if y_true.ndim == 1:
            self.y_true = y_true[:, np.newaxis]
        else:
            self.y_true = y_true
        if y_predicted.ndim == 1:
            self.y_pred = y_predicted[:, np.newaxis]
        else:
            self.y_pred = y_predicted
        self.samples = np.concatenate([self.y_true, self.y_pred], axis=1)
        self.labels = np.unique([y_predicted, y_true])
        self.n_labels = len(self.labels)

    @cached_property
    def combs(self):
        return self.confusion_matrix[0]

    @cached_property
    def counts(self):
        return self.confusion_matrix[1]

    @cached_property
    def confusion_matrix(self):
        combs, counts = np.unique(self.samples, axis=0, return_counts=True)
        return (combs, counts, self.labels)

    def print_confusion_matrix(self):
        N = self.n_labels + 1
        matrix = [[0 for _ in range(N)] for _ in range(N)]
        matrix[0][0] = r'pred\act'

        for i in range(0, N-1):
            matrix[0][i+1] = self.labels[i]
            matrix[i+1][0] = self.labels[i]

        for idx, (i, j) in enumerate(self.combs):
            matrix[j+1][i+1] = self.counts[idx]

        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))


class BinaryMetrics(Metrics):
    @cached_property
    def true_positive(self):
        return np.mean(self.y_true & self.y_pred)

    @cached_property
    def false_positive(self):
        return np.mean(~self.y_true & self.y_pred)

    @cached_property
    def true_negative(self):
        return np.mean(~self.y_true & ~self.y_pred)

    @cached_property
    def false_negative(self):
        return np.mean(self.y_true & ~self.y_pred)

    @cached_property
    def precision(self):
        return self.true_positive / (self.true_positive + self.false_positive)

    @cached_property
    def recall(self):
        return self.true_positive / (self.true_positive + self.false_negative)

    @cached_property
    def f1_score(self):
        return 2 * (self.precision * self.recall) /\
                   (self.precision + self.recall)


class MulticlassMetrics(Metrics):
    @cached_property
    def n_predicted(self):
        "return an array containing the number of predictions for each label"
        return np.array([np.sum([self.counts[i]
                                 for i, (x, y) in enumerate(self.combs)
                                 if y == label])
                         for label in self.labels])

    @cached_property
    def n_true(self):
        "return an array containing count for each label in 'true' column"
        return np.array([np.sum([self.counts[i]
                                 for i, (x, y) in enumerate(self.combs)
                                 if x == label])
                         for label in self.labels])

    @cached_property
    def n_true_positive(self):
        "return the number of true positives for each label"
        return np.array([np.sum([self.counts[i]
                                 for i, (x, y) in enumerate(self.combs)
                                 if x == y == label])
                         for label in self.labels])

    @cached_property
    def macro_precision(self):
        filt = self.n_predicted != 0
        return np.mean(self.n_true_positive[filt] / self.n_predicted[filt])

    @cached_property
    def macro_recall(self):
        filt = self.n_true != 0
        return np.mean(self.n_true_positive[filt] / self.n_true[filt])

    @cached_property
    def macro_f1_score(self):
        if self.macro_precision * self.macro_recall == 0:
            return 0
        return 2 * (self.macro_precision * self.macro_recall) /\
                   (self.macro_precision + self.macro_recall)

    @cached_property
    def micro_precision(self):
        filt = self.n_predicted != 0
        return (np.sum(self.n_true_positive[filt]) /
                np.sum(self.n_predicted[filt]))

    @cached_property
    def micro_recall(self):
        return self.micro_precision

    @cached_property
    def micro_f1_score(self):
        return self.micro_precision
