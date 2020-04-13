# -*- coding: utf-8 -*-
import numpy as np
from functools import cached_property

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


class Metrics:
    def __init__(self, y_true, y_predicted):
        self.y_true = y_true
        self.y_pred = y_predicted

    @cached_property
    def confusion_matrix(self):
        labels = np.unique([self.y_pred, self.y_true])
        combinations = np.concatenate([self.y_pred, self.y_true], axis=1)
        combs, counts = np.unique(combinations, axis=0, return_counts=True)
        return (labels, combs, counts)

    def print_confusion_matrix(self):
        labels, comb, counts = self.confusion_matrix
        N = len(labels)+1
        matrix = [[0 for _ in range(N)] for _ in range(N)]
        matrix[0][0] = r'pred\act'
        for i in range(0, N-1):
            matrix[0][i+1] = labels[i]
            matrix[i+1][0] = labels[i]
        for idx, (i, j) in enumerate(comb):
            matrix[i+1][j+1] = counts[idx]

        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))


class BinaryMetrics(Metrics):
    @cached_property
    def true_positive(self):
        return np.sum(self.y_true & self.y_pred) / self.y_true.shape[0]

    @cached_property
    def false_positive(self):
        return np.sum(~self.y_true & self.y_pred) / self.y_true.shape[0]

    @cached_property
    def true_negative(self):
        return np.sum(~self.y_true & ~self.y_pred) / self.y_true.shape[0]

    @cached_property
    def false_negative(self):
        return np.sum(self.y_true & ~self.y_pred) / self.y_true.shape[0]

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
    def macro_precision(self):
        labels, combs, counts = self.confusion_matrix
        N = len(labels)
        preds = [np.sum([counts[i] for i, (x, y) in enumerate(combs) if x == j])
                 for j in range(N)]
        tps = [counts[i] for i, (x, y) in enumerate(combs) if x == y]
        precisions = np.array(tps) / np.array(preds)
        return np.mean(precisions)

    @cached_property
    def macro_recall(self):
        labels, combs, counts = self.confusion_matrix
        N = len(labels)
        acts = [np.sum([counts[i] for i, (x, y) in enumerate(combs) if y == j])
                for j in range(N)]
        tps = [counts[i] for i, (x, y) in enumerate(combs) if x == y]
        recalls = np.array(tps) / np.array(acts)
        return np.mean(recalls)

    @cached_property
    def macro_f1_score(self):
        return 2 * (self.macro_precision * self.macro_recall) /\
                   (self.macro_precision + self.macro_recall)
