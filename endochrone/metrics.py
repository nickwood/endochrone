# -*- coding: utf-8 -*-
import numpy as np

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def confusion_matrix(ytest, ypred):
    labels = np.unique([ypred, ytest])
    combinations = np.concatenate([ypred, ytest], axis=1)
    combs, counts = np.unique(combinations, axis=0, return_counts=True)
    return (labels, combs, counts)


def print_confusion_matrix(ytest, ypred):
    labels, comb, counts = confusion_matrix(ytest, ypred)
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


def true_positive(ytest, ypred):
    return np.sum(ytest & ypred) / ytest.shape[0]


def false_positive(ytest, ypred):
    return np.sum(~ytest & ypred) / ytest.shape[0]


def true_negative(ytest, ypred):
    return np.sum(~ytest & ~ypred) / ytest.shape[0]


def false_negative(ytest, ypred):
    return np.sum(ytest & ~ypred) / ytest.shape[0]


def precision(ytest, ypred):
    return true_positive(ytest, ypred) /\
           (true_positive(ytest, ypred) + false_positive(ytest, ypred))


def recall(ytest, ypred):
    return true_positive(ytest, ypred) /\
           (true_positive(ytest, ypred) + false_negative(ytest, ypred))


def f1_score(ytest, ypred):
    return 2 * (precision(ytest, ypred) * recall(ytest, ypred)) /\
           (precision(ytest, ypred) + recall(ytest, ypred))


def multiclass_precision(ytest, ypred):
    "return macro precision"
    labels, combs, counts = confusion_matrix(ytest, ypred)
    N = len(labels)
    preds = [np.sum([counts[i] for i, (x, y) in enumerate(combs) if x == j])
             for j in range(N)]
    tps = [counts[i] for i, (x, y) in enumerate(combs) if x == y]
    precisions = np.array(tps) / np.array(preds)
    return np.mean(precisions)


def multiclass_recall(ytest, ypred):
    "return macro recall"
    labels, combs, counts = confusion_matrix(ytest, ypred)
    N = len(labels)
    acts = [np.sum([counts[i] for i, (x, y) in enumerate(combs) if y == j])
            for j in range(N)]
    tps = [counts[i] for i, (x, y) in enumerate(combs) if x == y]
    recalls = np.array(tps) / np.array(acts)
    return np.mean(recalls)


def multiclass_f1_score(ytest, ypred):
    prec = multiclass_precision(ytest, ypred)
    recall = multiclass_recall(ytest, ypred)
    return 2 * (prec * recall) / (prec + recall)
