# -*- coding: utf-8 -*-
import numpy as np

__author__ = "nickwood"
__copyright__ = "nickwood"
__license__ = "mit"


def confusion_matrix(ytest, ypred):
    labels = np.unique([ypred, ytest])
    combinations = np.concatenate([ypred, ytest], axis=1)
    comb, counts = np.unique(combinations, axis=0, return_counts=True)
    return (labels, comb, counts)


def print_confusion_matrix(ytest, ypred):
    labels, comb, counts = confusion_matrix(ytest, ypred)
    N = len(labels)+1
    matrix = [[0 for _ in range(N)] for _ in range(N)]
    matrix[0][0] = r'act\exp'
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
