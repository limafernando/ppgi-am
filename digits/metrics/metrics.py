import numpy as np


def compute_acc(Y, Y_pred):
    acc = 0
    sample_size = len(Y)
    for idx in range(sample_size):
        if Y_pred[idx] == Y[idx]:
            acc += 1
    return acc/sample_size
