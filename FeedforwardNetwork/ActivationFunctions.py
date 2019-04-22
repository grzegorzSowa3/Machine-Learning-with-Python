import numpy as np


def sigmoid(x, derriv=False):
    result = 1 / (1 + np.exp(-x))
    if not derriv:
        return result
    else:
        return result * (1 - result)


def tanh(x, derriv=False):
    if not derriv:
        return np.tanh(x)
    else:
        return 1 - np.tanh(x) ** 2


def relu(x, derriv=False):
    if not derriv:
        return np.max(0, x)
    else:
        return int(x >= 0)


def leaky_relu(x, derriv=False):
    if not derriv:
        return np.max(0.01 * x, x)
    else:
        return np.where(x >= 0, 1, 0.01)
