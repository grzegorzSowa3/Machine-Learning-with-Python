import numpy as np


def sigmoid(x, derivative=False):
    result = 1 / (1 + np.exp(-x))
    if not derivative:
        return result
    else:
        return result * (1 - result)


def tanh(x, derivative=False):
    if not derivative:
        return np.tanh(x)
    else:
        return 1 - np.tanh(x) ** 2


def relu(x, derivative=False):
    if not derivative:
        return np.max(0, x)
    else:
        return int(x >= 0)


def leaky_relu(x, derivative=False):
    if not derivative:
        return np.max(0.01 * x, x)
    else:
        return np.where(x >= 0, 1, 0.01)
