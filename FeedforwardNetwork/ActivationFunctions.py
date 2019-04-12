import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_p(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_p(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.max(0, x)


def relu_p(x):
    return int(x >= 0)


def leaky_relu(x):
    return np.max(0.01 * x, x)


def leaky_relu_p(x):
    return np.where(x >= 0, 1, 0.01)
