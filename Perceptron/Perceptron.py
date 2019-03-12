import numpy as np


class Perceptron:

    def __init__(self, learning_rate, features_num):
        self.learning_rate = learning_rate
        self.features_num = features_num
        self.weights = np.ones(features_num)
        self.threshold = 1

    def __net_input(self, vector):
        return np.dot(vector, self.weights) - self.threshold

    @staticmethod
    def __bipolar_step_function(n):
        return np.where(n >= 0, 1, -1)

    def learn(self, sample, target):
        error = target - self.predict(sample)
        weights_delta = self.learning_rate * error * sample
        threshold_delta = self.learning_rate * error
        self.weights = self.weights + weights_delta
        self.threshold += threshold_delta

    def predict(self, sample):
        return self.__bipolar_step_function(self.__net_input(sample))

    def __str__(self):
        return f'Perceptron: \n {self.features_num}; {self.learning_rate} \n {self.weights} \n {self.threshold}'

