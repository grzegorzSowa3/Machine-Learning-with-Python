import numpy as np


class Perceptron:

    # Simple neural network consisting of single McCulloch-Pitts neuron

    def __init__(self, features_num, learning_rate=0.5, iters_num=20):
        self.features_num = features_num
        self.weights = np.ones(features_num, dtype=float)
        self.thresholds = 1.0
        self.learning_rate = learning_rate
        self.iters_num = iters_num

    def __activation(self, inputs):
        return self.__bipolar_step_function(inputs)

    def __net_input(self, inputs):
        return np.dot(self.weights, inputs) + self.thresholds

    @staticmethod
    def __bipolar_step_function(x):
        return np.where(x >= 0, 1, -1)

    def learn(self, samples, targets):
        errors_nums = []
        for i in range(self.iters_num):
            errors_num = 0
            for sample, target in zip(samples, targets):
                error = target - self.predict(sample)
                weights_delta = self.learning_rate * sample * error
                threshold_delta = self.learning_rate * error
                self.weights = self.weights + weights_delta
                self.thresholds = self.thresholds + threshold_delta
                errors_num += int(error != 0.0)
            errors_nums.append(errors_num)
        return errors_nums

    def predict(self, sample):
        return self.__activation(self.__net_input(sample))

    def __str__(self):
        return f'Perceptron: \n {self.features_num}; {self.learning_rate} \n {self.weights} \n {self.thresholds}'
