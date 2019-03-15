import numpy as np


class Adaline:

    # Simple neural network consisting of a single Adaptive Linear neuron

    def __init__(self, features_num, learning_rate=0.5, epochs_num=5, batch_size=10):
        self.features_num = features_num
        self.weights = np.ones((features_num, 1), dtype=float)
        self.bias = 1.0
        self.learning_rate = learning_rate
        self.epochs_num = epochs_num
        self.batch_size = batch_size

    def __activation(self, inputs):
        return self.__net_input(inputs)

    @staticmethod
    def __bipolar_step_function(x):
        return np.where(x >= 0, 1, -1)

    def __net_input(self, inputs):
        return inputs.dot(self.weights) + self.bias

    def learn(self, samples, targets):
        errors_history = np.array([])
        samples_batches = [samples[i:i + self.batch_size] for i in range(0, len(samples), self.batch_size)]
        targets_batches = [targets[i:i + self.batch_size] for i in range(0, len(targets), self.batch_size)]
        for i in range(self.epochs_num):
            for samples_batch, targets_batch in zip(samples_batches, targets_batches):
                errors_history = np.append(errors_history,
                                           self.batch_learn(
                                               samples_batch.reshape(len(samples_batch), self.features_num),
                                               targets_batch.reshape(len(targets_batch), 1)))
        return errors_history

    def batch_learn(self, samples, targets):
        outputs = self.predict(samples)
        errors = targets - outputs
        weights_delta = self.learning_rate * samples.T.dot(errors)
        bias_delta = self.learning_rate * np.sum(errors)
        self.weights = self.weights + weights_delta
        self.bias = self.bias + bias_delta
        return (errors ** 2).sum()

    def predict(self, sample):
        return self.__bipolar_step_function(self.__activation(sample))

    def __str__(self):
        return f'Adaptive linear neuron: \n {self.features_num}; {self.learning_rate} \n {self.weights} \n {self.bias}'
