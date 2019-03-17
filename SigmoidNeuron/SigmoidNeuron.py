import numpy as np


class SigmoidNeuron:

    # Simple neural network consisting of a single sigmoid neuron

    def __init__(self, features_num, learning_rate=0.5, beta=1):
        self.features_num = features_num
        self.weights = np.ones((features_num, 1), dtype=np.float32)
        self.bias = 1.0
        self.learning_rate = learning_rate
        self.beta = beta

    def __sigmoid(self, x, derivative=False):
        result = 1 / (1 + np.exp(-self.beta * np.array(x)))
        if derivative:
            return result * (1 - result)
        else:
            return result

    def __net_input(self, inputs):
        return inputs.dot(self.weights) + self.bias

    def learn(self, samples, targets, epochs_num=5, batch_size=15):
        errors_history = np.array([])
        samples_batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
        targets_batches = [targets[i:i + batch_size] for i in range(0, len(targets), batch_size)]
        for i in range(epochs_num):
            for samples_batch, targets_batch in zip(samples_batches, targets_batches):
                errors_history = np.append(errors_history,
                                           self.batch_learn(
                                               samples_batch.reshape(len(samples_batch), self.features_num),
                                               targets_batch.reshape(len(targets_batch), 1)))
        return errors_history

    def batch_learn(self, samples, targets):
        net_inputs = self.__net_input(samples)
        errors = targets - self.__sigmoid(net_inputs)
        print(np.shape(samples.T.dot(errors * self.__sigmoid(net_inputs, derivative=True))))
        weights_delta = self.learning_rate * samples.T.dot(errors * self.__sigmoid(net_inputs, derivative=True))
        # print(np.shape(samples.T.dot(errors)))
        bias_delta = self.learning_rate * np.sum(errors * self.__sigmoid(net_inputs, derivative=True))
        self.weights = self.weights + weights_delta
        self.bias = self.bias + bias_delta
        return (errors ** 2).sum()

    def predict(self, sample):
        return self.__sigmoid(self.__net_input(sample))

    def __str__(self):
        return f'Sigmoid neuron: \n {self.features_num}; {self.learning_rate}; {self.beta} \n {self.weights} \n {self.bias}'
