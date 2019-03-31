import numpy as np


class FeedforwardNetwork:

    # Multi-layer neural network consisting of sigmoid neurons

    def __init__(self, features_num, layer_sizes, learning_rate=0.5, beta=1):
        # weights for first layer
        tmp_weights = [np.random.rand(layer_sizes[0], features_num)]

        # weights for further layers
        for i in range(1, len(layer_sizes)):
            tmp_weights.append(np.random.rand(layer_sizes[i], layer_sizes[i - 1]))
        self.weights = np.asarray(tmp_weights)

        # biases for all layers
        tmp_biases = []
        for layer_size in layer_sizes:
            tmp_biases.append(np.random.rand(layer_size, 1))
        self.biases = np.asarray(tmp_biases)

        # hiperparameters
        self.features_num = features_num
        self.learning_rate = learning_rate
        self.beta = beta

    def __sigmoid(self, x, derivative=False):
        result = 1 / (1 + np.exp(-self.beta * x))
        if derivative:
            return result * (1 - result)
        else:
            return result

    def __net_input(self, inputs):
        pass

    def learn(self, samples, targets, epochs_num=5, batch_size=15):
        errors_history = []
        for i in range(epochs_num):
            shuffle = list(zip(samples, targets))
            np.random.shuffle(shuffle)
            samples, targets = zip(*shuffle)
            samples_batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
            targets_batches = [targets[i:i + batch_size] for i in range(0, len(targets), batch_size)]
            for samples_batch, targets_batch in zip(samples_batches, targets_batches):
                errors_history.append(self.batch_learn(samples_batch, targets_batch))
        return errors_history

    def batch_learn(self, samples, targets):
        weight_deltas_sum = np.zeros(np.shape(self.weights))
        bias_deltas_sum = np.zeros(np.shape(self.biases))
        error_sums = []
        for sample, target in zip(samples, targets):
            net_inputs = self.calculate_net_inputs(sample)
            errors = self.__sigmoid(net_inputs[-1]) - target
            bias_delta = [self.learning_rate * 2 * errors * self.__sigmoid(net_inputs[-1], derivative=True)]
            weight_delta = [bias_delta[0].dot(self.__sigmoid(net_inputs[-2].T))]
            for i in range(1, len(self.weights)):
                bias_delta.insert(0, self.learning_rate
                                  * self.__sigmoid(net_inputs[-i - 1], derivative=True)
                                  * self.weights[-i].T.dot(bias_delta[-i]))
                weight_delta.insert(0, bias_delta[0].dot(self.__sigmoid(net_inputs[-i - 2].T)))
            weight_deltas_sum += weight_delta
            bias_deltas_sum += bias_delta
            error_sums.append(sum(errors ** 2))
        self.weights -= weight_deltas_sum
        self.biases -= bias_deltas_sum
        return np.sum(error_sums) / len(error_sums)

    def calculate_net_inputs(self, sample):
        net_inputs = [sample]
        for i in range(0, len(self.weights)):
            net_inputs.append(self.weights[i].dot(net_inputs[i]) + self.biases[i])
        return net_inputs

    def sample_learn(self, sample, target):
        net_inputs = self.calculate_net_inputs(sample)
        errors = self.__sigmoid(net_inputs[-1]) - target
        bias_deltas = [self.learning_rate * 2 * errors * self.__sigmoid(net_inputs[-1], derivative=True)]
        weights_deltas = [bias_deltas[0].dot(self.__sigmoid(net_inputs[-2].T))]
        for i in range(1, len(self.weights)):
            bias_deltas.insert(0, self.learning_rate
                               * self.__sigmoid(net_inputs[-i - 1], derivative=True)
                               * self.weights[-i].T.dot(bias_deltas[-i]))
            weights_deltas.insert(0, bias_deltas[0].dot(self.__sigmoid(net_inputs[-i - 2].T)))
        self.weights = self.weights - weights_deltas
        self.biases = self.biases - bias_deltas
        return sum(errors ** 2)

    def predict(self, sample):
        activations = self.__sigmoid(self.weights[0].dot(sample) + self.biases[0])
        for i in range(1, len(self.weights)):
            activations = self.__sigmoid(self.weights[i].dot(activations) + self.biases[i])
        return activations

    def test(self, samples, targets):
        successes = 0
        for sample, target in zip(samples, targets):
            result = np.zeros(len(self.weights[-1]))
            result[np.argmax(self.predict(sample))] = 1
            successes += int(np.array_equal(result, np.ndarray.flatten(target)))
        return successes / len(targets)

    def __str__(self):
        return f'Sigmoid neuron: \n ' \
            f'Features: {self.features_num}; \n ' \
            f'Learning rate: {self.learning_rate}; \n ' \
            f'Beta: {self.beta} \n ' \
            f'Weights: \n {self.weights} \n ' \
            f'Biases: \n {self.biases}'
