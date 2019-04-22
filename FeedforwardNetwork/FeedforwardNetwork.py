import numpy as np


class FeedforwardNetwork:

    # Multi-layer neural network consisting of sigmoid neurons

    def __init__(self, features_num, layer_sizes, activation_functions, learning_rate=0.5):
        # weights for first layer
        self.layer_weights = [np.random.rand(layer_sizes[0], features_num) * 0.01]
        # weights for further layers
        for i in range(1, len(layer_sizes)):
            self.layer_weights.append(np.random.rand(layer_sizes[i], layer_sizes[i - 1]) * 0.01)
        # biases for all layers
        self.layer_biases = []
        for layer_size in layer_sizes:
            self.layer_biases.append(np.zeros((layer_size, 1)))
        # hiperparameters
        self.features_num = features_num
        self.layers_num = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.learning_rate = learning_rate

    def learn(self, samples, targets, epochs_num=5, batch_size=15):
        errors_history = []
        for i in range(epochs_num):
            shuffle = list(zip(samples, targets))
            np.random.shuffle(shuffle)
            samples, targets = zip(*shuffle)
            samples = np.asarray(samples)
            targets = np.asarray(targets)
            samples_batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
            targets_batches = [targets[i:i + batch_size] for i in range(0, len(targets), batch_size)]
            for samples_batch, targets_batch in zip(samples_batches, targets_batches):
                errors_history.append(self.__batch_learn(samples_batch.T, targets_batch.T, len(samples_batch)))
            print("Epoch: " + str(i + 1) + " mean squared error: " + str(errors_history[-1]))
        return errors_history

    def __batch_learn(self, samples, targets, m):
        net_inputs, activations = self.__propagate_forward(samples)
        activations.insert(0, samples)
        d_net_inputs = [activations[-1] - targets]
        d_biases = [np.sum(d_net_inputs[-1], axis=1, keepdims=True) / m]
        d_weights = [np.dot(d_net_inputs[-1], activations[-2].T) / m]
        for i in range(1, self.layers_num):
            d_net_inputs.insert(0, self.layer_weights[-i].T.dot(d_net_inputs[-i])
                                * self.activation_functions[-i - 1](net_inputs[-i - 1], derivative=True))
            d_weights.insert(0, np.dot(d_net_inputs[0], activations[-i - 2].T) / m)
            d_biases.insert(0, np.sum(d_net_inputs[0], axis=1, keepdims=True) / m)
        for i in range(0, len(self.layer_weights)):
            self.layer_weights[i] -= self.learning_rate * d_weights[i]
            self.layer_biases[i] -= self.learning_rate * d_biases[i]
        return np.sum(d_net_inputs[-1] ** 2 / m)

    def __propagate_forward(self, samples):
        net_inputs = [self.layer_weights[0].dot(samples) + self.layer_biases[0]]
        activations = [self.activation_functions[0](net_inputs[0])]
        for i in range(1, len(self.layer_weights)):
            net_inputs.append(self.layer_weights[i].dot(activations[-1]) + self.layer_biases[i])
            activations.append(self.activation_functions[i](net_inputs[i]))
        return net_inputs, activations

    def predict(self, sample):
        activations = self.activation_functions[0](self.layer_weights[0].dot(sample) + self.layer_biases[0])
        for i in range(1, len(self.layer_weights)):
            activations = self.activation_functions[i](self.layer_weights[i].dot(activations) + self.layer_biases[i])
        return activations

    def test(self, samples, targets):
        successes = 0
        for sample, target in zip(samples, targets):
            result = np.zeros(len(self.layer_weights[-1]))
            result[np.argmax(self.predict(np.reshape(sample, (-1, 1))))] = 1
            successes += int(np.array_equal(result, np.ndarray.flatten(target)))
        return successes / len(targets)

    def __str__(self):
        return f'Sigmoid neuron: \n ' \
            f'Features: {self.features_num}; \n ' \
            f'Learning rate: {self.learning_rate}; \n ' \
            f'Weights: \n {self.layer_weights} \n ' \
            f'Biases: \n {self.layer_biases}'
