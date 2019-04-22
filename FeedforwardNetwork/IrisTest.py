import pandas as pd
from ActivationFunctions import *
import numpy as np
import matplotlib.pyplot as plt
from FeedforwardNetwork import FeedforwardNetwork

# crucial parameters

EPOCHS_NUM = 500
LEARNING_RATE = 0.3
BATCH_SIZE = 10
SHAPE = [4, 4, 3]
ACTIVATION_FUNCTIONS = [tanh, tanh, softmax]
L2_REGULARIZATION_FACTOR = 0.01
SPECIES = {'Iris-setosa': [1, 0, 0], 'Iris-versicolor': [0, 1, 0], 'Iris-virginica': [0, 0, 1]}

# load data
print("loading data...")
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None).values
print("data loaded")

# shuffle data and extract targets

np.random.shuffle(data)
learn_samples = [(row[:-1] / 10).tolist() for row in data[:-30]]
learn_targets = [SPECIES[row[-1]] for row in data[:-30]]
learn_samples = np.array(learn_samples, dtype=np.float32)
learn_targets = np.array(learn_targets, dtype=np.float32)

test_samples = [(row[:-1] / 10) for row in data[-30:]]
test_targets = [np.array(SPECIES[row[-1]]) for row in data[-30:]]
test_samples = np.array(test_samples, dtype=np.float32)
test_targets = np.array(test_targets, dtype=np.float32)

# construct and learn neuron

network = FeedforwardNetwork(4, SHAPE, ACTIVATION_FUNCTIONS, learning_rate=LEARNING_RATE,
                             regularization_factor=L2_REGULARIZATION_FACTOR)

errors = network.learn(learn_samples, learn_targets, epochs_num=EPOCHS_NUM, batch_size=BATCH_SIZE)
success_rate = network.test(learn_samples, learn_targets)
# success_rates.append(network.test(learn_samples, learn_targets))

print(f"Neural network with shape: {SHAPE}, learning rate: {LEARNING_RATE};")
print(f"Trained on {len(learn_samples)} train samples, epochs: {EPOCHS_NUM}, batch size: {BATCH_SIZE};")
print(f"Tested on {len(test_samples)} test samples;")
print()
print(f"Success rate: {success_rate};")

plt.scatter(range(1, len(errors) + 1), errors, color='black', marker='o', label='Errors')
plt.xlabel('Batch')
plt.ylabel('Mean squared error')
plt.show()
