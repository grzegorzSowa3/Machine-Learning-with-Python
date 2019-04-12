import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FeedforwardNetwork import FeedforwardNetwork

# crucial parameters

EPOCHS_NUM = 100
LEARNING_RATE = 0.8
BATCH_SIZE = 10
BETA = 1
SPECIES = {'Iris-setosa': [1, 0, 0], 'Iris-versicolor': [0, 1, 0], 'Iris-virginica': [0, 0, 1]}

# load data
print("loading data...")
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None).values
print("data loaded")

# shuffle data and extract targets

np.random.shuffle(data)
learning_samples = []
learning_targets = []
test_samples = []
test_targets = []
for row in data[:-50]:
    learning_samples.append(row[:-1].reshape((-1, 1)) / 10)
    learning_targets.append(np.array(SPECIES[row[-1]]).reshape((-1, 1)))
learning_samples = np.array(learning_samples, dtype=np.float32)
learning_targets = np.array(learning_targets, dtype=np.float32)
for row in data[-50:]:
    test_samples.append(row[:-1].reshape((-1, 1)) / 10)
    test_targets.append(np.array(SPECIES[row[-1]]).reshape((-1, 1)))
test_samples = np.array(test_samples, dtype=np.float32)
test_targets = np.array(test_targets, dtype=np.float32)

# construct and learn neuron

network = FeedforwardNetwork(4, [8, 3], learning_rate=LEARNING_RATE, beta=BETA)
success_rates = []
for i in range(1, EPOCHS_NUM):
    network.learn(learning_samples, learning_targets, epochs_num=1, batch_size=BATCH_SIZE)
    success_rates.append(network.test(learning_samples, learning_targets))

# plot learning curve
plt.scatter(range(1, len(success_rates) + 1), success_rates, color='black', marker='o', label='Errors')
plt.xlabel('Test')
plt.ylabel('Success rate')
plt.show()
