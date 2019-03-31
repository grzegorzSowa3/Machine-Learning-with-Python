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
samples = []
targets = []
for row in data:
    samples.append(row[:-1].reshape((-1, 1)) / 4)
    targets.append(np.array(SPECIES[row[-1]]).reshape((-1, 1)))
samples = np.array(samples, dtype=np.float32)
targets = np.array(targets, dtype=np.float32)

# construct and learn neuron

network = FeedforwardNetwork(4, [3], learning_rate=LEARNING_RATE, beta=BETA)

# print(network.calculate_net_inputs(samples[0]))
errors = network.learn(samples, targets, epochs_num=EPOCHS_NUM, batch_size=BATCH_SIZE)
# print(errors)
success_rate = network.test(samples, targets)
print(f"Success rate: {success_rate}")

# plot learning curve
plt.scatter(range(1, len(errors) + 1), errors, color='black', marker='o', label='Errors')
plt.xlabel('Try')
plt.ylabel('Mean Squared Error')
plt.show()
