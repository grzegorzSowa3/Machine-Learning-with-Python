import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SigmoidNeuron import SigmoidNeuron

# crucial parameters

EPOCHS_NUM = 10
LEARNING_RATE = 1
BATCH_SIZE = 10
BETA = 1.5
SPECIES = {'Iris-setosa': 1.0, 'Iris-versicolor': 0}

# load data and throw away Iris-virginica

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None).values
data = data[:-50]

# shuffle data and extract targets

np.random.shuffle(data)
samples = []
targets = []
for row in data:
    samples.append(row[:-1] / 10)
    targets.append(SPECIES[row[-1]])
samples = np.array(samples, dtype=np.float32)
targets = np.array(targets, dtype=np.float32)

# construct and learn neuron

network = SigmoidNeuron(4, learning_rate=LEARNING_RATE, beta=BETA)
errors = network.learn(samples, targets, epochs_num=EPOCHS_NUM, batch_size=BATCH_SIZE)
print(errors)

# plot learning curve

plt.scatter(range(1, len(errors) + 1), errors, color='black', marker='o', label='Errors')
plt.xlabel('Batch')
plt.ylabel('Mean Squared Error')
plt.show()
