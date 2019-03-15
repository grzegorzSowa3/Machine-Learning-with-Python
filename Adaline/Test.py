import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Adaline import Adaline

# crucial parameters

EPOCHS_NUM = 3
LEARNING_RATE = 0.1
BATCH_SIZE = 15
SPECIES = {'Iris-setosa': 1.0, 'Iris-versicolor': -1.0}

# load data and throw away Iris-virginica

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None).values
data = data[:-50]

# shuffle data and extract targets

np.random.shuffle(data)
samples = []
targets = []
for row in data:
    samples.append(row[:-1])
    targets.append(SPECIES[row[-1]])
samples = np.array(samples)
targets = np.array(targets)

# construct and learn perceptron

network = Adaline(4, learning_rate=LEARNING_RATE, epochs_num=EPOCHS_NUM, batch_size=BATCH_SIZE)
errors = network.learn(samples, targets)

# plot learning curve

print(np.shape(errors))
plt.scatter(range(1, len(errors) + 1), errors, color='black', marker='o', label='Errors')
plt.xlabel('Batch')
plt.ylabel('Mean Squared Error')
plt.show()
