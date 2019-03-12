import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron

# crucial parameters

ITERS_NUM = 10
LEARNING_RATE = 0.7
SPECIES = {'Iris-setosa': 1.0, 'Iris-versicolor': -1.0}

# load data and throw away Iris-virginica

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None).values
data = data[:-50]

# plot sepal and petal length data

plt.scatter(data[:50, 0], data[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(data[51:, 0], data[51:, 1], color='blue', marker='o', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()

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

perceptron = Perceptron(4, 1, iters_num=ITERS_NUM)
errors = perceptron.learn(samples, targets)

# plot learning curve

plt.scatter(range(0, ITERS_NUM), errors, color='black', marker='o', label='Errors')
plt.xlabel('Iteration')
plt.ylabel('Errors')
plt.show()

print(str(perceptron))
