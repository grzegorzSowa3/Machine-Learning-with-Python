import pandas as pd
from Perceptron import Perceptron

ITER_NUM = 5
SPECIES = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

print("Running tests...")

print("Downloading iris data...")
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

neurons = [Perceptron(0.5, 4)] * 3

for neuron in neurons:
    print(str(neuron))

for sample in data.values:
    for neuron_num, neuron in enumerate(neurons):
        if neuron_num == SPECIES[sample[-1]]:
            neuron.learn(sample[:-1], 1)
        else:
            neuron.learn(sample[:-1], 0)

for neuron in neurons:
    print(str(neuron))
