from mlxtend.data import loadlocal_mnist
import numpy as np
import matplotlib.pyplot as plt

from FeedforwardNetwork import FeedforwardNetwork

EPOCHS_NUM = 100
LEARNING_RATE = 0.01
BATCH_SIZE = 100
SHAPE = [300]
OUTPUT_DICT = {
    0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

print("Preparing train data...")
train_samples, train_labels = loadlocal_mnist(
    images_path='D:/datasets/train-images.idx3-ubyte',
    labels_path='D:/datasets/train-labels.idx1-ubyte'
)
train_samples = [(train_input / 255).tolist() for train_input in train_samples]
train_labels = [OUTPUT_DICT[train_label] for train_label in train_labels]
train_samples = np.array(train_samples, dtype=np.float32)
train_labels = np.array(train_labels, dtype=np.float32)
print("Train data prepared.")

print("Preparing test data...")
test_samples, test_labels = loadlocal_mnist(
    images_path='D:/datasets/test-images.idx3-ubyte',
    labels_path='D:/datasets/test-labels.idx1-ubyte'
)
test_samples = [(test_input / 255).tolist() for test_input in test_samples]
test_labels = [OUTPUT_DICT[test_label] for test_label in test_labels]
test_samples = np.array(test_samples, dtype=np.float32)
test_labels = np.array(test_labels, dtype=np.float32)
print("Test data prepared.")

print(f"Training neural network, on {EPOCHS_NUM} epochs...")
network = FeedforwardNetwork(784, 10, SHAPE, learning_rate=LEARNING_RATE)
errors = network.learn(train_samples, train_labels, epochs_num=EPOCHS_NUM, batch_size=BATCH_SIZE)
print("Training completed.")

print("Plotting training results...")
plt.scatter(range(1, len(errors) + 1), errors, color='black', marker='o', label='Errors')
plt.xlabel('Batch')
plt.ylabel('Mean squared error')
plt.show()

print("Testing neural network...")
success_rate = network.test(train_samples, train_labels)

print(f"Neural network with shape: {SHAPE}, learning rate: {LEARNING_RATE};")
print(f"Trained on {len(train_samples)} train samples, epochs: {EPOCHS_NUM}, batch size: {BATCH_SIZE}")
print(f"Tested on {len(test_samples)} test samples;")
print()
print(f"Success rate: {success_rate};")
