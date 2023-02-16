import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_matrix(elements):
    matrix = elements.reshape(28, 28)
    plt.imshow(matrix, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    mint = tf.keras.datasets.mnist
    (train_x, train_y), (test_x, test_y) = mint.load_data()
    print(train_x.shape, train_y.shape) # train data, 60000, 28*28
    print(test_x.shape, test_y.shape) # test data, 10000
    # the data sets have 70000 samples.

    plot_matrix(train_x[0])
