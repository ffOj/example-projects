from ANN.backpropagation.DenseLayer import DenseLayer

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm

res = []

if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data(
        path='mnist.npz'
    )

    lr = 0.01
    l1 = DenseLayer(630, learning_rate=lr)
    l2 = DenseLayer(370, learning_rate=lr)
    l3 = DenseLayer(10, learning_rate=lr)

    n_images = 20
    iterations = 50
    for it in tqdm(range(iterations)):
        for i, (x, y) in enumerate(zip(train_X[:n_images], train_y[:n_images])):
            inputs = tf.reshape(x, shape=28 * 28) / 255
            expected = tf.one_hot(y, 10)

            out = l3.__call__(l2.__call__(l1.__call__(inputs)))

            if it < iterations - 1:
                error = out - expected

                error = l3.update(error)
                error = l2.update(error)
                _ = l1.update(error)

            if it == iterations - 1:
                res.append(out)

    print(train_y[:n_images])
    print(np.array(res).argmax(axis=1))

    preds = np.array(res).argmax(axis=1)
    r = np.arange(n_images)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(r, preds)
    plt.show()
