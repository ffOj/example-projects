
from SpikingNN.broadcast_alignment.LIFLayer import LIFLayer

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm


potentials = []
expectations = []

firing_neurons = []
firing_times = []

res = []


if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data(
        path='mnist.npz'
    )

    lr = 0.01
    l1 = LIFLayer(630, lr=lr)
    l2 = LIFLayer(370, lr=lr)
    l3 = LIFLayer(10, lr=lr, is_output_layer=True)

    n_images = 10
    n_steps = 60
    iterations = 40
    for it in tqdm(range(iterations)):
        for i, (x, y) in enumerate(zip(train_X[:n_images], train_y[:n_images])):
            l1.reset()
            l2.reset()
            l3.reset()

            inputs = tf.reshape(x, shape=28*28) / 255
            expected = tf.one_hot(y, 10)

            if it == iterations - 1:
                res.append([])
            for step in range(n_steps):
                out = l3(l2(l1(inputs)))

                if step > 20 and it < iterations - 1:
                    error = out - expected

                    l1.update(error)
                    l2.update(error)
                    l3.update(error)

                if it == iterations - 1:
                    res[i].append(out)

                potentials.append(l3.state.u)
                expectations.append(y)

                neurons = np.where(np.array(out) == 1.)
                firing_neurons.append(neurons[0])
                firing_times.append(np.zeros(neurons[0].size) + step + (i * n_steps) + (it * n_images * n_steps))

    print(np.argmax(np.array(res)[:].sum(axis=1), axis=1))
    print(train_y[:n_images])
    print(np.array(res)[:].sum(axis=1))

    xs = np.concatenate(firing_times).ravel()
    ys = np.concatenate(firing_neurons).ravel()

    r = np.arange(n_images * n_steps * iterations)

    fig, ax = plt.subplots(3, 1)
    ax[0].scatter(xs, ys)
    ax[1].plot(r, potentials)
    ax[1].legend(np.arange(10))
    ax[2].plot(r, expectations)
    plt.show()
