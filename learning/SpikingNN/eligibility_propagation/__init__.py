"""
Starts the simulation
"""

from SpikingNN.eligibility_propagation.LIFLayer import LIFLayer
import tensorflow as tf

import numpy as np
from tqdm import tqdm as tqdm

tf.config.experimental.set_visible_devices([], 'GPU')


def train_step(data: (tf.Tensor, tf.Tensor), epochs: int, time_steps: int, warmup_steps: int, batch_size: int,
               metric: tf.keras.metrics.Metric):
    xs, ys = data
    assert ((xs.shape[0] == ys.shape[0]) or (xs.shape[0] % batch_size == 0),
            "data needs to have the identical length and batch_size needs to fit")

    xs = tf.split(xs, len(xs) // batch_size)
    ys = tf.split(ys, len(ys) // batch_size)

    for _ in tqdm(range(epochs)):
        for x, y in zip(xs, ys):
            x = tf.transpose(x)
            y = tf.transpose(y)

            i1 = l1.zero_state()
            i2 = l2.zero_state()
            i3 = l3.zero_state()

            spikes = []
            for step in range(time_steps):
                z1, i1, h1 = l1.__call__(x, i1)
                z2, i2, h2 = l2.__call__(z1, i2)
                out, i3, h3 = l3.__call__(z2, i3)

                if step > warmup_steps:
                    error = tf.math.subtract(out, y)

                    l1.update(error, x, h1)
                    l2.update(error, z1, h2)
                    l3.update(error, z2, h3)

                    spikes.append(out)

            metric.update_state(np.argmax(y, axis=0), np.argmax(np.array(spikes).sum(axis=0), axis=0))
        print("accuracy:", metric.result().numpy())
        metric.reset_state()


def test_step(data: (tf.Tensor, tf.Tensor), time_steps: int, batch_size: int, warm_up: int,
              metric: tf.keras.metrics.Metric, logging: bool = False):
    # logging
    firing_neurons = []
    firing_times = []

    xs, ys = data

    xs = tf.split(xs, len(xs) // batch_size)
    ys = tf.split(ys, len(ys) // batch_size)

    for it, (x, y) in enumerate(zip(xs, ys)):
        x = tf.transpose(x)
        y = tf.transpose(y)

        i1 = l1.zero_state()
        i2 = l2.zero_state()
        i3 = l3.zero_state()

        spikes = []
        for step in range(time_steps):
            z1, i1, h1 = l1.__call__(x, i1)
            z2, i2, h2 = l2.__call__(z1, i2)
            out, i3, h3 = l3.__call__(z2, i3)

            if step > warm_up:
                spikes.append(out)

            if logging:
                neurons = np.where(np.array(out) == 1.)
                firing_neurons.append(neurons[0])
                firing_times.append(np.zeros(neurons[0].size) + step + (it * time_steps))

        metric.update_state(np.argmax(y, axis=0), np.argmax(np.array(spikes).sum(axis=0), axis=0))

    return metric.result().numpy(), firing_neurons, firing_times


if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data(
        path='mnist.npz'
    )

    train_X = tf.reshape(train_X, shape=(-1, 28 * 28)) / 255
    test_X = tf.reshape(test_X, shape=(-1, 28 * 28)) / 255

    train_y_ = np.zeros(shape=(train_y.shape[0], 10), dtype=np.float32)
    for i, e in enumerate(train_y):
        train_y_[i] = tf.one_hot(e, 10, dtype=tf.float32)

    test_y_ = np.zeros(shape=(test_y.shape[0], 10))
    for i, e in enumerate(test_y):
        test_y_[i] = tf.one_hot(e, 10, dtype=tf.float32)

    lr = 0.1
    l1 = LIFLayer(120, lr=lr)
    l2 = LIFLayer(80, lr=lr)
    l3 = LIFLayer(10, lr=lr, is_output_layer=True)

    n_images = 58880
    t_presentation = 20  # one training step

    train_step(data=(train_X[:n_images], train_y_[:n_images]), epochs=30, time_steps=t_presentation, warmup_steps=10,
               batch_size=128,
               metric=tf.keras.metrics.Accuracy())

    acc, fn, ft = test_step(
        data=(test_X, test_y_), time_steps=t_presentation, batch_size=100, warm_up=10,
        metric=tf.keras.metrics.Accuracy())

    print(acc)
