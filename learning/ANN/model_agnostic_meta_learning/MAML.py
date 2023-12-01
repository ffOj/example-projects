"""
MAML algorithm
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm as tqdm

tf.config.experimental.set_visible_devices([], 'GPU')


def classifier():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(120, activation='tanh'),
        tf.keras.layers.Dense(80, activation='tanh'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])


def clone_classifier(model, x):
    clone = classifier()
    clone(x)
    clone.set_weights(model.get_weights())

    return clone


def train_step(model, data, optimizer, loss_fn, lr):
    x, y = data
    model(x)

    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            out = model(x)
            loss = loss_fn(out, y)

        inner_gradients = inner_tape.gradient(loss, model.trainable_variables)

        cloned_model = clone_classifier(model, x)
        for i in range(len(model.layers)):
            j = i * 2
            cloned_model.layers[i].kernel = model.layers[i].kernel - lr * inner_gradients[j]
            cloned_model.layers[i].bias = model.layers[i].bias - lr * inner_gradients[j + 1]

        outer_loss = loss_fn(cloned_model(x), y)

    outer_gradients = outer_tape.gradient(outer_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(outer_gradients, model.trainable_variables))

    return model


if __name__ == "__main__":
    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data(
        path='mnist.npz'
    )
    train_X = tf.reshape(train_X, shape=(-1, 784,)) / 255
    test_X = tf.reshape(test_X, shape=(-1, 784,)) / 255

    train_y_ = np.zeros(shape=(train_y.shape[0], 10), dtype=np.float32)
    for i, e in enumerate(train_y):
        train_y_[i] = tf.one_hot(e, 10, dtype=tf.float32)

    test_y_ = np.zeros(shape=(test_y.shape[0], 10), dtype=np.float32)
    for i, e in enumerate(test_y):
        test_y_[i] = tf.one_hot(e, 10, dtype=tf.float32)

    n_images = 60000
    bs = 100
    epochs = 1

    model = classifier()

    for epoch in range(epochs):
        for i in tqdm(range(n_images)):
            model = train_step(model, (train_X[(i * bs):(i * bs) + bs], train_y_[(i * bs):(i * bs) + bs]),
                               tf.keras.optimizers.Adam(),
                               tf.keras.losses.CategoricalCrossentropy(), 0.001)

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.evaluate(test_X, test_y_)
