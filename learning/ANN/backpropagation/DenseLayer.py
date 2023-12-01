"""
Implementation of a dense layer with custom broadcast alignment update rule
"""

import tensorflow as tf


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, n_neurons: int, learning_rate: float = 0.1,
                 mu: float = 0.0, sigma: float = 0.01, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n_neurons = n_neurons
        self.lr = learning_rate

        self.mean = mu
        self.std = sigma

        self.W = None
        self.b = None
        self.n_inputs = None
        self.inputs = None
        self.gradient = None

    def build(self, input_shape: list) -> None:
        self.n_inputs = input_shape[0]
        self.W = tf.Variable(
            tf.random.normal(
                mean=self.mean,
                stddev=self.std,
                shape=(self.n_neurons, input_shape[0]),
                dtype=tf.float32
            ), trainable=True
        )

        self.b = tf.Variable(
            tf.random.normal(
                mean=self.mean,
                stddev=self.std,
                shape=(self.n_neurons,),
                dtype=tf.float32
            ), trainable=True
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        self.inputs = x

        with tf.GradientTape() as grad:
            z = tf.linalg.matvec(self.W, x) + self.b
            a = tf.tanh(z)

        self.gradient = grad.gradient(a, z)

        return a

    def update(self, error: tf.Tensor) -> tf.Tensor:
        delta = tf.math.multiply(error, self.gradient)

        self.W.assign_sub(self.lr * tf.linalg.einsum('i,j->ij', delta, self.inputs))
        self.b.assign_sub(self.lr * delta)

        return tf.linalg.matvec(tf.transpose(self.W), error)
