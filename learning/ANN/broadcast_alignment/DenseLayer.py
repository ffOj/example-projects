"""
Implementation of a dense layer with custom broadcast alignment update rule
"""

import tensorflow as tf


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, n_neurons: int, learning_rate: float = 0.1,
                 mu: float = 0.0, sigma: float = 0.01, is_output_layer: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n_neurons = n_neurons
        self.lr = learning_rate
        self.is_output = is_output_layer

        self.mean = mu
        self.std = sigma

        self.W = None
        self.b = None
        if not is_output_layer:
            self.B = None
        self.n_inputs = None
        self.inputs = None
        self.gradient = None

    def build(self, input_shape: list) -> None:
        self.n_inputs = input_shape[0]
        self.W = tf.Variable(
            tf.zeros(
                shape=(self.n_neurons, input_shape[0]),
                dtype=tf.float32
            ), trainable=True
        )

        self.b = tf.Variable(
            tf.zeros(
                shape=(self.n_neurons,),
                dtype=tf.float32
            ), trainable=True
        )

    def build_remaining(self, error_shape: list) -> None:
        xavier_init_val = 1.0 / tf.sqrt(tf.cast(self.n_neurons, dtype=tf.float32))
        self.B = tf.random.uniform(
            minval=-xavier_init_val,
            maxval=xavier_init_val,
            shape=(self.n_neurons, error_shape[0]),
            dtype=tf.float32
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        self.inputs = inputs

        with tf.GradientTape() as grad:
            z = tf.linalg.matvec(self.W, inputs) + self.b
            a = tf.tanh(z)

        self.gradient = grad.gradient(a, z)

        return a

    def update(self, error: tf.Tensor) -> None:
        if not self.is_output and self.B is None:
            self.build_remaining(error.shape)

        if not self.is_output:
            delta = tf.math.multiply(tf.linalg.matvec(self.B, error), self.gradient)
        else:
            delta = error

        self.W.assign_sub(self.lr * tf.linalg.einsum('i,j->ij', delta, self.inputs))
        self.b.assign_sub(self.lr * delta)

