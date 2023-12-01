"""
Implementation of a layer of LIF neurons using TensorFlow
"""
from collections import namedtuple

import tensorflow as tf

InternalState = namedtuple('InternalState', ('u', 'r', 'e', 'eps'))


class LIFLayer(tf.keras.layers.Layer):
    def __init__(self, n_neurons: int, rest: float = 0.0, threshold: float = 0.6, refractory: float = 0.0,
                 tau: float = 10.0, gamma: float = 0.3, lr: float = 0.01, dt: float = 1.0,
                 is_output_layer: bool = False, batch_size: int = 1, **kwargs) -> None:
        """
        Custom LIF neuron layer, randomly generates weights between w_min and w_max

        :param n_neurons: number of outputs
        :param rest: resting potential in mV
        :param threshold: firing threshold in mV
        :param refractory: refractory period - time to refract after a spike - in ms
        :param tau: decay constant in ms
        :param gamma: dampening factor for the pseudo-derivative 'h' for the eligibility traces
        :param lr: learning rate
        :param dt: length of a simulation time-step in ms
        :param is_output_layer: boolean value whether this layer is the output layer
        """

        super().__init__(**kwargs)
        self.n_neurons = n_neurons
        self.n_inputs = 1
        self.batch_size = batch_size

        self.resting_potential = rest
        self.threshold = threshold
        self.refractory = refractory
        self.tau = tau
        self.gamma = gamma
        self.lr = lr
        self.dt = dt

        self.W = None
        self.is_output = is_output_layer
        if not is_output_layer:
            self.B = None

        self.alpha = dt / tau

    def build(self, input_shape: list) -> None:
        self.n_inputs = input_shape[0]
        xavier_init_val = 1 / tf.sqrt(tf.cast(self.n_inputs, dtype=tf.float32))
        self.W = tf.Variable(
            tf.random.uniform(
                minval=-xavier_init_val,
                maxval=xavier_init_val,
                shape=(self.n_neurons, input_shape[0]),
                dtype=tf.float32
            ),
            trainable=False
        )

    def build_remaining(self, shape: list) -> None:
        xavier_init_val = 1 / tf.sqrt(tf.cast(self.n_neurons, dtype=tf.float32))
        self.B = tf.random.uniform(
            minval=-xavier_init_val,
            maxval=xavier_init_val,
            shape=(self.n_neurons, shape[0]),
            dtype=tf.float32
        )

    def zero_state(self) -> InternalState:
        return InternalState(
            u=tf.fill([self.n_neurons, self.batch_size], self.resting_potential),
            r=tf.zeros(shape=(self.n_neurons, self.batch_size), dtype=tf.float32),
            e=tf.zeros(shape=(self.n_neurons, self.n_inputs), dtype=tf.float32),
            eps=tf.zeros(shape=(self.n_inputs, self.batch_size), dtype=tf.float32)
        )

    def call(self, inputs: tf.Tensor, state: InternalState) -> (tf.Tensor, InternalState, tf.Tensor):
        """
        simulate one time-step, updates internal values and returns whether a spike occurred or not

        :param inputs: one dimensional tensor of inputs with size n_in
        :param state: previous state
        :return: boolean tensor of spike or non-spike values, internal state, h (pseudo-derivative)
        """

        u, r, e, eps = state

        # integration
        new_v = tf.linalg.matmul(self.W, inputs)
        new_u = u + new_v * self.dt / self.tau

        # sort out resting neurons
        resting_ = r - self.dt > 0.
        new_u = tf.where(resting_, u, new_u)
        new_r = tf.where(resting_, r - self.dt, 0.)

        # firing
        firing_ = new_u > self.threshold
        new_u = tf.where(firing_, new_u - self.threshold, new_u)
        new_r = tf.where(firing_, self.refractory, new_r)
        z = tf.where(firing_, 1., 0.)

        new_eps = tf.add(self.alpha * eps, inputs)
        h = self.gamma * tf.maximum(0., 1. - tf.abs((new_u - self.threshold) / self.threshold))

        new_e = tf.einsum('ib,jb->ij', h, new_eps)

        # new internal state
        state = InternalState(new_u, new_r, new_e, new_eps)

        return z, state, h

    def update(self, error: tf.Tensor, inputs: tf.Tensor, h: tf.Tensor) -> None:
        if not self.is_output and self.B is None:
            self.build_remaining(error.shape)

        if not self.is_output:
            delta = tf.linalg.matmul(self.B, error)
            iota = tf.math.multiply(delta, h)
        else:
            iota = error

        dw = tf.einsum('ib,jb->ij', iota, inputs) / self.batch_size

        self.W.assign_sub(self.lr * dw)
