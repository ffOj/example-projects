"""
Implementation of a layer of LIF neurons using TensorFlow
"""
from collections import namedtuple

import tensorflow as tf

# (membrane potential, 'resting times', 'drive (weighted input)')
InternalState = namedtuple('InternalState', ('u', 'r', 'v'))


class LIFLayer(tf.keras.layers.Layer):
    def __init__(self, n_neurons: int, rest: float = 0.0, threshold: float = 0.4, refractory: float = 1.0,
                 tau: float = 20.0, lr: float = 0.005, dt: float = 0.25,
                 mu: float = 0.0, sigma: float = 0.1, epsilon: float = 0.001, **kwargs) -> None:
        """
        Custom LIF neuron layer, randomly generates weights normally distributed with mu and sigma

        :param n_neurons: number of outputs
        :param rest: resting potential in mV
        :param threshold: firing threshold in mV
        :param refractory: refractory period - time to refract after a spike - in ms
        :param tau: decay constant in ms
        :param gamma: dampening factor for the pseudo-derivative 'h' for the eligibility traces
        :param lr: learning rate
        :param dt: length of a simulation time-step in ms
        :param mu: mean for initialising weights and biases
        :param sigma: standard deviation for initialising weights and biases
        """

        super().__init__(**kwargs)
        self.n_neurons = n_neurons
        self.n_inputs = None
        self.resting_potential = rest
        self.threshold = threshold
        self.refractory = refractory
        self.tau = tau
        self.lr = lr
        self.eps = epsilon
        self.dt = dt
        self.mean = mu
        self.std = sigma
        self.w = None
        self.b = None
        self.B = None
        self.state = None
        self.inputs = None
        self.alpha = tf.exp(-self.dt / self.tau)

    def build(self, input_shape: list) -> None:
        self.n_inputs = input_shape[0]
        self.inputs = tf.zeros(input_shape[0], dtype=tf.float32)

        self.w = tf.Variable(
            tf.random.normal(
                mean=self.mean,
                stddev=self.std,
                shape=(self.n_neurons, input_shape[0]),
                dtype=tf.float32
            ),
            trainable=True
        )
        self.b = tf.Variable(
            tf.random.normal(
                mean=self.mean,
                stddev=self.std,
                shape=(self.n_neurons,),
                dtype=tf.float32
            ),
            trainable=True
        )
        self.B = tf.random.normal(
            mean=self.mean,
            stddev=self.std,
            shape=(input_shape[0], self.n_neurons),
            dtype=tf.float32
        )
        self.state = self.zero_state()

    def zero_state(self) -> InternalState:
        return InternalState(
            u=tf.fill(self.n_neurons, self.resting_potential),
            r=tf.zeros(self.n_neurons, tf.float32),
            v=tf.zeros(self.n_neurons, tf.float32)
        )

    def call(self, inputs: tf.Tensor) -> (tf.Tensor, InternalState):
        """
        simulate one time-step, updates internal values and returns whether a spike occurred or not

        :param inputs: one dimensional tensor of inputs with size n_in
        :return: boolean tensor of spike or non-spike values
        """

        # old internal state
        u, r, _ = self.state

        v = tf.linalg.matvec(self.w, inputs) + self.b
        new_r = tf.where(r > 0., r + self.dt, r)
        new_r = tf.where(new_r > self.refractory, 0., new_r)
        new_u = tf.where(new_r > 0., self.resting_potential, u + (v - u) * self.dt / self.tau)
        new_r = tf.where(new_u > self.threshold, self.eps, new_r)

        # new observable state
        z = tf.where(new_r > 0, 1.0, 0.0)

        # new internal state
        self.state = InternalState(new_u, new_r, v)
        self.inputs = inputs

        return z

    def reset(self):
        self.state = self.zero_state()

    def update(self, error: tf.Tensor) -> tf.Tensor:
        # pseudo derivative
        h = tf.where(self.state.v > 0, (1 / tf.math.cosh(0.08 * self.state.v)) ** 2, 0.)

        self.w.assign_sub(self.lr * tf.linalg.einsum('i,j->ij', error, self.inputs))
        self.b.assign_sub(tf.multiply(self.lr, error))

        # error of n - 1
        return tf.linalg.matvec(self.B, tf.math.multiply(error, h))
