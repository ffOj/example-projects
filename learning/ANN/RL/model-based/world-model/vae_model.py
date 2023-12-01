
import tensorflow as tf
import tensorflow_probability as tfp


def reconstruction_loss(actual, prediction):
    return tf.norm(actual - prediction, ord='euclidean')


def kl_divergence(distribution_params):
    distribution_a = tfp.distributions.Normal(loc=distribution_params[0], scale=distribution_params[1])
    distribution_b = tfp.distributions.Normal(loc=0.0, scale=1.0)
    return tfp.distributions.kl_divergence(distribution_a, distribution_b)


class VaeModel(tf.keras.Model):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, x, *args, **kwargs):
        params = self.encoder(x)
        z = params[0] + params[1] * tf.random.normal(shape=params[0].shape)
        return z, self.decoder(z), params

    def latent_vector(self, x):
        params = self.encoder(x)
        return params[0] + params[1] * tf.random.normal(shape=params[0].shape)

    def train_step(self, data):
        with tf.GradientTape(persistent=True) as tape:
            z, out, params = self(data)

            loss = reconstruction_loss(data, out) + kl_divergence((params[0][0], params[1][0] + 0.0001))

        grads_decoder = tape.gradient(loss, self.decoder.trainable_variables)
        grads_encoder = tape.gradient(loss, self.encoder.trainable_variables)

        self.encoder.optimizer.apply_gradients(zip(grads_encoder, self.encoder.trainable_variables))
        self.decoder.optimizer.apply_gradients(zip(grads_decoder, self.decoder.trainable_variables))

        self.compiled_metrics.update_state(data, out)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        _, out, _ = self(data)

        self.compiled_metrics.update_state(data, out)
        return {m.name: m.result() for m in self.metrics}
