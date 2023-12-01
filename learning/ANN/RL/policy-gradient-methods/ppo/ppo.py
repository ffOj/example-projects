"""
implementation of proximal policy optimisation
"""
import numpy as np
import tensorflow as tf


def log_probs(logits, a):
    prob_dist = tf.nn.log_softmax(logits)
    log_prob = tf.reduce_sum(
        tf.one_hot(a, 4) * prob_dist, axis=1
    )
    return log_prob


class PPOModel(tf.keras.Model):
    def __init__(self, actor, critic, epsilon, gamma, lmbd):
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.eps = epsilon
        self.gamma = gamma
        self.lmbd = lmbd

    def predict(self, state):
        if len(state.shape) == 1:
            state = np.array([state])
        prob = self.actor(state)
        action = tf.squeeze(tf.random.categorical(prob, 1), axis=1)
        return prob, action[0].numpy()

    def predict_critic(self, state):
        return self.critic(np.array([state]))

    @tf.function
    def train_step(self, s, a, lp, ret, adv):
        with tf.GradientTape(persistent=True) as tape:
            lp_new = log_probs(self.actor(s), a)
            ratio = tf.exp(lp_new - lp)
            clipped = tf.where(adv < 0, 1 - self.eps, 1 + self.eps) * adv

            loss_actor = -tf.reduce_mean(
                tf.minimum(ratio * adv, clipped)
            )
            loss_critic = tf.reduce_mean((ret - self.critic(s)) ** 2)

        gradients_actor = tape.gradient(loss_actor, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(gradients_actor, self.actor.trainable_variables))

        gradients_critic = tape.gradient(loss_critic, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(gradients_critic, self.critic.trainable_variables))

        return {m.name: ret[-1] for m in self.metrics}
