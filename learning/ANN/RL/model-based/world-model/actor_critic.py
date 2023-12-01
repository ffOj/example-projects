import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

a0 = [1, 0, 0]
a1 = [0, 1, 0]
a2 = [0, 0, 1]
actions = [a0, a1, a2]


class ActorCriticModel(tf.keras.Model):
    def __init__(self, actor: tf.keras.Model, critic: tf.keras.Model, gamma: float = 0.99):
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.gamma = gamma

    def train_step(self, env, vae, _from, _to, steps):
        acc_reward = 0
        done = False

        state = env.reset()
        img = tf.expand_dims(state[_from:_to, _from:_to], axis=0) / 255
        latent_vector = vae.latent_vector(img)

        c = 0
        while not done and c < steps:
            c += 1
            with tf.GradientTape(persistent=True) as tape:
                img = tf.expand_dims(state[_from:_to, _from:_to], axis=0) / 255
                new_latent_vector = vae.latent_vector(img)
                action_distribution = self.actor(new_latent_vector)
                dist = tfp.distributions.Categorical(probs=action_distribution, dtype=tf.float32)
                action = actions[int(dist.sample()[0])]

                # perform action and observe S', R
                (new_state, reward, done, _, _) = env.step(action)

                # compute the error
                v_state, v_new_state = self.critic(np.squeeze(np.array([latent_vector, new_latent_vector])))

                target = reward + v_new_state * (1 - int(done))
                td_error = target - v_state

                # compute loss
                loss_actor = -dist.log_prob(action)[0] * td_error
                loss_critic = td_error ** 2

            latent_vector = new_latent_vector

            grads_actor = tape.gradient(loss_actor, self.actor.trainable_variables)
            grads_critic = tape.gradient(loss_critic, self.critic.trainable_variables)

            acc_reward += reward

            self.actor.optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
            self.critic.optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

        return acc_reward
