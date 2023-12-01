import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class ActorCriticModel(tf.keras.Model):
    def __init__(self, actor: tf.keras.Model, critic: tf.keras.Model, gamma: float = 0.99):
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.gamma = gamma

    def train_step(self, env):
        acc_reward = 0
        done = False

        state = env.reset(seed=0)

        c = 0
        while not done and c < 200:
            c += 1
            env.render()
            with tf.GradientTape(persistent=True) as tape:
                action_distribution = self.actor(np.array([state]))
                dist = tfp.distributions.Categorical(probs=action_distribution, dtype=tf.float32)
                action = int(dist.sample()[0])

                # perform action and observe S', R
                (new_state, reward, done, _, _) = env.step(action)

                # compute the error
                v_state, v_new_state = self.critic(np.array([state, new_state]))[:, 0]

                target = reward + v_new_state * (1 - int(done))
                td_error = target - v_state

                # compute loss
                loss_actor = -dist.log_prob(action)[0] * td_error
                loss_critic = td_error ** 2

            state = new_state

            grads_actor = tape.gradient(loss_actor, self.actor.trainable_variables)
            grads_critic = tape.gradient(loss_critic, self.critic.trainable_variables)

            acc_reward += reward

            self.actor.optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
            self.critic.optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

        return acc_reward
