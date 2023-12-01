from ppo import PPOModel, log_probs
from buffer import Buffer

import matplotlib.pyplot as plt
import numpy as np
import gym
import tensorflow as tf

env = gym.make("LunarLander-v2", render_mode=None, new_step_api=True)

actor = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation=None)
])
actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mae')
critic = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation=None)
])
critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mae')

rewards = []
n_epochs = 1000
time_steps = 800
n_update_steps = 80
epsilon = 0.2
gamma = 0.99
lmbd = 0.97

model = PPOModel(actor, critic, epsilon, gamma, lmbd)
model.compile(optimizer='adam', loss='mae')
buffer = Buffer(time_steps, 8)

if __name__ == "__main__":
    state = env.reset()

    acc_reward = 0
    for epoch in range(n_epochs):
        for t in range(time_steps):
            logits, action = model.predict(state)
            new_state, reward, done, _, _ = env.step(action)

            log_prob = log_probs(logits, action)
            value = model.predict_critic(new_state)
            buffer.add_data(new_state, action, log_prob, reward, value)

            state = new_state

            acc_reward += reward

            if done or (t == time_steps - 1):
                last_value = 0 if done else model.predict_critic(state)
                buffer.finish_trajectory(gamma, lmbd, last_value)

                state = env.reset()

        s, a, lp, _, _, ret, adv = buffer.get()
        for _ in range(n_update_steps):
            model.train_step(s, a, lp, ret, adv)
        buffer.reset()

        print('epoch', epoch, acc_reward)
        rewards.append(acc_reward)
        acc_reward = 0

    env = gym.make('LunarLander-v2', render_mode='human', new_step_api=True)
    for _ in range(3):
        state, done, step = env.reset(), False, 0
        while not done and step < 400:
            _, action = model.predict(state)
            state, _, done, _, _ = env.step(action)
            step += 1

    plt.plot(np.arange(len(rewards)), np.array(rewards))
    plt.xlabel('epochs')
    plt.ylabel('rewards')
    plt.title('accumulated rewards')
    plt.show()
    env.close()
