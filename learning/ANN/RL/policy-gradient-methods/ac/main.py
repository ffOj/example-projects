from policy_gradient_methods.ac import actor_critic as ac

import matplotlib.pyplot as plt
import numpy as np
import gym
import tensorflow as tf

env = gym.make("LunarLander-v2", render_mode='human', new_step_api=True)

actor = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3), loss='mae')
critic = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation=None)
])
critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3), loss='mae')

rewards = []
n_epochs = 10000

model = ac.ActorCriticModel(actor, critic, 0.99)
continuously_high = 0
for i in range(n_epochs):
    r = model.train_step(env)
    rewards.append(r)
    print('epoch:', i+1, "has reward:", r)
    if r > 100:
        continuously_high += 1
        if continuously_high > 50:
            break
    else:
        continuously_high = 0


plt.figure()
plt.plot(np.arange(n_epochs), rewards)
plt.xlabel('epoch')
plt.ylabel('cumulative reward')
plt.show()

env.close()

