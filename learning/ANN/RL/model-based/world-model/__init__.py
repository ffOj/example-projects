from vae_model import VaeModel
from actor_critic import ActorCriticModel

import tensorflow as tf
import numpy as np
import gym

from matplotlib import pyplot as plt

env = gym.make('CarRacing-v2', render_mode=None, new_step_api=True)
env_size = 96

latent_space_size = 32
image_size = 64
half_diff_size = (96 - 64) // 2
_from = half_diff_size
_to = image_size+half_diff_size

input_layer = tf.keras.Input(shape=(image_size, image_size, 3))
hidden_layers = tf.keras.layers.Flatten()(
    tf.keras.layers.Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(
        tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(
            tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(
                tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation='relu',
                                       padding='same')(
                    input_layer)
            )
        ))
)
output_layer_mean = tf.keras.layers.Dense(units=latent_space_size, activation='relu')(hidden_layers)
output_layer_sigma = tf.keras.layers.Dense(units=latent_space_size, activation='relu')(hidden_layers)
encoder = tf.keras.Model(inputs=[input_layer], outputs=[output_layer_mean, output_layer_sigma])
encoder.compile(optimizer='adam', loss='mse')

decoder = tf.keras.Sequential([
    tf.keras.Input(shape=(latent_space_size,)),
    tf.keras.layers.Dense(units=1024, activation='relu'),
    tf.keras.layers.Reshape(target_shape=(1, 1, 1024)),
    tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(2, 2), activation='relu'),
    tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu'),
    tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(6, 6), strides=(2, 2), activation='relu'),
    tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(6, 6), strides=(2, 2), activation='sigmoid'),
])
decoder.compile(optimizer='adam', loss='mse')

vae = VaeModel(encoder, decoder)
vae.compile(optimizer='adam', loss='mse')


actor = tf.keras.Sequential([
    tf.keras.Input(shape=(latent_space_size,)),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=8, activation='relu'),
    tf.keras.layers.Dense(units=3, activation='sigmoid')
])
actor.compile(optimizer='adam', loss='mse')

critic = tf.keras.Sequential([
    tf.keras.Input(shape=(latent_space_size,)),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=8, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='linear')
])
critic.compile(optimizer='adam', loss='mse')

ac = ActorCriticModel(actor, critic)
ac.compile(optimizer='adam', loss='mse')

state = env.reset()

# initial training for vae model
epochs = 300
batch_size = 64
for epoch in range(epochs):
    states = np.zeros(shape=(batch_size, image_size, image_size, 3))
    for bs in range(batch_size):
        state, _, done, _, _ = env.step(np.array(tf.one_hot(np.random.choice(np.arange(3), size=1)[0], 3)))
        states[bs] = state[_from:_to, _from:_to]

    state = env.reset()

    data = tf.cast(
        tf.stack(states) / 255, tf.float32
    )
    vae.fit(data, epochs=1, batch_size=batch_size)

_ = env.reset()
for _ in range(50):
    state = env.step([0, 0.5, 0])[0]

plt.figure()
img = tf.expand_dims(state[_from:_to, _from:_to], axis=0) / 255
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img[0])
ax[1].imshow(tf.squeeze(vae(img)[1]))
plt.title('reconstruction of VAE network')
plt.show()
plt.close(fig)

# train action selector
time_steps = 1200
epochs = 1000
rewards = []

state = env.reset()
for epoch in range(epochs):
    reward = ac.train_step(env, vae, _from, _to, time_steps)
    rewards.append(reward)
    print('reward at epoch', epoch, 'is', reward)

plt.figure()
plt.plot(np.arange(epochs), rewards)
plt.xlabel('epoch')
plt.ylabel('reward')
plt.title('Car-Racing v2')


_ = env.reset()
for _ in range(50):
    state = env.step([0, 0.5, 0])[0]

plt.figure()
img = tf.expand_dims(state[_from:_to, _from:_to], axis=0) / 255
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img[0])
ax[1].imshow(tf.squeeze(vae(img)[1]))
plt.title('reconstruction of VAE network')
plt.show()
plt.close(fig)
