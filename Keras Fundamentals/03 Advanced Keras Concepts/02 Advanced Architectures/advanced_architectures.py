import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# %% [1. Introduction to Advanced Architectures]
# Learn advanced Keras architectures for ML.
# Covers Transformers, generative models (VAEs, GANs), and autoencoders.

print("TensorFlow/Keras version:", tf.__version__)

# %% [2. Transformers (Vision Transformer)]
# Build a simple Vision Transformer for CIFAR-10.
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
def create_vit_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    x = keras.layers.Reshape((30*30, 64))(x)
    for _ in range(2):
        x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = keras.layers.LayerNormalization()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    return keras.Model(inputs, outputs)
vit_model = create_vit_model()
vit_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
vit_history = vit_model.fit(X_train, y_train, epochs=3, batch_size=64,
                            validation_data=(X_test, y_test), verbose=0)
print("\nVision Transformer trained on CIFAR-10.")

# %% [3. Generative Models (VAE)]
# Build a Variational Autoencoder (VAE).
(X_train, _), (_, _) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
latent_dim = 2
encoder_inputs = keras.Input(shape=(784,))
x = keras.layers.Dense(256, activation='relu')(encoder_inputs)
z_mean = keras.layers.Dense(latent_dim)(x)
z_log_var = keras.layers.Dense(latent_dim)(x)
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
decoder_inputs = keras.Input(shape=(latent_dim,))
x = keras.layers.Dense(256, activation='relu')(decoder_inputs)
outputs = keras.layers.Dense(784, activation='sigmoid')(x)
decoder = keras.Model(decoder_inputs, outputs, name='decoder')
vae_outputs = decoder(encoder(encoder_inputs)[2])
vae = keras.Model(encoder_inputs, vae_outputs)
vae_loss = keras.losses.BinaryCrossentropy()(encoder_inputs, vae_outputs)
vae_loss += -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae_history = vae.fit(X_train, X_train, epochs=5, batch_size=128, verbose=0)
print("VAE trained on MNIST.")

# %% [4. Generative Models (GAN)]
# Build a simple GAN for MNIST.
generator = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(100,)),
    keras.layers.Dense(784, activation='sigmoid')
])
discriminator = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    keras.layers.Dense(1, activation='sigmoid')
])
discriminator.compile(optimizer='adam', loss='binary_crossentropy')
discriminator.trainable = False
gan_inputs = keras.Input(shape=(100,))
gan_outputs = discriminator(generator(gan_inputs))
gan = keras.Model(gan_inputs, gan_outputs)
gan.compile(optimizer='adam', loss='binary_crossentropy')
def train_gan(epochs=5, batch_size=128):
    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise, verbose=0)
        real_imgs = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
train_gan(epochs=3)
print("GAN trained on MNIST.")

# %% [5. Practical ML Application]
# Visualize VAE generated images and training performance.
generated = decoder.predict(np.random.normal(0, 1, (16, latent_dim)))
plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.savefig('vae_generated_images.png')
plt.close()
plt.plot(vae_history.history['loss'], label='VAE Loss')
plt.title('VAE Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('vae_training_loss.png')
plt.close()
print("VAE generated images and training loss saved.")

# %% [6. Interview Scenario: Advanced Architectures]
# Discuss advanced architectures for ML.
print("\nInterview Scenario: Advanced Architectures")
print("Q: How do you implement a GAN in Keras?")
print("A: Build separate generator and discriminator models, combine in a GAN.")
print("Key: GANs require adversarial training for generative tasks.")
print("Example: gan = keras.Model(gan_inputs, discriminator(generator(gan_inputs)))")