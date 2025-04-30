import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# %% [1. Introduction to Distributed Training]
# Learn how to scale Keras models with distributed training.
# Covers data parallelism, multi-GPU training, and distributed datasets.

print("TensorFlow/Keras version:", tf.__version__)

# %% [2. Data Parallelism with MirroredStrategy]
# Train a model using MirroredStrategy for data parallelism.
strategy = tf.distribute.MirroredStrategy()
print("\nNumber of devices:", strategy.num_replicas_in_sync)
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
with strategy.scope():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5, batch_size=128,
                    validation_data=(X_test, y_test), verbose=0)
print("Model trained with MirroredStrategy.")

# %% [3. Multi-GPU Training]
# Explicit multi-GPU training (same as MirroredStrategy in most cases).
with strategy.scope():
    model_multi_gpu = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model_multi_gpu.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_multi_gpu = model_multi_gpu.fit(X_train, y_train, epochs=5, batch_size=128,
                                        validation_data=(X_test, y_test), verbose=0)
print("Multi-GPU training completed.")

# %% [4. Distributed Datasets]
# Use distributed datasets with tf.data.
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(128)
train_dataset = strategy.experimental_distribute_dataset(train_dataset)
with strategy.scope():
    model_dist = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model_dist.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_dist.fit(train_dataset, epochs=5, verbose=0)
print("Model trained with distributed dataset.")

# %% [5. Practical ML Application]
# Visualize distributed training performance.
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Distributed Training: Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Distributed Training: Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('distributed_training_performance.png')
plt.close()
print("Distributed training performance saved as 'distributed_training_performance.png'")

# %% [6. Interview Scenario: Distributed Training]
# Discuss distributed training for ML.
print("\nInterview Scenario: Distributed Training")
print("Q: How do you implement distributed training in Keras?")
print("A: Use tf.distribute.MirroredStrategy for data parallelism across GPUs.")
print("Key: MirroredStrategy synchronizes gradients across devices.")
print("Example: strategy = tf.distribute.MirroredStrategy(); with strategy.scope(): ...")