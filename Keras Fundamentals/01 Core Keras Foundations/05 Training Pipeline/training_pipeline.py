import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# %% [1. Introduction to Training Pipeline]
# Learn how to set up a complete Keras training pipeline.
# Covers training loops, model saving/loading, GPU training, and TensorBoard.

print("TensorFlow/Keras version:", tf.__version__)

# %% [2. Training/Evaluation Loops]
# Train and evaluate a model on MNIST.
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5, batch_size=128,
                    validation_data=(X_test, y_test), verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("\nTest accuracy:", test_acc)

# %% [3. Model Saving/Loading]
# Save and load the model.
model.save('mnist_model.h5')
loaded_model = keras.models.load_model('mnist_model.h5')
loaded_loss, loaded_acc = loaded_model.evaluate(X_test, y_test, verbose=0)
print("Loaded model test accuracy:", loaded_acc)

# %% [4. GPU Training]
# Train on GPU if available.
with tf.device('/GPU:0'):
    model_gpu = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model_gpu.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history_gpu = model_gpu.fit(X_train, y_train, epochs=5, batch_size=128, verbose=0)
print("GPU training completed.")

# %% [5. Monitoring with TensorBoard]
# Set up TensorBoard callback (commented for non-interactive environments).
tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
model_tb = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model_tb.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_tb.fit(X_train, y_train, epochs=5, batch_size=128,
             validation_data=(X_test, y_test), callbacks=[tensorboard_callback], verbose=0)
print("TensorBoard training completed.")

# %% [6. Practical ML Application]
# Visualize training performance.
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training Pipeline Performance')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.savefig('pipeline_performance.png')
plt.close()
print("Pipeline performance plot saved as 'pipeline_performance.png'")

# %% [7. Interview Scenario: Training Pipeline]
# Discuss training pipelines for ML.
print("\nInterview Scenario: Training Pipeline")
print("Q: How do you set up a Keras training pipeline?")
print("A: Compile model, use model.fit with callbacks, save with model.save.")
print("Key: Callbacks like TensorBoard monitor training.")
print("Example: model.fit(X, y, callbacks=[TensorBoard()])")