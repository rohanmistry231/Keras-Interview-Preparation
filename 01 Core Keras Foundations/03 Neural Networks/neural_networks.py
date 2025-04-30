import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# %% [1. Introduction to Neural Networks]
# Learn how to build and train feedforward neural networks in Keras.
# Covers FNNs, compiling, training, and callbacks.

print("TensorFlow/Keras version:", tf.__version__)

# %% [2. Building Feedforward Neural Networks]
# Create a simple FNN for classification.
model_fnn = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model_fnn.summary()
print("\nFNN created.")

# %% [3. Compiling Models]
# Compile the FNN with optimizer, loss, and metrics.
model_fnn.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
print("FNN compiled.")

# %% [4. Training Models]
# Train the FNN on MNIST dataset.
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
history = model_fnn.fit(X_train, y_train, epochs=5, batch_size=128,
                        validation_data=(X_test, y_test), verbose=0)
test_loss, test_acc = model_fnn.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", test_acc)

# %% [5. Callbacks]
# Train with callbacks: EarlyStopping, ModelCheckpoint.
callbacks = [
    keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
]
model_fnn.fit(X_train, y_train, epochs=10, batch_size=128,
              validation_data=(X_test, y_test), callbacks=callbacks, verbose=0)
print("FNN trained with callbacks.")

# %% [6. Practical ML Application]
# Visualize training performance.
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('FNN Training Performance')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.savefig('fnn_performance.png')
plt.close()
print("FNN performance plot saved as 'fnn_performance.png'")

# %% [7. Interview Scenario: Neural Networks]
# Discuss neural networks for ML.
print("\nInterview Scenario: Neural Networks")
print("Q: How do you train a neural network in Keras?")
print("A: Build with Sequential, compile with model.compile, train with model.fit.")
print("Key: Use callbacks to optimize training.")
print("Example: model.fit(X, y, epochs=5, callbacks=[EarlyStopping()])")