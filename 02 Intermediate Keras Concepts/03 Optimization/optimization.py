import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
try:
    from keras import mixed_precision
except ImportError:
    mixed_precision = None

# %% [1. Introduction to Optimization]
# Learn how to optimize Keras models for better performance.
# Covers hyperparameter tuning, regularization, and mixed precision training.

print("TensorFlow/Keras version:", tf.__version__)

# %% [2. Hyperparameter Tuning]
# Experiment with learning rates and batch sizes.
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
learning_rates = [0.001, 0.01]
batch_sizes = [32, 128]
histories = []
for lr in learning_rates:
    for bs in batch_sizes:
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=3, batch_size=bs,
                            validation_data=(X_test, y_test), verbose=0)
        histories.append((lr, bs, history))
print("\nHyperparameter tuning completed.")

# %% [3. Regularization]
# Apply Dropout and L2 regularization.
model_reg = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,),
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])
model_reg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_reg = model_reg.fit(X_train, y_train, epochs=5, batch_size=128,
                            validation_data=(X_test, y_test), verbose=0)
print("Model with regularization trained.")

# %% [4. Mixed Precision Training]
# Train with mixed precision for efficiency.
if mixed_precision:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    model_mp = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax', dtype='float32')
    ])
    model_mp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history_mp = model_mp.fit(X_train, y_train, epochs=5, batch_size=128,
                              validation_data=(X_test, y_test), verbose=0)
    mixed_precision.set_global_policy('float32')
    print("Mixed precision training completed.")
else:
    print("Mixed precision unavailable; skipping.")

# %% [5. Practical ML Application]
# Visualize hyperparameter tuning results.
plt.figure(figsize=(12, 4))
for lr, bs, history in histories:
    plt.plot(history.history['val_accuracy'], label=f'LR={lr}, BS={bs}')
plt.title('Hyperparameter Tuning: Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.savefig('hyperparameter_tuning.png')
plt.close()
plt.plot(history_reg.history['loss'], label='Training Loss')
plt.plot(history_reg.history['val_loss'], label='Validation Loss')
plt.title('Regularization: Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('regularization_loss.png')
plt.close()
print("Hyperparameter tuning and regularization plots saved.")

# %% [6. Interview Scenario: Optimization]
# Discuss optimization for ML.
print("\nInterview Scenario: Optimization")
print("Q: How do you prevent overfitting in a Keras model?")
print("A: Use Dropout, L2 regularization, or early stopping.")
print("Key: Regularization improves generalization.")
print("Example: keras.layers.Dropout(0.5), kernel_regularizer=keras.regularizers.l2(0.01)")