import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# %% [1. Introduction to Customization]
# Learn how to customize Keras models for ML.
# Covers custom layers, loss functions, Functional API, and debugging.

print("TensorFlow/Keras version:", tf.__version__)

# %% [2. Custom Layers]
# Define a custom layer.
class CustomDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super(CustomDense, self).__init__()
        self.units = units
        self.activation = keras.activations.get(activation)
    
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
    
    def call(self, inputs):
        output = tf.matmul(inputs, self.w) + self.b
        if self.activation is not None:
            output = self.activation(output)
        return output

model_custom = keras.Sequential([
    CustomDense(64, activation='relu', input_shape=(10,)),
    CustomDense(32, activation='relu'),
    CustomDense(1, activation='sigmoid')
])
model_custom.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nModel with custom layer created.")

# %% [3. Custom Loss Functions]
# Define a custom loss function.
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred)) + 0.1 * tf.reduce_mean(tf.abs(y_pred))

np.random.seed(42)
X_train = np.random.rand(1000, 10).astype(np.float32)
y_train = (np.sum(X_train, axis=1) > 5).astype(np.float32)
model_loss = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])
model_loss.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
history_loss = model_loss.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
print("Model with custom loss trained.")

# %% [4. Functional API for Complex Models]
# Build a complex model with Functional API.
inputs = keras.Input(shape=(10,))
x = keras.layers.Dense(64, activation='relu')(inputs)
x1 = keras.layers.Dense(32, activation='relu')(x)
x2 = keras.layers.Dense(32, activation='relu')(x)
outputs = keras.layers.Concatenate()([x1, x2])
outputs = keras.layers.Dense(1, activation='sigmoid')(outputs)
model_func = keras.Model(inputs, outputs)
model_func.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_func = model_func.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
print("Functional API model trained.")

# %% [5. Debugging Model Performance]
# Analyze model performance with validation data.
X_val = np.random.rand(200, 10).astype(np.float32)
y_val = (np.sum(X_val, axis=1) > 5).astype(np.float32)
model_debug = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])
model_debug.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_debug = model_debug.fit(X_train, y_train, epochs=10, batch_size=32,
                                validation_data=(X_val, y_val), verbose=0)
plt.plot(history_debug.history['loss'], label='Training Loss')
plt.plot(history_debug.history['val_loss'], label='Validation Loss')
plt.title('Model Debugging: Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('debug_loss.png')
plt.close()
print("Debug loss plot saved as 'debug_loss.png'")

# %% [6. Interview Scenario: Customization]
# Discuss customization for ML.
print("\nInterview Scenario: Customization")
print("Q: How do you create a custom layer in Keras?")
print("A: Subclass keras.layers.Layer, define build and call methods.")
print("Key: Custom layers add flexibility for unique architectures.")
print("Example: class CustomLayer(keras.layers.Layer): def call(self, inputs): ...")