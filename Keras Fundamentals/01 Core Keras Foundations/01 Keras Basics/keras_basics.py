import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# %% [1. Introduction to Keras Basics]
# Learn the fundamentals of Keras for building ML models.
# Covers model creation, layers, activations, loss functions, and optimizers.

print("TensorFlow/Keras version:", tf.__version__)

# %% [2. Model Creation: Sequential API]
# Build a simple Sequential model for classification.
model_seq = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])
model_seq.summary()
print("\nSequential model created.")

# %% [3. Model Creation: Functional API]
# Build a model with Functional API for flexibility.
inputs = keras.Input(shape=(10,))
x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(32, activation='relu')(x)
outputs = keras.layers.Dense(3, activation='softmax')(x)
model_func = keras.Model(inputs, outputs)
model_func.summary()
print("Functional API model created.")

# %% [4. Model Creation: Subclassing]
# Define a custom model by subclassing keras.Model.
class CustomModel(keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense1 = keras.layers.Dense(64, activation='relu')
        self.dense2 = keras.layers.Dense(32, activation='relu')
        self.dense3 = keras.layers.Dense(3, activation='softmax')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

model_sub = CustomModel()
model_sub.build(input_shape=(None, 10))
model_sub.summary()
print("Subclassed model created.")

# %% [5. Layers and Activations]
# Explore common layers and activations.
np.random.seed(42)
X = np.random.rand(100, 10)
model_layers = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Conv1D(32, kernel_size=3, activation='sigmoid', padding='same'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(),
    keras.layers.Dense(3, activation='softmax')
])
model_layers.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model with various layers created.")

# %% [6. Loss Functions and Optimizers]
# Compile a model with different loss functions and optimizers.
model_opt = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])
model_opt.compile(optimizer='sgd', loss='mse', metrics=['mae'])
model_opt.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_opt.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled with different optimizers/losses.")

# %% [7. Practical ML Application]
# Train a simple model on synthetic data.
np.random.seed(42)
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 3, 1000)
y_train_cat = keras.utils.to_categorical(y_train, 3)
model_ml = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])
model_ml.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_ml.fit(X_train, y_train_cat, epochs=5, batch_size=32, verbose=0)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.savefig('training_curves.png')
plt.close()
print("Training curves saved as 'training_curves.png'")

# %% [8. Interview Scenario: Keras Basics]
# Discuss Keras basics for ML.
print("\nInterview Scenario: Keras Basics")
print("Q: How do you build a neural network in Keras?")
print("A: Use Sequential API for simple models, Functional API for complex ones, or subclass keras.Model.")
print("Key: Choose API based on model complexity.")
print("Example: model = keras.Sequential([keras.layers.Dense(64, activation='relu')])")