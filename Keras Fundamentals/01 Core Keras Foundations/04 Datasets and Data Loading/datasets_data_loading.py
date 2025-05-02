import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
try:
    import tensorflow_datasets as tfds
except ImportError:
    tfds = None

# %% [1. Introduction to Datasets and Data Loading]
# Learn how to load and preprocess data for Keras models.
# Covers built-in datasets, TensorFlow Datasets, data pipelines, and preprocessing.

print("TensorFlow/Keras version:", tf.__version__)

# %% [2. Built-in Datasets]
# Load and preprocess MNIST dataset.
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
print("\nMNIST dataset loaded.")

# %% [3. TensorFlow Datasets]
# Load a dataset with tfds (fallback to MNIST if tfds unavailable).
if tfds:
    ds, info = tfds.load('mnist', split='train', with_info=True, as_supervised=True)
    print("TensorFlow Datasets MNIST loaded.")
else:
    print("TensorFlow Datasets unavailable; using keras.datasets.")

# %% [4. Data Pipeline with tf.data.Dataset]
# Create a data pipeline with batching and shuffling.
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
print("Data pipeline created.")

# %% [5. Preprocessing]
# Apply preprocessing to MNIST images.
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset_pre = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset_pre = train_dataset_pre.map(preprocess_image).batch(128)
print("Preprocessing applied.")

# %% [6. Practical ML Application]
# Train a model with the data pipeline.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_dataset, epochs=5, verbose=0)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.title('Model Training with Data Pipeline')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.savefig('data_pipeline_training.png')
plt.close()
print("Data pipeline training plot saved as 'data_pipeline_training.png'")

# %% [7. Interview Scenario: Datasets]
# Discuss datasets for ML.
print("\nInterview Scenario: Datasets")
print("Q: How do you load and preprocess data in Keras?")
print("A: Use keras.datasets or tf.data.Dataset with map, batch, shuffle.")
print("Key: Efficient pipelines improve training speed.")
print("Example: dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(128)")