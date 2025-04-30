import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
except ImportError:
    confusion_matrix, sns = None, None

# %% [1. Introduction to Model Architectures]
# Learn intermediate Keras model architectures for ML.
# Covers CNNs, RNNs (LSTMs, GRUs), and transfer learning.

print("TensorFlow/Keras version:", tf.__version__)

# %% [2. Convolutional Neural Networks (CNNs)]
# Build and train a CNN on CIFAR-10.
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
cnn_model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_history = cnn_model.fit(X_train, y_train, epochs=5, batch_size=64,
                            validation_data=(X_test, y_test), verbose=0)
print("\nCNN trained on CIFAR-10.")

# %% [3. Recurrent Neural Networks (RNNs, LSTMs)]
# Build and train an LSTM on synthetic sequence data.
np.random.seed(42)
seq_length = 10
X_seq = np.random.rand(1000, seq_length, 1).astype(np.float32)
y_seq = np.sum(X_seq, axis=1).reshape(-1, 1) > seq_length / 2
y_seq = keras.utils.to_categorical(y_seq, 2)
lstm_model = keras.Sequential([
    keras.layers.LSTM(32, input_shape=(seq_length, 1), return_sequences=False),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm_history = lstm_model.fit(X_seq, y_seq, epochs=5, batch_size=32, verbose=0)
print("LSTM trained on synthetic sequence data.")

# %% [4. Transfer Learning]
# Use a pretrained VGG16 model for CIFAR-10.
base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model.trainable = False
tl_model = keras.Sequential([
    base_model,
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
tl_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
tl_history = tl_model.fit(X_train, y_train, epochs=3, batch_size=64,
                          validation_data=(X_test, y_test), verbose=0)
print("Transfer learning model trained with VGG16.")

# %% [5. Practical ML Application]
# Visualize CNN training performance and confusion matrix.
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['loss'], label='Training Loss')
plt.plot(cnn_history.history['val_loss'], label='Validation Loss')
plt.title('CNN Training Performance')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['accuracy'], label='Training Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Training Performance')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('cnn_performance.png')
plt.close()
if confusion_matrix and sns:
    y_pred = np.argmax(cnn_model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('CNN Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('cnn_confusion_matrix.png')
    plt.close()
    print("CNN performance and confusion matrix saved.")
else:
    print("CNN performance saved; confusion matrix skipped (missing sklearn/seaborn).")

# %% [6. Interview Scenario: Model Architectures]
# Discuss model architectures for ML.
print("\nInterview Scenario: Model Architectures")
print("Q: How do you build a CNN in Keras for image classification?")
print("A: Use Conv2D, MaxPooling2D, Flatten, and Dense layers in Sequential.")
print("Key: CNNs are effective for spatial data like images.")
print("Example: model = keras.Sequential([keras.layers.Conv2D(32, (3, 3), activation='relu')])")