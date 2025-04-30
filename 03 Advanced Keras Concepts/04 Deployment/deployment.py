import tensorflow as tf
from tensorflow import keras
import numpy as np
try:
    import onnx
    import tf2onnx
    import tensorflow.lite as tflite
except ImportError:
    onnx, tf2onnx, tflite = None, None, None

# %% [1. Introduction to Deployment]
# Learn how to deploy Keras models for production.
# Covers model export, serving, and edge deployment.

print("TensorFlow/Keras version:", tf.__version__)

# %% [2. Model Export (SavedModel)]
# Train and export a model as SavedModel.
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test), verbose=0)
model.save('saved_model')
print("\nModel exported as SavedModel to 'saved_model'.")

# %% [3. Model Export (ONNX)]
# Export the model to ONNX format.
if onnx and tf2onnx:
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=[
        tf.TensorSpec((None, 32, 32, 3), tf.float32, name='input')
    ])
    with open('model.onnx', 'wb') as f:
        f.write(model_proto.SerializeToString())
    print("Model exported to ONNX as 'model.onnx'.")
else:
    print("ONNX or tf2onnx unavailable; skipping ONNX export.")

# %% [4. Serving with TensorFlow Serving]
# Instructions for TensorFlow Serving (commented as it requires server setup).
"""
# Save model in SavedModel format (already done).
# Install TensorFlow Serving: pip install tensorflow-serving-api
# Run server: tensorflow_model_server --rest_api_port=8501 --model_name=cifar_model --model_base_path=/path/to/saved_model
# Query model via REST API:
import requests
data = {'instances': X_test[:1].tolist()}
response = requests.post('http://localhost:8501/v1/models/cifar_model:predict', json=data)
print(response.json())
"""
print("TensorFlow Serving instructions provided (commented).")

# %% [5. Edge Deployment with TensorFlow Lite]
# Convert and save model for edge devices.
if tflite:
    converter = tflite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model converted to TensorFlow Lite as 'model.tflite'.")
else:
    print("TensorFlow Lite unavailable; skipping TFLite conversion.")

# %% [6. Practical ML Application]
# Test SavedModel inference and visualize predictions.
loaded_model = tf.keras.models.load_model('saved_model')
predictions = loaded_model.predict(X_test[:10], verbose=0)
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_test[i])
    plt.title(f"Pred: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.savefig('saved_model_predictions.png')
plt.close()
print("SavedModel predictions saved as 'saved_model_predictions.png'")

# %% [7. Interview Scenario: Deployment]
# Discuss model deployment for ML.
print("\nInterview Scenario: Deployment")
print("Q: How do you deploy a Keras model for edge devices?")
print("A: Convert to TensorFlow Lite using TFLiteConverter.")
print("Key: TFLite optimizes models for resource-constrained devices.")
print("Example: converter = tflite.TFLiteConverter.from_keras_model(model)")