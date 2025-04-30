import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
try:
    from keras_tuner import RandomSearch
    import tensorflow_addons as tfa
except ImportError:
    RandomSearch, tfa = None, None

# %% [1. Introduction to Custom Extensions]
# Learn how to extend Keras with custom functionality.
# Covers custom metrics, callbacks, Keras Tuner, and TensorFlow Addons.

print("TensorFlow/Keras version:", tf.__version__)

# %% [2. Custom Metrics]
# Define a custom F1 score metric.
class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = keras.metrics.Precision()
        self.recall = keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + keras.backend.epsilon()))
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
model_metric = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])
model_metric.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', F1Score()])
history_metric = model_metric.fit(X_train, y_train, epochs=3, batch_size=64,
                                  validation_data=(X_test, y_test), verbose=0)
print("\nModel trained with custom F1 metric.")

# %% [3. Custom Callbacks]
# Define a custom callback to log epoch time.
class TimeLogger(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.times = []
    
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = tf.timestamp()
    
    def on_epoch_end(self, epoch, logs=None):
        self.times.append(tf.timestamp() - self.start_time)
        print(f"Epoch {epoch+1} time: {self.times[-1]:.2f} seconds")

model_callback = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])
model_callback.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
time_logger = TimeLogger()
history_callback = model_callback.fit(X_train, y_train, epochs=3, batch_size=64,
                                      validation_data=(X_test, y_test), callbacks=[time_logger], verbose=0)
print("Model trained with custom TimeLogger callback.")

# %% [4. Keras Tuner for Hyperparameter Optimization]
# Use Keras Tuner for hyperparameter search.
if RandomSearch:
    def build_model(hp):
        model = keras.Sequential([
            keras.layers.Conv2D(hp.Int('filters', 16, 64, step=16), (3, 3), activation='relu', input_shape=(32, 32, 3)),
            keras.layers.Flatten(),
            keras.layers.Dense(hp.Int('units', 32, 128, step=32), activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=3, executions_per_trial=1)
    tuner.search(X_train, y_train, epochs=3, validation_data=(X_test, y_test), verbose=0)
    best_model = tuner.get_best_models(num_models=1)[0]
    print("Keras Tuner hyperparameter search completed.")
else:
    print("Keras Tuner unavailable; skipping hyperparameter search.")

# %% [5. Practical ML Application]
# Visualize training performance with custom metric.
plt.plot(history_metric.history['f1_score'], label='Training F1 Score')
plt.plot(history_metric.history['val_f1_score'], label='Validation F1 Score')
plt.title('Custom Metric: F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.savefig('custom_f1_score.png')
plt.close()
print("Custom F1 score plot saved as 'custom_f1_score.png'")

# %% [6. Interview Scenario: Custom Extensions]
# Discuss custom extensions for ML.
print("\nInterview Scenario: Custom Extensions")
print("Q: How do you implement a custom metric in Keras?")
print("A: Subclass keras.metrics.Metric, define update_state and result.")
print("Key: Custom metrics track specialized performance.")
print("Example: class F1Score(keras.metrics.Metric): def result(self): ...")