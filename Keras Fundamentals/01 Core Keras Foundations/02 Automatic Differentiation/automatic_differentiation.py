import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# %% [1. Introduction to Automatic Differentiation]
# Learn how Keras uses automatic differentiation for gradient computation.
# Covers tf.GradientTape, custom gradients, and optimizer application.

print("TensorFlow/Keras version:", tf.__version__)

# %% [2. Gradient Computation with tf.GradientTape]
# Compute gradients for a simple function.
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2
dy_dx = tape.gradient(y, x)
print("\nGradient of y = x^2 at x = 3:", dy_dx.numpy())

# %% [3. Custom Gradient Workflow]
# Train a simple linear model with custom gradients.
np.random.seed(42)
X = np.random.rand(100, 1).astype(np.float32)
y = 2 * X + 1 + np.random.normal(0, 0.1, (100, 1)).astype(np.float32)
w = tf.Variable(0.0)
b = tf.Variable(0.0)
def loss_fn(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

learning_rate = 0.01
for _ in range(100):
    with tf.GradientTape() as tape:
        y_pred = w * X + b
        loss = loss_fn(y_pred, y)
    dw, db = tape.gradient(loss, [w, b])
    w.assign_sub(learning_rate * dw)
    b.assign_sub(learning_rate * db)
print("Learned w:", w.numpy(), "Learned b:", b.numpy())

# %% [4. Optimizer Application]
# Use an optimizer with tf.GradientTape.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
w = tf.Variable(0.0)
b = tf.Variable(0.0)
for _ in range(100):
    with tf.GradientTape() as tape:
        y_pred = w * X + b
        loss = loss_fn(y_pred, y)
    gradients = tape.gradient(loss, [w, b])
    optimizer.apply_gradients(zip(gradients, [w, b]))
print("Optimized w:", w.numpy(), "Optimized b:", b.numpy())

# %% [5. Practical ML Application]
# Train a neural network with custom gradients.
np.random.seed(42)
X_train = np.random.rand(1000, 10).astype(np.float32)
y_train = np.random.randint(0, 2, 1000).astype(np.float32)
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])
optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss_fn = keras.losses.BinaryCrossentropy()
losses = []
for epoch in range(5):
    with tf.GradientTape() as tape:
        y_pred = model(X_train, training=True)
        loss = loss_fn(y_train, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    losses.append(loss.numpy())
plt.plot(losses, label='Loss')
plt.title('Custom Gradient Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('custom_gradient_loss.png')
plt.close()
print("Custom gradient loss plot saved as 'custom_gradient_loss.png'")

# %% [6. Interview Scenario: Automatic Differentiation]
# Discuss automatic differentiation for ML.
print("\nInterview Scenario: Automatic Differentiation")
print("Q: How does tf.GradientTape work in Keras?")
print("A: It records operations to compute gradients for optimization.")
print("Key: Enables custom training loops for flexibility.")
print("Example: with tf.GradientTape() as tape: loss = fn(x)")