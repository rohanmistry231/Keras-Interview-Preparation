# 🏗️ Core Keras Foundations (`keras`)

## 📖 Introduction
Keras, built on TensorFlow, is a high-level API for designing, training, and evaluating neural networks in AI and machine learning (ML). This section introduces the **Core Keras Foundations**, covering **Keras Basics**, **Automatic Differentiation**, **Neural Networks**, **Datasets and Data Loading**, and **Training Pipeline**. With practical examples and interview insights, it equips beginners to build and train neural networks, complementing Pandas, NumPy, and Matplotlib skills.

## 🎯 Learning Objectives
- Build neural networks using Keras APIs (`Sequential`, `Functional`, `Model Subclassing`).
- Understand gradient computation with `tf.GradientTape` for custom training.
- Train and evaluate feedforward neural networks (FNNs) with callbacks.
- Load and preprocess datasets using `keras.datasets` and `tf.data`.
- Set up efficient training pipelines with saving, GPU support, and monitoring.

## 🔑 Key Concepts
- **Keras Basics**:
  - Model Creation (`Sequential`, `Functional API`, `Model Subclassing`).
  - Layers: Dense, Convolutional, Pooling, Normalization.
  - Activations: ReLU, Sigmoid, Softmax.
  - Loss Functions: MSE, Categorical Crossentropy.
  - Optimizers: SGD, Adam, RMSprop.
- **Automatic Differentiation**:
  - Gradient Computation with `tf.GradientTape`.
  - Custom Gradient Workflows.
  - Optimizer Application (`optimizer.minimize`).
- **Neural Networks**:
  - Building Feedforward Neural Networks (FNNs).
  - Compiling Models (`model.compile`).
  - Training Models (`model.fit`, `model.evaluate`).
  - Callbacks: EarlyStopping, ModelCheckpoint, LearningRateScheduler.
- **Datasets and Data Loading**:
  - Built-in Datasets (`keras.datasets`).
  - TensorFlow Datasets (`tfds.load`).
  - Data Pipeline (`tf.data.Dataset`, map, batch, shuffle).
  - Preprocessing (`keras.preprocessing`).
- **Training Pipeline**:
  - Training/Evaluation Loops.
  - Model Saving/Loading (`model.save`, `model.load`).
  - GPU Training (`tf.device`).
  - Monitoring with TensorBoard.

## 📝 Example Walkthroughs
The following Python files demonstrate each subsection:

1. **`keras_basics.py`**:
   - Builds models with `Sequential`, `Functional API`, and `Model Subclassing`.
   - Uses layers (Dense, Conv1D, BatchNormalization) and activations (ReLU, Softmax).
   - Compiles models with different losses (MSE, Crossentropy) and optimizers (Adam, SGD).
   - Trains a model on synthetic data and visualizes performance.

   Example code:
   ```python
   model = keras.Sequential([keras.layers.Dense(64, activation='relu')])
   model.compile(optimizer='adam', loss='categorical_crossentropy')
   ```

2. **`automatic_differentiation.py`**:
   - Computes gradients with `tf.GradientTape` for a simple function.
   - Trains a linear model with custom gradients.
   - Uses an optimizer (`Adam`) with `GradientTape`.
   - Trains a neural network with custom gradients and plots loss.

   Example code:
   ```python
   with tf.GradientTape() as tape:
       y_pred = w * X + b
       loss = tf.reduce_mean(tf.square(y_pred - y))
   ```

3. **`neural_networks.py`**:
   - Builds a feedforward neural network (FNN) for MNIST classification.
   - Compiles with `model.compile` and trains with `model.fit`.
   - Uses callbacks (`EarlyStopping`, `ModelCheckpoint`).
   - Visualizes training loss and accuracy.

   Example code:
   ```python
   model.fit(X_train, y_train, epochs=5, callbacks=[keras.callbacks.EarlyStopping()])
   ```

4. **`datasets_data_loading.py`**:
   - Loads MNIST with `keras.datasets`.
   - Explores TensorFlow Datasets (`tfds.load`, commented if unavailable).
   - Creates a `tf.data.Dataset` pipeline with shuffle and batch.
   - Applies preprocessing and trains a model with the pipeline.

   Example code:
   ```python
   dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(10000).batch(128)
   ```

5. **`training_pipeline.py`**:
   - Sets up a training loop for MNIST with `model.fit` and `model.evaluate`.
   - Saves and loads models with `model.save` and `model.load`.
   - Trains on GPU with `tf.device`.
   - Monitors training with TensorBoard and visualizes performance.

   Example code:
   ```python
   model.fit(X_train, y_train, callbacks=[keras.callbacks.TensorBoard()])
   model.save('model.h5')
   ```

## 🛠️ Practical Tasks
1. **Keras Basics**:
   - Build a `Sequential` model with Dense layers and ReLU activation.
   - Create a model using the Functional API with multiple inputs.
   - Compile a model with Adam optimizer and categorical crossentropy loss.
2. **Automatic Differentiation**:
   - Compute gradients for a quadratic function using `tf.GradientTape`.
   - Train a linear model with custom gradients.
   - Use `GradientTape` to train a small neural network.
3. **Neural Networks**:
   - Build and train an FNN on MNIST with `model.fit`.
   - Add `EarlyStopping` and `ModelCheckpoint` callbacks.
   - Plot training and validation loss curves.
4. **Datasets and Data Loading**:
   - Load MNIST with `keras.datasets` and preprocess images.
   - Create a `tf.data.Dataset` pipeline with batching and shuffling.
   - Train a model using the data pipeline.
5. **Training Pipeline**:
   - Train a model on MNIST with validation data.
   - Save and load the trained model.
   - Set up TensorBoard to monitor training metrics.

## 💡 Interview Tips
- **Common Questions**:
  - How do you build a neural network in Keras?
  - What is `tf.GradientTape` used for?
  - How do you preprocess data for Keras models?
  - How do you save and load a Keras model?
- **Tips**:
  - Explain the difference between `Sequential` and `Functional API`.
  - Highlight `GradientTape` for custom training loops.
  - Be ready to code a simple FNN or data pipeline.
- **Coding Tasks**:
  - Build and train a small FNN on a dataset.
  - Compute gradients with `tf.GradientTape` for a loss function.
  - Create a `tf.data.Dataset` pipeline for training.

## 📚 Resources
- [Keras Documentation](https://keras.io/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Kaggle: Intro to Deep Learning](https://www.kaggle.com/learn/intro-to-deep-learning)
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)