import tensorflow as tf


def relu(X):
    """ReLU activation function."""
    return tf.maximum(X, 0)


def sigmoid(X):
    """Sigmoid activation function."""
    return 1 / (1 + tf.exp(-X))
