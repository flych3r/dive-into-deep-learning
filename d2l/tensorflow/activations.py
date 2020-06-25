import tensorflow as tf


def relu(X):
    """ReLU activation function."""
    return tf.maximum(X, 0)
