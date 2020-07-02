import tensorflow as tf


def get_activation(activation):
    if activation == 'relu':
        return relu
    if activation == 'sigmoid':
        return sigmoid
    return no_activation


def relu(X):
    """ReLU activation function."""
    return tf.maximum(X, 0)


def sigmoid(X):
    """Sigmoid activation function."""
    return 1 / (1 + tf.exp(-X))


def no_activation(X):
    return X
