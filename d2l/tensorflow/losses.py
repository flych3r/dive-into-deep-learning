import tensorflow as tf


def squared_loss(y, y_hat):
    """Squared loss function."""
    y_hat = tf.reshape(y_hat, y.shape)
    return (y - y_hat) ** 2 / 2


def softmax(X):
    """Softmax implementation."""
    X_exp = tf.math.exp(X)
    partition = tf.reduce_sum(X_exp, axis=1, keepdims=True)
    return X_exp / partition


def cross_entropy(y, y_hat):
    """Cross entropy loss function."""
    return -tf.math.log(
        tf.gather_nd(y_hat, tf.reshape(y, (-1, 1)), batch_dims=1)
    )


def softmax_cross_entropy(y, y_hat):
    """Softmax cross entropy loss function."""
    loss = cross_entropy(y, softmax(y_hat))

    filter_ = ~tf.math.is_finite(loss)
    replace_ = tf.zeros_like(loss)

    return tf.where(filter_, replace_, loss)
