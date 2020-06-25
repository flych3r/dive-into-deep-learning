import tensorflow as tf


def l2_penalty(W):
    """L2 Penalty function."""
    return tf.reduce_sum(W * W) / 2


def dropout(X, drop_rate):
    """Dropout implementation."""
    assert 0 <= drop_rate <= 1, 'dropout rate must be a probability'
    if drop_rate == 1:
        return tf.zeros_like(X)
    mask = tf.random.uniform(shape=X.shape, minval=0, maxval=1) > drop_rate
    return (X * tf.cast(mask, X.dtype)) / (1 - drop_rate)
