import tensorflow as tf


def l2_penalty(lambd, W):
    """L2 Penalty function."""
    return lambd * (tf.reduce_sum(W * W) / 2)


def dropout(X, drop_rate):
    """Dropout implementation."""
    assert 0 <= drop_rate <= 1, 'dropout rate must be a probability'
    if drop_rate == 1:
        return tf.zeros_like(X)
    mask = tf.random.uniform(shape=X.shape, minval=0, maxval=1) > drop_rate
    return (X * tf.cast(mask, X.dtype)) / (1 - drop_rate)


def batch_norm(
    X, gamma, beta, moving_mean, moving_var, eps, momentum, inference=False
):
    # determine whether the current mode is training mode or prediction mode.
    if inference:
        # use the moving average
        X_hat = (X - moving_mean) / tf.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:  # fully connected layer
            mean = tf.reduce_mean(X, axis=0)
            var = tf.reduce_mean((X - mean) ** 2, axis=0)
        else:  # convolution, hence per layer
            mean = tf.reduce_mean(X, axis=(0, 2, 3), keepdims=True)
            var = tf.reduce_mean((X - mean) ** 2, axis=(0, 2, 3), keepdims=True)
            # In training mode, the current mean and variance
            # are used for the standardization.
            X_hat = (X - mean) / tf.sqrt(var + eps)
            # Update the mean and variance of the moving average.
            moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
            moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift.
    return Y, moving_mean, moving_var
