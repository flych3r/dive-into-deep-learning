import tensorflow as tf


def initialize_parameters(num_inputs, num_outputs, method=None, scale=None, sigma=None):
    """Initializes weights and bias parameters."""
    if method == 'gaussian':
        method = gaussian_initialization
    if method == 'xavier':
        method = xavier_initialization

    W = tf.Variable(
        method((num_inputs, num_outputs), scale, sigma),
        name='weights'
    )
    b = tf.Variable(
        tf.zeros(shape=(num_outputs,)),
        name='bias'
    )

    return W, b


def gaussian_initialization(shape, scale=None, sigma=None):
    if scale is None:
        scale = 0
    if sigma is None:
        sigma = 0.1

    return tf.random.normal(mean=scale, stddev=sigma, shape=shape)


def xavier_initialization(shape, scale=None, sigma=None):
    if scale is None:
        scale = 6
    if sigma is None:
        sigma = 0.5

    interval = (scale / sum(shape)) ** sigma
    return tf.random.uniform(minval=-interval, maxval=interval, shape=shape)
