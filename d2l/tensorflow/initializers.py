import tensorflow as tf


def initialize_parameters(
    num_inputs, num_outputs, bias_units=None,
    method='gaussian', magnitude=0, scale=0.1,
    trainable=True
):
    """Initializes weights and bias parameters."""
    if not bias_units:
        bias_units = num_outputs

    W = initialize_weights((num_inputs, num_outputs), trainable, method, magnitude, scale)

    b = initialize_bias(bias_units, trainable)

    return W, b


def initialize_weights(
    shape, trainable=True, method='gaussian', magnitude=0, scale=0.1
):
    if method == 'gaussian':
        method = gaussian_initialization
    if method == 'xavier':
        method = xavier_initialization

    if trainable:
        tensor = tf.Variable
    else:
        tensor = tf.constant

    return tensor(
        method(shape, magnitude, scale),
        name='weights'
    )


def initialize_bias(bias_units, trainable=True):
    if trainable:
        tensor = tf.Variable
    else:
        tensor = tf.constant

    return tensor(
        tf.zeros(shape=(bias_units,)),
        name='bias'
    )


def gaussian_initialization(shape, magnitude=None, scale=None):
    if magnitude is None:
        magnitude = 0
    if scale is None:
        scale = 0.1

    return tf.random.normal(mean=magnitude, stddev=scale, shape=shape)


def xavier_initialization(shape, magnitude=None, scale=None):
    if magnitude is None:
        magnitude = 6
    if scale is None:
        scale = 0.5

    interval = (magnitude / sum(shape)) ** scale
    return tf.random.uniform(minval=-interval, maxval=interval, shape=shape)
