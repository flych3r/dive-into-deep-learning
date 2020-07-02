import tensorflow as tf

from d2l.tensorflow.activations import get_activation
from d2l.tensorflow.initializers import initialize_parameters, initialize_weights


class BaseLayer:
    _identifier = -1
    __type__ = 'base'

    def __init__(self):
        self.weights = None
        self.bias = None

    def __call__(self):
        pass


class Dense(BaseLayer):
    def __init__(
        self, n_inputs, n_outputs, activation='relu',
        initialization='gaussian', scale=None, sigma=None
    ):
        Dense.__type__ = 'compute'
        Dense._identifier += 1
        self.__name__ = '{}_{}'.format(Dense.__name__, Dense._identifier).lower()

        self.weights, self.bias = initialize_parameters(
            n_inputs, n_outputs, None, initialization, scale, sigma
        )
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = get_activation(activation)

    def __call__(self, X):
        return self.activation(tf.matmul(X, self.weights) + self.bias)

    def __repr__(self):
        return str({
            'name': self.__name__,
            'type': self.__type__,
            'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
            'activation': self.activation.__name__,
            'weights': self.weights.numpy(),
            'bias': self.bias.numpy()
        })


class Conv2D(BaseLayer):
    def __init__(
        self, filters, kernel_shape, channels, activation=None,
        initialization='gaussian', magnitude=None, scale=None
    ):
        Conv2D.__type__ = 'compute'
        Conv2D._identifier += 1
        self.__name__ = '{}_{}'.format(Conv2D.__name__, Conv2D._identifier).lower()

        self.filter_shape = (channels,) + kernel_shape
        self.filters = tf.Variable(tf.stack([
            initialize_weights(
                self.filter_shape, method=initialization,
                magnitude=magnitude, scale=scale
            ) for _ in range(filters)
        ]))
        self.filter_shape = kernel_shape + (channels, filters)
        self.kernel_shape = kernel_shape
        self.channels = channels
        self.activation = get_activation(activation)

    def __call__(self, X):
        msg = 'Input must be 4 dimentional (batch_size, height, width, channels)'
        assert len(X.shape) == 4, msg
        return tf.squeeze(
            tf.nn.conv2d(
                tf.reshape(X, X.shape),
                tf.reshape(self.filters, self.filter_shape),
                strides=1,
                padding='VALID'
            )
        )

    def __repr__(self):
        return str({
            'name': self.__name__,
            'type': self.__type__,
            'kernel_shape': self.kernel_shape,
            'channels': self.channels,
            'activation': self.activation.__name__,
            'filters': self.filters.numpy()
        })
