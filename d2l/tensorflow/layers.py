
from d2l.tensorflow.activations import get_activation
from d2l.tensorflow.initializers import initialize_parameters


class BaseLayer:
    _identifier = -1
    __type__ = 'base'

    def __init__(self):
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
            n_inputs, n_outputs, initialization, scale, sigma
        )
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = get_activation(activation)

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
