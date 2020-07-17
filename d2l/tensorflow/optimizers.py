import itertools


def sgd(params, gradients, learning_rate, batch_size):
    """Stochastic Gradient Descent update implementation."""
    if isinstance(params[0], list):
        params = [*itertools.chain(*params)]
        gradients = [*itertools.chain(*gradients)]
    else:
        params = [*itertools.chain(params)]
        gradients = [*itertools.chain(gradients)]
    for param, grad in zip(params, gradients):
        param.assign_sub((learning_rate * grad) / batch_size)
