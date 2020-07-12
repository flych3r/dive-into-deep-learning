def sgd(params, gradients, learning_rate, batch_size):
    """Stochastic Gradient Descent update implementation."""
    for i in range(len(params)):
        params[i].assign_sub((learning_rate * gradients[i]) / batch_size)
