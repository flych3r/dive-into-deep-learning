def sgd(W, b, dW, db, learning_rate, batch_size):
    """Stochastic Gradient Descent update implementation."""
    if not isinstance(W, list):
        W, b, dW, db = [W], [b], [dW], [db]
    for i in range(len(W)):
        W[i].assign_sub((learning_rate * dW[i]) / batch_size)
        b[i].assign_sub((learning_rate * db[i]) / batch_size)
