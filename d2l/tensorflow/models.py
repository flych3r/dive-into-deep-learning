import tensorflow as tf

from d2l.tensorflow.activations import relu
from d2l.tensorflow.regularization import dropout


def initialize_parameters(num_inputs, num_outputs, mean=0, stdev=1):
    """Initializes weights and bias parameters."""
    W = tf.Variable(
        tf.random.normal(mean=mean, stddev=stdev, shape=(num_inputs, num_outputs)),
        name='weights'
    )
    b = tf.Variable(
        tf.zeros(shape=(num_outputs,)),
        name='bias'
    )

    return W, b


def linreg(X, W, b):
    """Linear regression implementation."""
    return tf.tensordot(X, W, axes=1) + b


def perceptron(W, b, X, y):
    """Perceptron implementation."""
    if y * (tf.tensordot(W, X, axes=1) + b) <= 0:
        W.assign_add(y * X)
        b.assign_add([y])
        return 1
    return 0


def mlp(X, W, b, drop_rates, num_inputs, inference=False):
    """Multi Layer Perceptron implementation."""
    X = tf.reshape(X, (-1, num_inputs))

    for i in range(len(W) - 1):
        X = relu(tf.matmul(X, W[i]) + b[i])
        if not inference:
            X = dropout(X, drop_rates[i])

    return tf.matmul(X, W[i + 1]) + b[i + 1]


# class Model:
#     def __init__(self, net, Ws, bs):
#         self.net = net
#         self.Ws = Ws
#         self.bs = bs

#     def train_epoch(self, data_iter, loss_function, learning_rate, batch_size):
#         """Training iteration."""
#         metric = Accumulator(3)  # train_loss_sum, train_acc_sum, num_examples

#         for X, y in data_iter:
#             with tf.GradientTape() as t:
#                 y_hat = self.net(X)
#                 loss = loss_function(y, y_hat)
#             dWs, dbs = t.gradient(loss, [self.Ws, self.bs])
#             for i in range(len(self.Ws)):
#                 self.Ws[i].assign_sub((learning_rate * dWs[i]) / batch_size)
#                 self.bs[i].assign_sub((learning_rate * dbs[i]) / batch_size)
#             metric.add(tf.reduce_sum(loss), accuracy(y, y_hat), y.shape[0])

#         # Return training loss and training accuracy
#         return metric[0] / metric[2], metric[1] / metric[2]

#     def train(self, train_iter, test_iter, loss_function, eval_function,
#               num_epochs, learning_rate, batch_size):
#         """Model training function."""
#         animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
#                             ylim=[0.3, 0.9],
#                             legend=['train loss', 'train acc', 'test acc'])
#         for epoch in range(num_epochs):
#             train_metrics = self.train_epoch(
#                 train_iter, loss_function, batch_size, learning_rate
#             )
#             test_eval = eval_function(self.net, test_iter)
#             animator.add(epoch + 1, train_metrics + (test_eval,))
