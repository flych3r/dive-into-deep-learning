import tensorflow as tf

from d2l.tensorflow.activations import relu
from d2l.tensorflow.regularization import dropout
from d2l.tensorflow.plot import Animator
from d2l.tensorflow.metrics import Accumulator


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


class BaseModel:
    _identifier = -1

    def __init__(self):
        self.loss_function = None
        self.eval_metric = None
        self.eval_function = None
        self.optimizer = None
        self.compiled = False


class Sequential(BaseModel):
    def __init__(self):
        super(Sequential, self).__init__()
        Sequential._identifier += 1
        self.__name__ = '{}_{}'.format(
            Sequential.__name__, Sequential._identifier
        ).lower()
        self.layers = list()

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss_function, eval_metric, optimizer):
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.optimizer = optimizer
        self.compiled = True

    def net(self, X, inference=False):
        X = tf.reshape(X, (-1, self.layers[0].n_inputs))

        for layer in self.layers[:-1]:
            X = layer.activation(tf.matmul(X, layer.weights) + layer.bias)

        return tf.matmul(X, self.layers[-1].weights) + self.layers[-1].bias

    def fit(self, epochs, train_iter, val_iter, learning_rate, batch_size, animate=False):
        if not self.compiled:
            print('model not compiled')
            return

        animator = None
        if animate:
            animator = Animator(
                xlabel='epoch', xlim=[1, epochs], ylim=[0, 1],
                legend=['train loss', 'train eval', 'val loss', 'val eval'],
                title='Training loss and eval'
            )

        for epoch in range(epochs):

            metric_train = Accumulator(3)
            W = [layer.weights for layer in self.layers]
            b = [layer.bias for layer in self.layers]

            for X, y in train_iter:
                with tf.GradientTape() as t:
                    y_hat = self.net(X)
                    loss = self.loss_function(y, y_hat)
                dW, db = t.gradient(loss, [W, b])
                self.optimizer(W, b, dW, db, learning_rate, batch_size)
                metric_train.add(
                    tf.reduce_sum(loss), self.eval_metric(y, y_hat), y.shape[0]
                )

            train_metrics = (
                metric_train[0] / metric_train[2],
                metric_train[1] / metric_train[2]
            )
            val_metrics = self.predict(val_iter)
            if animator:
                animator.add(epoch + 1, train_metrics + val_metrics)
            else:
                print(
                    'epoch {0} => '
                    '[train loss: {1[0]}, train eval: {1[1]}]'
                    ' | '
                    '[val loss: {2[0]}, val eval: {2[1]}]'.format(
                        epoch + 1, train_metrics, val_metrics
                    )
                )

    def predict(self, test_iter):
        metric_test = Accumulator(3)

        for X, y in test_iter:
            y_hat = self.net(X, inference=True)
            loss = self.loss_function(y, y_hat)
            metric_test.add(tf.reduce_sum(loss), self.eval_metric(y, y_hat), y.shape[0])

        return metric_test[0] / metric_test[2], metric_test[1] / metric_test[2]

    def __repr__(self):
        if self.compiled:
            return str({
                'name': self.__name__,
                'layers': self.layers,
                'loss function': self.loss_function.__name__,
                'eval metric': self.eval_metric.__name__,
                'optimizer': self.optimizer.__name__
            })
        return str({
            'name': self.__name__,
            'layers': self.layers,
        })
