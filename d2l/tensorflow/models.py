import tensorflow as tf

from d2l.tensorflow.activations import relu
from d2l.tensorflow.metrics import Accumulator
from d2l.tensorflow.plot import Animator
from d2l.tensorflow.regularization import dropout
from d2l.tensorflow.data import seq_data_iter_consecutive, seq_data_iter_random
import time
import math
from tqdm import tqdm


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


def to_onehot(X, size):
    ones = [tf.one_hot(x, size) for x in tf.transpose(X)]
    return tf.stack(ones)


def predict_rnn(
    prefix, num_chars, rnn, params, init_rnn_state,
    num_hiddens, vocab_size, idx_to_char, char_to_idx
):
    state = init_rnn_state(1, num_hiddens)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # The output of the previous time step is taken
        # as the input of the current time step.
        X = to_onehot([output[-1]], vocab_size)
        # Calculate the output and update the hidden state.
        (Y, state) = rnn([X], state, params)
        # The input to the next time step is the character in 3
        # the prefix or the current best predicted character.
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            # This is maximum likelihood decoding, not sampling
            output.append(int(tf.argmax(Y[0], axis=1)))
    return ''.join([idx_to_char[i] for i in output])


def grad_clipping(gradients, theta):
    gradients = [tf.Variable(grad) for grad in gradients]
    norm = tf.Variable(0.)
    for grad in gradients:
        norm.assign_add(tf.reduce_sum(grad ** 2))
        norm.assign(tf.sqrt(norm))
    if norm > theta:
        for grad in gradients:
            grad.assign_add(grad * (theta / norm))
    return gradients

def train_and_predict_rnn(rnn, loss, optimizer, get_params, init_rnn_state, num_hiddens,
                          vocab_size, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = seq_data_iter_random
    else:
        data_iter_fn = seq_data_iter_consecutive
    params = get_params()
    
    for epoch in tqdm(range(num_epochs)):
        if not is_random_iter:  
            # If adjacent sampling is used, the hidden state is initialized 
            # at the beginning of the epoch.
            state = init_rnn_state(batch_size, num_hiddens)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps)
        for X, Y in data_iter:
            if is_random_iter:  
                # If random sampling is used, the hidden state is initialized 
                # before each mini-batch update.
                state = init_rnn_state(batch_size, num_hiddens)
            # else:  
            #     # Otherwise, the detach function needs to be used to separate 
            #     # the hidden state from the computational graph to avoid 
            #     # backpropagation beyond the current sample.
            #     for s in state:
            #         s.detach()
            with tf.GradientTape() as t:
                inputs = to_onehot(X, vocab_size)
                # outputs is num_steps terms of shape (batch_size, vocab_size)
                (outputs, state) = rnn(inputs, state, params)
                # after stitching it is (num_steps * batch_size, vocab_size).
                outputs = tf.concat(outputs, axis=0)
                # The shape of Y is (batch_size, num_steps), and then becomes 
                # a vector with a length of batch * num_steps after 
                # transposition. This gives it a one-to-one correspondence 
                # with output rows.
                y = tf.reshape(tf.transpose(Y), (-1,))
                # Average classification error via cross entropy loss.
                l = tf.reduce_mean(loss(y, outputs))
            gradients = t.gradient(l, params)
            gradients = grad_clipping(gradients, clipping_theta)  # Clip the gradient.
            optimizer(params, gradients, lr, 1)  
            # Since the error is the mean, no need to average gradients here.
            l_sum += l * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', 
                    predict_rnn(
                        prefix, pred_len, rnn, params, init_rnn_state,
                        num_hiddens, vocab_size, idx_to_char, char_to_idx
                    )
                )

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



