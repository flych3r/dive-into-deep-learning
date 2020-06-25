import tensorflow as tf


def accuracy(y, y_hat):
    """Accuracy score function."""
    hits = tf.math.equal(tf.math.argmax(y_hat, axis=1), y)
    return tf.math.reduce_sum(
        tf.cast(hits, dtype=tf.int32)
    )


def evaluate_accuracy(model, data_iter):
    metric = Accumulator(2)  # num_corrected_examples, num_examples
    for X, y in data_iter:
        metric.add(accuracy(y, model(X, inference=True)), y.shape[0])
    return metric[0] / metric[1]


class Accumulator:
    """Sum a list of numbers over time."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
