import random

import tensorflow_datasets as tfds
from tensorflow import convert_to_tensor, gather, one_hot
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.image import resize
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical


def transform(example, sparse=True):
    """Transforms the dataset containing images."""
    return (
        example['image'] / 255,
        example['label'] if sparse else one_hot(example['label'], depth=10)
    )


def get_fashion_mnist_labels(labels):
    """Converts fashion mnist labels to string."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def batch_iter(features, labels, batch_size=256, shuffle=True):
    """Iterates thru a dataset."""
    num_examples = features.shape[0]
    indices = [*range(num_examples)]
    if shuffle:
        random.shuffle(indices)
    while True:
        for i in range(0, num_examples, batch_size):
            j = convert_to_tensor(indices[i: min(i + batch_size, num_examples)])
            yield gather(features, j), gather(labels, j)


def load_tfds_dataset(ds_name, batch_size=256, sparse=True):
    """Loads and transforms a tfds dataset."""
    ds = tfds.load(ds_name, shuffle_files=True)
    train = ds['train'].shuffle(1024).batch(batch_size).prefetch(AUTOTUNE).map(
        lambda x: transform(x, sparse)
    )
    test = ds['test'].shuffle(1024).batch(batch_size).prefetch(AUTOTUNE).map(
        lambda x: transform(x, sparse)
    )
    return train, test


def load_fashion_mnist_keras(sparse=False):
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    if not sparse:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test


def resize_images_generator(X, y, batch_size, shape):
    size = len(X)
    epochs = len(X) // batch_size + 1

    while True:
        for e in range(epochs):
            images = X[e * batch_size: (e + 1) * batch_size]
            labels = y[e * batch_size: (e + 1) * batch_size]
            yield resize(images, shape), labels
