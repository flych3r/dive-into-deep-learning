import random

import tensorflow_datasets as tfds
from tensorflow import convert_to_tensor, gather
from tensorflow.data.experimental import AUTOTUNE


def transform(example):
    """Transforms the dataset containing images."""
    return example['image'] / 255, example['label']


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


def load_tfds_dataset(ds_name, batch_size=256):
    """Loads and transforms a tfds dataset."""
    ds = tfds.load(ds_name, shuffle_files=True)
    train = ds['train'].shuffle(1024).batch(batch_size).prefetch(AUTOTUNE).map(transform)
    test = ds['test'].shuffle(1024).batch(batch_size).prefetch(AUTOTUNE).map(transform)
    return train, test
