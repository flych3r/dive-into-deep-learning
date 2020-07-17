import collections
import random
import re

import tensorflow_datasets as tfds
from tensorflow import convert_to_tensor, gather, one_hot, reshape
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


def batch_iter(features, labels, batch_size=256, shuffle=True, continuous=True):
    """Iterates thru a dataset."""
    num_examples = features.shape[0]
    indices = [*range(num_examples)]
    if shuffle:
        random.shuffle(indices)

    while True:
        for i in range(0, num_examples, batch_size):
            j = convert_to_tensor(indices[i: min(i + batch_size, num_examples)])
            yield gather(features, j), gather(labels, j)
        if not continuous:
            break


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


def load_corpus(file_path, max_tokens=-1):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [re.sub('[^A-Za-z]+', ' ', line.strip().lower()) for line in lines]
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[tk] for line in tokens for tk in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def tokenize(lines, token='word'):
    """Split sentences into word or char tokens."""
    if token == 'word':
        return [line.split(' ') for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type ' + token)


def count_corpus(sentences):
    # Flatten a list of token lists into a list of tokens
    tokens = [tk for line in sentences for tk in line]
    return collections.Counter(tokens)


class Vocab:
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        counter = collections.Counter(tokens)
        token_freqs = sorted(counter.items(), key=lambda x: x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True)
        if use_special_tokens:
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
        else:
            self.unk = 0
            special_tokens = ['<unk>']
        tokens = [token for token, freq in token_freqs
                  if freq >= min_freq and token not in special_tokens]
        self.idx_to_token = []
        self.token_to_idx = dict()
        for token in special_tokens + tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        else:
            return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        else:
            return [self.idx_to_token[index] for index in indices]


def seq_data_iter_random(corpus_indices, batch_size, num_steps):
    # offset for the iterator over the data for uniform starts
    offset = int(random.uniform(0, num_steps))
    corpus_indices = corpus_indices[offset:]
    # subtract 1 extra since we need to account for the sequence length
    num_examples = ((len(corpus_indices) - 1) // num_steps) - 1
    # discard half empty batches
    num_batches = num_examples // batch_size
    example_indices = list(range(0, num_examples * num_steps, num_steps))
    random.shuffle(example_indices)

    # This returns a sequence of the length num_steps starting from pos.
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(0, batch_size * num_batches, batch_size):
        # batch_size indicates the random examples read each time.
        batch_indices = example_indices[i:(i + batch_size)]
        X = [_data(j) for j in batch_indices]
        Y = [_data(j + 1) for j in batch_indices]

        yield convert_to_tensor(X), convert_to_tensor(Y)


def seq_data_iter_consecutive(corpus_indices, batch_size, num_steps):
    # offset for the iterator over the data for uniform starts
    offset = int(random.uniform(0, num_steps))
    # slice out data - ignore num_steps and just wrap around
    num_indices = ((len(corpus_indices) - offset) // batch_size) * batch_size
    indices = convert_to_tensor(corpus_indices[offset:(offset + num_indices)])
    indices = reshape(indices, (batch_size, -1))
    # need to leave one last token since targets are shifted by 1
    num_epochs = ((num_indices // batch_size) - 1) // num_steps

    for i in range(0, num_epochs * num_steps, num_steps):
        X = indices[:, i:(i + num_steps)]
        Y = indices[:, (i + 1):(i + 1 + num_steps)]
        yield convert_to_tensor(X), convert_to_tensor(Y)


class SeqDataLoader:
    """A iterator to load sequence data."""
    def __init__(self, file_path, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_consecutive
        self.corpus, self.vocab = load_corpus(file_path, max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_seq_data(
    file_path, batch_size, num_steps, use_random_iter=False, max_tokens=10000
):
    data_iter = SeqDataLoader(
        file_path, batch_size, num_steps, use_random_iter, max_tokens
    )
    return data_iter, data_iter.vocab
