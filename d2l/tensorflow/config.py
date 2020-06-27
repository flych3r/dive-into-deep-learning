import os
import tensorflow as tf


def set_tf_loglevel(level=3):
    """Sets tensorflow logging level.

    Parameters
    ----------
    level : int, optional, by default 3
        3: Fatal
        2: Error
        1: Warning
        0: Disabled
    """
    level = str(level)
    assert level in ['0', '1', '2', '3'], 'invalid logging level'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = level


def use_device(device='CPU'):
    """Configures tensorflow to use cpu or gpu.

    Parameters
    ----------
    device : str, optional, {'CPU' or 'GPU'}
        device to use, by default 'CPU'
    """
    device = device.upper()
    assert device in ['CPU', 'GPU'], 'device must be CPU or GPU'

    if device == 'CPU':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    tf.random.normal(shape=(1, 1))
    print('Tensorflow running on {}'.format(device))


def setup(device='CPU', level=3):
    """Configures tensorflow device and logging level

    Parameters
    ----------
    device : str, optional, {'CPU', 'GPU'}
        Device that tensorflow will use, by default 'CPU'
    level : int, optional, {0, 1 ,2, 3}
        Tensorflow logging level, by default 3
        3: Fatal
        2: Error
        1: Warning
        0: Disabled
    """
    set_tf_loglevel(level)
    use_device(device)
