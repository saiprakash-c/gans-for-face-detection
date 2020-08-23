import tensorflow as tf
import os

import numpy as np
import inspect

"""
vgg19_npy_path = None

if vgg19_npy_path is None:
    path = inspect.getfile(vgg19)
    path = os.path.abspath(os.path.join(path, os.pardir))
    path = os.path.join(path, "vgg19.npy")
    vgg19_npy_path = path
    print(vgg19_npy_path)


"""

data_dict = np.load('../vgg19.npy', encoding='latin1').item()



def get_conv_filter(self, name):
    return tf.constant(self.data_dict[name][0], name="filter")


def get_bias(self, name):
    return tf.constant(self.data_dict[name][1], name="biases")


def get_fc_weight(self, name):
    return tf.constant(self.data_dict[name][0], name="weights")