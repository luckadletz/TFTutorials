from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()

print("TensorFlow version {}".format(tf.VERSION))
print("Eager execution {}".format(tf.executing_eagerly()))
