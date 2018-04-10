from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf

tf.enable_eager_execution()

print("TensorFlow version {}".format(tf.VERSION))
print("Eager execution {}".format(tf.executing_eagerly()))

# Load training data
train_dataset_filePath = ".\\iris_training.csv"
train_dataset = tf.data.TextLineDataset(train_dataset_filePath)

# Skip the first header row
train_dataset = train_dataset.skip(1)
# Parse each row
train_dataset = train_dataset.map(parse_csv) 
# Randomize
train_dataset = train_dataset.shuffle(buffer_size = 1000)
train_dataset = train_dataset.batch(32)

# View a single example entry from a branch


def parse_csv(line):
    example_defaults = [[0.],[0.],[0.],[0.],[0.]] # Sets field types
    parsed_line = tf.decode_csv(line, example_defaults)
    # First 4 fields are features, combine into single tensor
    features = tf.reshape(parsed_line[:-1], shape=(4,))
    # last field is label
    label = tf.reshape(parsed_line[-1], shape = ())
    return features, label
