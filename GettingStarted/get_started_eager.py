from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

print("TensorFlow version {}".format(tf.VERSION))
print("Eager execution {}".format(tf.executing_eagerly()))

def parse_csv(line):
    example_defaults = [[0.],[0.],[0.],[0.],[0.]] # Sets field types
    parsed_line = tf.decode_csv(line, example_defaults)
    # First 4 fields are features, combine into single tensor
    features = tf.reshape(parsed_line[:-1], shape=(4,))
    # last field is label
    label = tf.reshape(parsed_line[-1], shape = ())
    return features, label


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
features, label = tfe.Iterator(train_dataset).next()
print("Example features:", features[0])
print("Example label:", label[0])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)), # input shape required
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(3)
])

def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

