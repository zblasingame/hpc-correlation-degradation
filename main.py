""" Method to find the HPCs with the highest correlation to performance
    degradation.
    Author: Zander Blasingame
    Organization: CAMEL at Clarkson University """

import numpy as np
import tensorflow as tf
import argparse
import json

from utils import parse_csv

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--train_file',
                    type=str,
                    default='train.csv',
                    help='Location of training file')

parser.add_argument('--test_file',
                    type=str,
                    default='test.csv',
                    help='Location of testing file')

parser.add_argument('--debug',
                    action='store_true',
                    help='Flag to d_print debug messages')

FLAGS = parser.parse_args()


# Helper function for debugging
def d_print(string):
    if FLAGS.debug:
        print(string)

# Grab data
headers, train_data = parse_csv(FLAGS.train_file)
headers, test_data = parse_csv(FLAGS.test_file)


num_in = train_data[0].size - 1
num_out = 1

# Network params
learning_rate = 0.005
training_epochs = 100
training_size = len(train_data)
testing_size = len(test_data)
display_step = 5

# Placeholders
X = tf.placeholder('float', [None, num_in])
Y = tf.placeholder('float', [None, num_out])
keep_prob = tf.placeholder('float')


# Helper function
def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


# Create NN model
def gen_model(X, activation_function, keep_prob):
    with tf.name_scope('weights'):
        weights = tf.Variable(tf.random_normal([num_in, num_out],
                                               stddev=0.1),
                              name='weights')
        variable_summaries(weights, 'weights')

    with tf.name_scope('biases'):
        biases = tf.Variable(tf.random_normal([num_out]))
        variable_summaries(biases, 'biases')

    z = tf.matmul(X, weights) + biases

    return tf.nn.dropout(activation_function(z), keep_prob)

prediction = gen_model(X, tf.identity, keep_prob)

with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.square(prediction - Y))
    tf.scalar_summary('cost', cost)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cost)

init_op = tf.initialize_all_variables()

# Merge all summaries
merged = tf.merge_all_summaries()

with tf.Session() as sess:
    train_writer = tf.train.SummaryWriter('logs/train', sess.graph)

    sess.run(init_op)

    step = 0
    for epoch in range(training_epochs):
        avg_cost = 0
        for i in range(training_size):
            deg, hpc_vec = np.split(train_data[i], [1], axis=1)

            _, c, summary = sess.run([optimizer, cost, merged],
                                     feed_dict={X: hpc_vec,
                                                Y: deg,
                                                keep_prob: 1.0})

            avg_cost += c / training_size

            train_writer.add_summary(summary, step)

            step += 1

        if (epoch + 1) % display_step == 0:
            d_print('Epoch: {0:03} with cost={1:.9f}'.format(epoch+1,
                                                             avg_cost))

    d_print('Optimization Finished')

    # Evaluation

    avg_cost = 0

    for i in range(testing_size):
        deg, hpc_vec = np.split(test_data[i], [1], axis=1)
        c = sess.run(cost, feed_dict={X: hpc_vec, Y: deg, keep_prob: 1.0})

        avg_cost += c / testing_size

    d_print('Average error: {}'.format(avg_cost))

    # Get weights into mapping
    weights = next(v for v in tf.all_variables()
                   if v.name == 'weights/weights:0')

    weight_map = [dict(name=headers[i+1], weight=weight[0])
                  for i, weight in enumerate(sess.run(weights))]

    impact_list = sorted(weight_map,
                         key=lambda k: abs(k['weight']),
                         reverse=True)

    for i, item in enumerate(impact_list):
        item['rank'] = i + 1
        item['weight'] = float(item['weight'])

    print('JSON out')
    print(json.dumps(impact_list, sort_keys=True, indent=4))
