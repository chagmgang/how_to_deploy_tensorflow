import tensorflow as tf
import numpy as np
import os, sys
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

DATA_SIZE = 100
SAVE_PATH = './save'
EPOCHS = 400
LEARNING_RATE = 0.01
MODEL_NAME = 'test'

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, 784], name='inputs')
y = tf.placeholder(tf.float32, shape=[None, 10], name='targets')

reshape = tf.reshape(x, [-1, 28, 28, 1])
l1 = tf.layers.conv2d(reshape, 32, [3, 3], activation=tf.nn.relu)
l1 = tf.layers.max_pooling2d(l1, [2, 2], [2, 2])
l1 = tf.layers.dropout(l1, 0.7)

L2 = tf.layers.conv2d(l1, 64, [3, 3], activation=tf.nn.relu)
L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
L2 = tf.layers.dropout(L2, 0.7)

L3 = tf.contrib.layers.flatten(L2)
L3 = tf.layers.dense(L3, 256, activation=tf.nn.relu)
L3 = tf.layers.dropout(L3, 0.5)

pred = tf.layers.dense(inputs=L3, units=10, activation=tf.nn.softmax, name='prediction')

cost = -tf.reduce_mean(y * tf.log(pred))
train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

#checkpoint = tf.train.latest_checkpoint(SAVE_PATH)
#should_train = checkpoint == None

saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(10):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([train_step, cost],
                               feed_dict={x: batch_xs,
                                          y: batch_ys})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.4f}'.format(total_cost / total_batch))

    result = sess.run(pred, feed_dict={x: batch_xs})
    print(result)

path = saver.save(sess, SAVE_PATH + '/' + MODEL_NAME + '.ckpt')
print("saved at {}".format(path))