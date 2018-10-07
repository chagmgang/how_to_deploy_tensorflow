import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from cyclegan import CycleGAN

batch_size = 1

with tf.device('/gpu:0'):
    model = CycleGAN(256, 256, xchan=3, ychan=3)

saver = tf.train.Saver()

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
start = 0
saver.restore(sess, 'models/model.ckpt')

img = plt.imread('horse2zebra/trainA/n02381460_86.jpg')
img = img / 255
b_from_a = model.sample_gy(sess, [img])
plt.subplot(2,1,1)
plt.imshow(img)
plt.subplot(2,1,2)
plt.imshow(b_from_a[0])
plt.show()