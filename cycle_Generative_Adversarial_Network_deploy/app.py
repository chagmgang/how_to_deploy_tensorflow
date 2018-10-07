import numpy as np
from flask import Flask
from flask import request
from flask import jsonify
import codecs
import os
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from cyclegan import CycleGAN
import time

app = Flask(__name__)
with tf.device('/gpu:0'):
    model = CycleGAN(256, 256, xchan=3, ychan=3)

saver = tf.train.Saver()

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
saver.restore(sess, 'models/model.ckpt')
print('restore')

@app.route("/prediction", methods=['POST'])
def get_prediction():
    req_data = request.get_json()
    raw_data = req_data['data']

    img = plt.imread('horse2zebra/trainA/n02381460_86.jpg')
    img = img / 255
    
    x = time.time()
    b_from_a = model.sample_gy(sess, [img])
    print(time.time() - x)
    
    # ndarray cannot be converted to JSON
    return jsonify({ 'predictions': b_from_a.tolist()})

if __name__ == '__main__':
    app.run(host='localhost',port=3000)