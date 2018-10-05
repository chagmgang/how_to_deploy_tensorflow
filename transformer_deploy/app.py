import numpy as np
from flask import Flask
from flask import request
from flask import jsonify
import codecs
import os
import tensorflow as tf
from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_vocab, load_en_vocab
from train import Graph
from nltk.translate.bleu_score import corpus_bleu

app = Flask(__name__)

g = Graph(is_training=False)
print('Graph loaded')
X, Sources, Targets = load_test_data()
de2idx, idx2de = load_de_vocab()
en2idx, idx2en = load_en_vocab()

with g.graph.as_default():
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
    print("Restored!")
mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
if not os.path.exists('results'): os.mkdir('results')
with codecs.open("results/" + mname, "w", "utf-8") as fout:
    list_of_refs, hypotheses = [], []
    for i in range(len(X) // hp.batch_size):
        ### Get mini-batches
        x = X[i*hp.batch_size: (i+1)*hp.batch_size]
        sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
        targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]

@app.route("/prediction", methods=['POST'])
def get_prediction():
    req_data = request.get_json()
    raw_data = req_data['data']
    ### Autoregressive inference
    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
    for j in range(hp.maxlen):
        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
        preds[:, j] = _preds[:, j]
    gots = []
    for source, target, pred in zip(sources, targets, preds): # sentence-wise
        got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
        gots.append(got)
    prediction = {'outputs': gots}

    # ndarray cannot be converted to JSON
    return jsonify({ 'predictions': prediction['outputs']})

if __name__ == '__main__':
    app.run(host='localhost',port=3000)