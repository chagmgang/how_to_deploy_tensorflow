import tensorflow as tf
import glob
import collections
from itertools import chain
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils import build_dataset


files = glob.glob('*.txt')

words = []
for f in files:
    file = open(f)
    words.append(file.read())
    file.close()

sentences = words[0].split('\n')
sentences = sentences[:-1]

vocabulary_size = 40000
data, count, dictionary, reverse_dictionary = build_dataset(sentences, vocabulary_size)

skip_window = 3
instances = 0
for i in range(len(data)):
    data[i] = [vocabulary_size]*skip_window + data[i] + [vocabulary_size]*skip_window

for sentence  in data:
    instances += len(sentence)-2*skip_window

context = np.zeros((instances,skip_window*2+1),dtype=np.int32)
labels = np.zeros((instances,1),dtype=np.int32)
doc = np.zeros((instances,1),dtype=np.int32)

k = 0
for doc_id, sentence  in enumerate(data):
    for i in range(skip_window, len(sentence)-skip_window):
#         buffer = sentence[i-skip_window:i+skip_window+1]
#         labels[k] = sentence[i]
#         del buffer[skip_window]
#         context[k] = buffer
#         doc[k] = doc_id
#         k += 1
        context[k] = sentence[i-skip_window:i+skip_window+1] # Get surrounding words
        labels[k] = sentence[i] # Get target variable
        doc[k] = doc_id
        k += 1
        
context = np.delete(context,skip_window,1) # delete the middle word        
        
shuffle_idx = np.random.permutation(k)
labels = labels[shuffle_idx]
doc = doc[shuffle_idx]
context = context[shuffle_idx]

batch_size = 256
context_window = 2*skip_window
embedding_size = 50 # Dimension of the embedding vector.
softmax_width = embedding_size # +embedding_size2+embedding_size3
num_sampled = 5 # Number of negative examples to sample.
sum_ids = np.repeat(np.arange(batch_size),context_window)

len_docs = len(data)

# Input data.
train_word_dataset = tf.placeholder(tf.int32, shape=[batch_size*context_window])
train_doc_dataset = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

segment_ids = tf.constant(sum_ids, dtype=tf.int32)

word_embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
word_embeddings = tf.concat([word_embeddings,tf.zeros((1,embedding_size))],0)
doc_embeddings = tf.Variable(tf.random_uniform([len_docs,embedding_size],-1.0,1.0))

softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, softmax_width],
                            stddev=1.0 / np.sqrt(embedding_size)))
softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Model.
# Look up embeddings for inputs.
embed_words = tf.segment_mean(tf.nn.embedding_lookup(word_embeddings, train_word_dataset),segment_ids)
embed_docs = tf.nn.embedding_lookup(doc_embeddings, train_doc_dataset)
embed = (embed_words+embed_docs)/2.0#+embed_hash+embed_users

# Compute the softmax loss, using a sample of the negative labels each time.
loss = tf.reduce_mean(tf.nn.nce_loss(softmax_weights, softmax_biases, train_labels, 
                                        embed, num_sampled, vocabulary_size))

# Optimizer.
optimizer = tf.train.AdagradOptimizer(0.5).minimize(loss)
    
norm = tf.sqrt(tf.reduce_sum(tf.square(doc_embeddings), 1, keep_dims=True))
input_signal = tf.placeholder(tf.float32, shape=[None], name='inputs')
normalized_doc_embeddings = tf.divide(doc_embeddings, norm, name='prediction')

print(input_signal)
print(normalized_doc_embeddings)


############################
# Chunk the data to be passed into the tensorflow Model
###########################

data_idx = 0

def generate_batch(batch_size, instances, labels, doc, context):
    global data_idx
    if data_idx+batch_size<instances:
        batch_labels = labels[data_idx:data_idx+batch_size]
        batch_doc_data = doc[data_idx:data_idx+batch_size]
        batch_word_data = context[data_idx:data_idx+batch_size]
        data_idx += batch_size
    else:
        overlay = batch_size - (instances-data_idx)
        batch_labels = np.vstack([labels[data_idx:instances],labels[:overlay]])
        batch_doc_data = np.vstack([doc[data_idx:instances],doc[:overlay]])
        batch_word_data = np.vstack([context[data_idx:instances],context[:overlay]])
        data_idx = overlay
    batch_word_data = np.reshape(batch_word_data,(-1,1))

    return batch_labels, batch_word_data, batch_doc_data

num_steps = 1000001

step_delta = int(num_steps/20)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

print('Initialized')
average_loss = 0
for step in range(num_steps):
    batch_labels, batch_word_data, batch_doc_data\
    = generate_batch(batch_size, instances, labels, doc, context)
    feed_dict = {train_word_dataset : np.squeeze(batch_word_data),
                    train_doc_dataset : np.squeeze(batch_doc_data),
                    train_labels : batch_labels}
    _, l = sess.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += l

    if step % step_delta == 0:
        if step > 0:
            average_loss = average_loss / step_delta
        # The average loss is an estimate of the loss over the last 2000 batches.
        print('Average loss at step %d: %f' % (step, average_loss))
        average_loss = 0

SAVE_PATH = './save'
MODEL_NAME = 'test'
path = saver.save(sess, SAVE_PATH + '/' + MODEL_NAME + '.ckpt')
print("saved at {}".format(path))
