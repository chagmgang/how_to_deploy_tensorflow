import tensorflow as tf

class s2s:
    def __init__(self, enc_sent_size, output_sent_size, vocab_size):

        self.enc_input = tf.placeholder(tf.float32, [None, enc_sent_size - 1, vocab_size], name='inputs')
        self.dec_input = tf.placeholder(tf.float32, [None, output_sent_size, vocab_size], name='outputs')
        # [batch size, time steps]
        self.targets = tf.placeholder(tf.float32, [None, output_sent_size], name='targets')
        targets = tf.cast(self.targets, tf.int64)

        with tf.variable_scope('encode'):
            enc_cell = [tf.nn.rnn_cell.GRUCell(size) for size in [256, 128]]
            enc_cell = tf.nn.rnn_cell.MultiRNNCell(enc_cell)
            outputs_enc, enc_states = tf.nn.dynamic_rnn(cell=enc_cell, inputs=self.enc_input, dtype=tf.float32)

        with tf.variable_scope('decode'):
            dec_cell = [tf.nn.rnn_cell.GRUCell(size) for size in [256, 128]]
            dec_cell = tf.nn.rnn_cell.MultiRNNCell(dec_cell)
            outputs_dec, dec_states = tf.nn.dynamic_rnn(cell=dec_cell, inputs=self.dec_input, initial_state=enc_states,
                                        dtype=tf.float32)

        self.model = tf.layers.dense(outputs_dec, vocab_size, activation=None)


        self.cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.model, labels=targets))

        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

        self.prediction = tf.argmax(self.model, 2, name='prediction')

        print(self.enc_input)
        print(self.dec_input)
        print(self.targets)
        print(self.prediction)