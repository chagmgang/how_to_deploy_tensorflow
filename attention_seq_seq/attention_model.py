import tensorflow as tf

class s2s:
    def __init__(self, enc_sent_size, output_sent_size, vocab_size):

        self.enc_input_size = enc_sent_size - 1

        self.enc_input = tf.placeholder(tf.float32, [None, self.enc_input_size, vocab_size], name='inputs')
        self.dec_input = tf.placeholder(tf.float32, [None, output_sent_size, vocab_size], name='outputs')
        # [batch size, time steps]
        self.targets = tf.placeholder(tf.float32, [None, None], name='targets')
        targets = tf.cast(self.targets, tf.int64)


        with tf.variable_scope('encode'):
            enc_cell = [tf.nn.rnn_cell.GRUCell(size) for size in [1024, 512, 256, 128]]
            enc_cell = tf.nn.rnn_cell.MultiRNNCell(enc_cell)
            outputs_enc, enc_states = tf.nn.dynamic_rnn(cell=enc_cell, inputs=self.enc_input, dtype=tf.float32)

        with tf.variable_scope('decode'):
            dec_cell = [tf.nn.rnn_cell.GRUCell(size) for size in [1024, 512, 256, 128]]
            dec_cell = tf.nn.rnn_cell.MultiRNNCell(dec_cell)
            outputs_dec, dec_states = tf.nn.dynamic_rnn(cell=dec_cell, inputs=self.dec_input, initial_state=enc_states,
                                        dtype=tf.float32)

        expand_outputs_dec = tf.expand_dims(outputs_dec, 2)
        expand_outputs_enc = tf.expand_dims(outputs_enc, 1)
        tile_outputs_dec = tf.tile(expand_outputs_dec, [1, 1, self.enc_input_size, 1])

        context_vector = tf.multiply(tile_outputs_dec, expand_outputs_enc)
        context_vector_reshape = tf.reshape(context_vector, [-1, output_sent_size, 128 * self.enc_input_size])
        context_vector = tf.layers.dense(inputs=context_vector_reshape, units=128, activation=None)
        
        output = tf.multiply(context_vector, outputs_dec)

        self.model = tf.layers.dense(output, vocab_size, activation=None)

        self.cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.model, labels=targets))

        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

        self.prediction = tf.argmax(self.model, 2, name='prediction')
