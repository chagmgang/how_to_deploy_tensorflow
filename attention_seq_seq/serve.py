import tensorflow as tf
import os

SAVE_PATH = './save'
MODEL_NAME = 'test'
VERSION = 1
SERVE_PATH = './serve/{}/{}'.format(MODEL_NAME, VERSION)

checkpoint = tf.train.latest_checkpoint(SAVE_PATH)

tf.reset_default_graph()

with tf.Session() as sess:
    # import the saved graph
    saver = tf.train.import_meta_graph(checkpoint + '.meta')
    # get the graph for this session
    graph = tf.get_default_graph()
    saver.restore(sess, checkpoint)
    # get the tensors that we need
    inputs = graph.get_tensor_by_name('inputs:0')
    outputs = graph.get_tensor_by_name('outputs:0')
    targets = graph.get_tensor_by_name('targets:0')
    predictions = graph.get_tensor_by_name('prediction:0')

    model_input = tf.saved_model.utils.build_tensor_info(inputs)
    model_output = tf.saved_model.utils.build_tensor_info(outputs)
    model_target = tf.saved_model.utils.build_tensor_info(targets)
    model_prediction = tf.saved_model.utils.build_tensor_info(predictions)

    signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'inputs': model_input, 'outputs': model_output, 'targets': model_target},
        outputs={'predictions': model_prediction},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    builder = tf.saved_model.builder.SavedModelBuilder(SERVE_PATH)

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_definition
        })
    # Save the model so we can serve it with a model server :)
    builder.save()