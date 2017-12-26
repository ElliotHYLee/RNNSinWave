import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers
from tensorflow.contrib import rnn
import numpy as np

def lstm_model(time_steps, rnn_layers, fcDim=None, learning_rate=0.1, optimizer='Adagrad',learning_rate_decay_fn = None):
    # [Ftrl, Adam, Adagrad, Momentum, SGD, RMSProp]
    """
        Creates a deep model based on:
            * stacked lstm cells
            * an optional dense layers
        :param num_units: the size of the cells.
        :param rnn_layers: list of int or dict
                             * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                             * list of dict: [{steps: int, keep_prob: int}, ...]
        :param dense_layers: list of nodes for each layer
        :return: the model definition
        """

    def lstm_cells():
        # return [rnn.DropoutWrapper(rnn.BasicLSTMCell(layer['TimeSteps'],state_is_tuple=True),layer['keep_prob'])
        #         if layer.get('keep_prob')
        #         else rnn.BasicLSTMCell(layer['TimeSteps'], state_is_tuple=True)
        #         for layer in rnn_layers]

        return [rnn.DropoutWrapper(rnn.BasicLSTMCell(layer['TimeSteps'],state_is_tuple=True),layer['keep_prob'])
                for layer in rnn_layers]

    # def dnn_layers(input_layers):
    #     return tflayers.stack(input_layers, tflayers.fully_connected, fcDim)

    def _lstm_model(X, y):
        x_ =  tf.unstack(X, num=time_steps, axis=1)
        stacked_lstm = rnn.MultiRNNCell(lstm_cells(), state_is_tuple=True)
        output, lastState = rnn.static_rnn(stacked_lstm, x_, dtype=dtypes.float32)
        output = tflayers.stack(output[-1], tflayers.fully_connected, fcDim)
        prediction, loss = tflearn.models.linear_regression(output, y)
        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(), optimizer=optimizer,
            learning_rate = tf.train.exponential_decay(learning_rate, tf.contrib.framework.get_global_step(),
            decay_steps = 100, decay_rate = 0.9, staircase=False, name=None))

        # print('learning_rate',learning_rate)
        return prediction, loss, train_op

    # https://www.tensorflow.org/versions/r0.10/api_docs/python/train/decaying_the_learning_rate

    return _lstm_model
