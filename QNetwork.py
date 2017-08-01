import tensorflow as tf
import tensorflow.contrib.slim as slim
from GlobalConstants import N_ELEMENTS, MAX_ACTIONS, COLOUR_BONUS

class QNetwork(object):

    def __init__(self, h_size, training_length, rnn_cell, myScope):

        #Network inputs
        self.featureInput = tf.placeholder(shape=[None,N_ELEMENTS], dtype=tf.float32)
        self.textInput = tf.placeholder(shape=[None,N_ELEMENTS], dtype=tf.float32)
        self.focusInput = tf.placeholder(shape=[None,N_ELEMENTS], dtype=tf.float32)

        #Network training parameters
        self.batch_size = tf.placeholder(shape=[], dtype=tf.float32)

        #merge inputs.
        self.merge = tf.concat([self.featureInput, self.textInput, self.focusInput], 1)

        #feed the merged input to recurrent neural network.
        self.observation = tf.reshape(self.merge, [self.batch_size, training_length, (N_ELEMENTS+N_ELEMENTS+N_ELEMENTS)])

        self.prev_state = rnn_cell.zero_state(self.batch_size, tf.float32)

        #function that unrolls the recurrent network.
        self.rnn, self.state = tf.nn.dynamic_rnn(inputs=self.observation, cell=rnn_cell, initial_state=self.prev_state,
                                                 dtype=tf.float32, scope=myScope+'_rnn')

        self.rnn = tf.reshape(self.rnn, shape=[-1,N_ELEMENTS+N_ELEMENTS+N_ELEMENTS])
        tf.summary.histogram(myScope+"_rnn_state", self.state)

        #recurrent output to feedforward layer
        self.hidden = slim.fully_connected(self.rnn, h_size, activation_fn=tf.nn.sigmoid, scope=myScope+'_hidden')
        tf.summary.histogram(myScope + "_hidden", self.hidden)

        #Q-output layer.
        self.Qout = slim.fully_connected(self.hidden, MAX_ACTIONS, activation_fn=None, scope=myScope+'_Qout')
        tf.summary.histogram(myScope + "_Qout", self.Qout)

        self.predict = tf.argmax(self.Qout, 1)

        #Calculate Loss
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions_onehot = tf.placeholder(self.actions, MAX_ACTIONS, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)
        self.td_error = (self.targetQ - self.Q)

        #Clip loss to avoid divergence.
        self.clipped_error = tf.where(tf.abs(self.td_error) < COLOUR_BONUS, 0.5*tf.square(self.td_error),
                                      COLOUR_BONUS * (tf.abs(self.td_error) - 0.5 * COLOUR_BONUS))

        self.loss = tf.reduce_mean(self.clipped_error)
        tf.summary.scalar("Loss", self.loss)

        #Optimizer for gradient decent.
        self.trainer = tf.train.AdamOptimizer()
        self.updateModel = self.trainer.minimize(self.loss)
        self.summary = tf.summary.merge_all()

