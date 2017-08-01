import numpy as np
import random
import tensorflow as tf
import itertools

from QNetwork import QNetwork
from Experiment_Env import Experiment_Env
from GlobalConstants import MAX_ACTIONS, N_ELEMENTS
from Utility_Func import *
from ExperienceBuffer import ExperienceBuffer

#Some training variables.
batch_size = 32
trace_length = 1
update_freq = 10000
gamma = .99
startE = 1.0
endE = 0.01
anneling_steps = 10000000
training_episodes = 20000000
pre_train_steps = 1000
load_model = False
h_size = 77
summary_length = 100000
tau = 0.001



env = Experiment_Env()
tf.reset_default_graph()

#recurrent nodes.

cell = tf.contrib.rnn.BasicLSTMCell(num_units=N_ELEMENTS+N_ELEMENTS+N_ELEMENTS) #RNN for online with default tanh activation.
cellTarget = tf.contrib.rnn.BasicLSTMCell(num_units=N_ELEMENTS+N_ELEMENTS+N_ELEMENTS) #RNN for target with default tanh activation.

mainQN = QNetwork(h_size, trace_length, cell, 'main')
targetQN = QNetwork(h_size, trace_length, cellTarget, 'target')

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=1)

trainable_vars = tf.trainable_variables()

targetOps = updateTargetGraph(trainable_vars, tau)

replay_buffer = ExperienceBuffer()

eps = startE
stepDrop = (startE - endE)/anneling_steps

#database to keep track for some model stats.
rList = [0] * 10000
qList = [0] * 10000

total_steps = 0



"""
Begin Model Training.
"""

with tf.Session() as sess:

    if load_model == True:
        print 'Loading Model...'
        ckpt = tf.train.get_checkpoint_state("./Model")
