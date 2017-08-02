import numpy as np
import random
import tensorflow as tf
import itertools

from QNetwork import QNetwork
from Experiment_Env import Experiment_Env
from GlobalConstants import MAX_ACTIONS, N_ELEMENTS, MAX_STEPS, TUPLE_SIZE
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

    #initialize all the variables in network.
    sess.run(init)

    #location to store training summary.
    writer = tf.summary.FileWriter("./Viz", sess.graph)

    #copy online network weights to offline network weights. [Double DQN Algorithm]
    updateTarget(targetOps, sess)

    #Begin training.
    for i in range(training_episodes):

        episode_buffer = []

        #reset the environment for new trial.
        colour_estimate, text_estimate, fixated_loc = env.reset()
        done = False
        total_reward = 0
        steps = 0
        loss = 0
        q_val = 0

        #reset recurrent layer hidden state.
        hidden_state = (np.zeros([1,N_ELEMENTS+N_ELEMENTS+N_ELEMENTS]), np.zeros([1,N_ELEMENTS+N_ELEMENTS+N_ELEMENTS]))

        #Deep Q-Learning.
        while steps < MAX_STEPS:

            #Choose action greedily with epsilon chance of randon action.
            if np.random.rand(1) < eps or total_steps < pre_train_steps:
                state_next, predict = sess.run([mainQN.state, mainQN.Qout],
                                               feed_dict={
                                                   mainQN.featureInput:[colour_estimate],
                                                   mainQN.textInput:[text_estimate],
                                                   mainQN.focusInput:[fixated_loc],
                                                   mainQN.prev_state:[hidden_state],
                                                   mainQN.batch_size:1
                                               })

                action = np.random.randint(0, MAX_ACTIONS)

            else:
                action, state_next, predict = sess.run([mainQN.predict, mainQN.state, mainQN.Qout],
                                               feed_dict={
                                                   mainQN.featureInput:[colour_estimate],
                                                   mainQN.textInput:[text_estimate],
                                                   mainQN.focusInput:[fixated_loc],
                                                   mainQN.prev_state:[hidden_state],
                                                   mainQN.batch_size:1
                                               })

                action = action[0]

            #perfom the selected action.
            colour_estimate_next, text_estimate_next, fixated_loc_next, reward, done = env.step(action)

            if action < N_ELEMENTS:
                steps += 1


            total_steps += 1

            #record the episode in buffer
            episode_buffer.append(np.reshape(np.array(colour_estimate, text_estimate, fixated_loc, action, reward,
                                                      colour_estimate_next, text_estimate_next, fixated_loc_next,
                                                      done), [1,TUPLE_SIZE]))

            if total_steps > pre_train_steps:

                if eps > endE:
                    eps -= stepDrop

                #Update offile network weights after every n-steps.[Double DQN Algorithm]
                if total_steps % update_freq == 0:
                    updateTarget(targetOps, sess)

                #perform experience replay after every 4 steps.
                if total_steps % 4 == 0:

                    #randomly sample stored experiences to train on.
                    training_batch = replay_buffer.sample(batch_size, trace_length)

                    hidden_state_train = (np.zeros([batch_size,N_ELEMENTS+N_ELEMENTS+N_ELEMENTS]),
                                          np.zeros([batch_size,N_ELEMENTS+N_ELEMENTS+N_ELEMENTS]))

                    #setup for Double DQN algorithm.
                    Q1 = sess.run(mainQN.predict,
                                  feed_dict={
                                      mainQN.featureInput: np.vstack(training_batch[:5]),
                                      mainQN.textInput: np.vstack(training_batch[:6]),
                                      mainQN.focusInput: np.vstack(training_batch[:7]),
                                      mainQN.prev_state: hidden_state_train,
                                      mainQN.batch_size: batch_size
                                  })

                    Q2 = sess.run(targetQN.Qout,
                                  feed_dict={
                                      targetQN.featureInput: np.vstack(training_batch[:5]),
                                      targetQN.textInput: np.vstack(training_batch[:6]),
                                      targetQN.focusInput: np.vstack(training_batch[:7]),
                                      targetQN.prev_state: hidden_state_train,
                                      targetQN.batch_size: batch_size
                                  })

                    #flip the booleans values.
                    end_mul = -(training_batch[:8] - 1)

                    doubleQ = Q2[range(batch_size*trace_length), Q1]

                    #end_mul to control if its terminal state then only return reward. Else, sum the second part.
                    targetQ = training_batch[:,4] + (gamma * doubleQ * end_mul)

                    #Update network.

                    model, summary = sess.run([mainQN.updateModel, mainQN.summary],
                                              feed_dict={
                                                  mainQN.featureInput: np.vstack(training_batch[:0]),
                                                  mainQN.textInput: np.vstack(training_batch[:1]),
                                                  mainQN.focusInput: np.vstack(training_batch[:2]),
                                                  mainQN.actions: np.vstack(training_batch[:3]),
                                                  mainQN.targetQ: targetQ,
                                                  mainQN.prev_state: hidden_state_train,
                                                  mainQN.batch_size: batch_size
                                              })

                    writer.add_summary(summary, total_steps)


            total_reward += reward
            q_val += predict[0, action]
            colour_estimate = colour_estimate_next
            text_estimate = text_estimate_next
            fixated_loc = fixated_loc_next
            hidden_state = state_next

            if done == True:
                break


        episode_buffer = zip(np.array(episode_buffer))

        #store episode buffer in replay memory.
        replay_buffer.add(episode_buffer)

        rList[i % 10000] = total_reward
        qList[i % 10000] = q_val

        #Periodically save model.
        if i % summary_length == 0 and i != 0:
            saver.save(sess, './Model/model-'+str(i)+'.cptk')
            print 'Saved model parameters'

        if i % 10000 == 0 and i != 0:
            print 'step: ' + str(i) + ', reward: ' + str(np.round(np.mean(rList), decimals=1)) + ', Q-Value: ' + str(np.round(np.mean(rList), decimals=1))

    saver.save(sess, './Model/model-Final.cptk')

