import numpy as np
import random
import tensorflow as tf
import itertools



def updateTargetGraph(tfVars, tau):
    """
    Function that creates a map of weights that needs to be copied from one network to other.
    :param tfVars: trainable variables in tensor graph.
    :param tau: update rate
    :return: op_holder: operation list
    """
    total_vars = len(tfVars)
    op_holder = []

    for idx, var in enumerate(tfVars[0:total_vars/2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))

    return op_holder

def updateTarget(op_holder, sess):
    """
    Function that actually updates the weights from the online to target network.
    :param op_holder:
    :param sess:
    :return:
    """
    for op in op_holder:
        sess.run(op)
