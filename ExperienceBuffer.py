import numpy as np
import random
from GlobalConstants import TUPLE_SIZE


class ExperienceBuffer(object):

    def __init__(self, buffer_size=100000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):

        if len(self.buffer) >= self.buffer_size:
            self.buffer[random.randint(0, self.buffer_size-1)] = experience
        else:
            self.buffer.append(experience)

    def sample(self, batch_size, trace_length):

        sampled_episodes = random.sample(self.buffer, batch_size)
        sampled_traces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode)+1-trace_length)
            sampled_traces.append(episode[point:point+trace_length])

            sampled_traces = np.array(sampled_traces)

        return np.reshape(sampled_traces, [batch_size*trace_length,TUPLE_SIZE])