from DisplayGenerator import DisplayGenerator
from ObservationModel import ObservationModel
from GlobalConstants import N_ELEMENTS, COLOUR_PENALTY, CLICK_ACTION, N_ROWS, N_COLS, COLOUR_BONUS, MAX_STEPS
import numpy as np
import random

"""
Environment adapted from: 
Visual Search of Displays of Many Objects: Modeling Detailed Eye Movement Effects with Improved EPIC
David E. Kieras, Anthony Hornof and Yunfeng Zhang
"""
class Experiment_Env(object):

    def __init__(self):
        self.steps = 0
        self.generator = DisplayGenerator()
        self.model = ObservationModel()
        self.current_display = None
        self.focus = None
        self.colour_estimate = None
        self.shape_estimate = None
        self.size_estimate = None
        self.text_estimate = None

    def step(self, action):
        self.steps += 1
        done = False

        if action < N_ELEMENTS:
            col, shp, size, txt = self.model.sample(action, self.current_display)
            self.focus = np.eye(N_ELEMENTS)[action]
            self.colour_estimate = col
            self.shape_estimate = shp
            self.size_estimate = size
            self.text_estimate = txt

        if action < N_ELEMENTS:
            reward  = COLOUR_PENALTY
        elif (action == CLICK_ACTION and
                      self.current_display.target.text ==
                      self.current_display.objects[(self.focus/N_COLS)][(self.focus%N_COLS)].text):
            reward = COLOUR_BONUS

        else:
            reward = -5

        if action == CLICK_ACTION or self.steps >= MAX_STEPS:
            done = True

        return np.reshape(self.colour_estimate.flatten(), [77]), self.focus, float(reward), done

    def reset(self):
        self.steps = 0
        self.current_display = self.generator.sample()
        self.focus = np.reshape(np.zeros((N_ROWS,N_COLS)).flatten(), [77])
        self.colour_estimate = np.reshape(np.zeros((N_ROWS,N_COLS)).flatten(), [77])
        self.shape_estimate = np.reshape(np.zeros((N_ROWS,N_COLS)).flatten(), [77])
        self.size_estimate = np.reshape(np.zeros((N_ROWS,N_COLS)).flatten(), [77])
        self.text_estimate = np.reshape(np.zeros((N_ROWS,N_COLS)).flatten(), [77])

        return self.colour_estimate, self.focus

#Test
#env = Experiment_Env()

#col, f = env.reset()

#print col
#print "-----"
#print f

#print "-----"
#print "-----"

#a = random.randint(0, N_ELEMENTS)
#print a
#print "-----"
#col, f, r, d = env.step(a)
#print col



