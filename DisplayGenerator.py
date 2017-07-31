from GlobalConstants import COLOR_LIST, SHAPE_LIST, SIZE_LIST
from Object import Object
import numpy as np
import copy
import random
from Display import Display

class DisplayGenerator(object):

    def __init__(self):
        self.objects = np.array( [[[Object(col, shp, size, 0) for col in COLOR_LIST]
                                   for shp in SHAPE_LIST] for size in SIZE_LIST], dtype=Object)

        self.objects = np.append(self.objects, Object("BLANK", "BLANK", 0.0, -1))
        self.objects = np.append(self.objects, Object("BLANK", "BLANK", 0.0, -1))
        np.random.shuffle(self.objects)

    def sample(self):

        np.random.shuffle(self.objects)

        display = np.empty((7,11), dtype=Object)

        i = 0

        for row in range(0, 7, 1):
            for col in range(0, 11, 1):
                self.objects[i].text = i+1
                display[row][col] = self.objects[i]
                i = i + 1
        flag = 0
        while flag == 0:
            target_indx = random.sample(range(0, 77, 1), 1)[0]
            if self.objects[target_indx].color != "BLANK":
                flag = 1

        x = [0.0]
        ecc_x = np.zeros((7, 11))
        for row in range(0, 7, 1):
            for col in range(0, 11, 1):
                if col == 0:
                    ecc_x[row][col] = x[0]
                    x[0] += display[row][col].size/2
                else:
                    x[0] += 1.0 + display[row][col].size/2
                    ecc_x[row][col] = x[0]
                    x[0] += display[row][col].size/2

            x[0] = 0.0

        y = [0.0]
        ecc_y = np.zeros((7, 11))
        for col in range(0, 11, 1):
            for row in range(0, 7, 1):
                if row == 0:
                    ecc_y[row][col] = y[0]
                    y[0] += display[row][col].size/2
                else:
                    y[0] += 1.0 + display[row][col].size/2
                    ecc_y[row][col] = y[0]
                    y[0] += display[row][col].size/2

            y[0] = 0.0

        env = Display(display, self.objects[target_indx], ecc_x, ecc_y)

        return env

    def get_objects(self):
        return self.objects
