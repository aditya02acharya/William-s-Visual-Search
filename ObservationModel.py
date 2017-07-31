from GlobalConstants import N_ROWS, N_COLS, FEATURE_COLOUR, FEATURE_SHAPE, FEATURE_SIZE, FEATURE_TEXT
import numpy as np
from scipy.spatial import distance
from DisplayGenerator import DisplayGenerator


class ObservationModel(object):

    def sample(self, action, current_display):
        """
        Samples a random observation from a given display.
        :param seed:
        :return: 2D array with colour and shape noisy observation
        """
        x = action / N_COLS
        y = action % N_COLS
        obs_space_col = self.observe_feature(current_display, x, y, FEATURE_COLOUR)
        obs_space_shp = self.observe_feature(current_display, x, y, FEATURE_SHAPE)
        obs_space_size = self.observe_feature(current_display, x, y, FEATURE_SIZE)
        obs_space_text = self.observe_feature(current_display, x, y, FEATURE_TEXT)

        return obs_space_col, obs_space_shp, obs_space_size, obs_space_text

    def observe_feature(self, display, x, y, feature):
        observation = self.add_feature_noise(display, x, y, feature)

        #observation = self.add_spatial_noise(temp, x, y, global_variables)

        return observation

    def add_feature_noise(self, features, x, y, feature):
        obs_space = np.ones((N_ROWS, N_COLS)) * -1

        #for COLOUR.
        if feature == FEATURE_COLOUR:
            for ext_x in range(0, N_ROWS, 1):
                for ext_y in range(0, N_COLS, 1):
                    e = self.get_eccentricity(features.x[x][y], features.y[x][y],
                                features.x[ext_x][ext_y], features.y[ext_x][ext_y])
                    mu = 0.05 + (0.2*e) + (0*e*e) + (0.0004*e*e*e)
                    if features.objects[ext_x][ext_y].color != "BLANK":
                        if features.objects[ext_x][ext_y].size > np.random.normal(mu,0.5,1)[0]:
                            if features.objects[ext_x][ext_y].color == features.target.color:
                                obs_space[ext_x][ext_y] = 1
                        else:
                            obs_space[ext_x][ext_y] = 0

        #for SHAPE.
        if feature == FEATURE_SHAPE:
            for ext_x in range(0, N_ROWS, 1):
                for ext_y in range(0, N_COLS, 1):
                    e = self.get_eccentricity(features.x[x][y], features.y[x][y],
                                features.x[ext_x][ext_y], features.y[ext_x][ext_y])
                    mu = 0.05 + (0.2*e) + (0*e*e) + (0.025*e*e*e)
                    if features.objects[ext_x][ext_y].color != "BLANK":
                        if features.objects[ext_x][ext_y].size > np.random.normal(mu,0.5,1)[0]:
                            if features.objects[ext_x][ext_y].shape == features.target.shape:
                                obs_space[ext_x][ext_y] = 1
                        else:
                            obs_space[ext_x][ext_y] = 0

        #for SIZE.
        if feature == FEATURE_SIZE:
            for ext_x in range(0, N_ROWS, 1):
                for ext_y in range(0, N_COLS, 1):
                    e = self.get_eccentricity(features.x[x][y], features.y[x][y],
                                features.x[ext_x][ext_y], features.y[ext_x][ext_y])
                    mu = 0.05 + (0.2*e) + (0*e*e) + (0.0004*e*e*e)
                    if features.objects[ext_x][ext_y].color != "BLANK":
                        if features.objects[ext_x][ext_y].size > np.random.normal(mu,0.5,1)[0]:
                            if features.objects[ext_x][ext_y].size == features.target.size:
                                obs_space[ext_x][ext_y] = 1
                        else:
                            obs_space[ext_x][ext_y] = 0

        #for TEXT.
        if feature == FEATURE_TEXT:
            for ext_x in range(0, N_ROWS, 1):
                for ext_y in range(0, N_COLS, 1):
                    e = self.get_eccentricity(features.x[x][y], features.y[x][y],
                                features.x[ext_x][ext_y], features.y[ext_x][ext_y])
                    mu = 0.05 + (0.1*e) + (0*e*e) + (0.05*e*e*e)
                    if features.objects[ext_x][ext_y].color != "BLANK":
                        if .26 > np.random.normal(mu,1.0,1)[0]:
                            if features.objects[ext_x][ext_y].text == features.target.text:
                                obs_space[ext_x][ext_y] = 1
                        else:
                            obs_space[ext_x][ext_y] = 0

        return obs_space

    def get_eccentricity(self, fix_x, fix_y, ext_x, ext_y):
        return distance.euclidean([fix_x, fix_y], [ext_x, ext_y])


gen = DisplayGenerator()

env = gen.sample()

space = np.ones((N_ROWS, N_COLS)) * -1
print env.target
for ext_x in range(0, N_ROWS, 1):
    for ext_y in range(0, N_COLS, 1):
        if env.objects[ext_x][ext_y].color == env.target.color:
            space[ext_x][ext_y] = 1

print space

model = ObservationModel()

col, shp, size, txt = model.sample(0, env)
print col
col, shp, size, txt = model.sample(10, env)
print col
col, shp, size, txt = model.sample(38, env)
print col
col, shp, size, txt = model.sample(70, env)
print col
col, shp, size, txt = model.sample(76, env)

print col