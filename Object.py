class Object(object):

    def __init__(self, color, shape, size, text):

        self.color = color
        self.shape = shape
        self.size = size
        self.text = text

    def get_color(self):
        return self.color

    def get_shape(self):
        return self.shape

    def get_size(self):
        return self.size

    def get_text(self):
        return self.text

    def __repr__(self):
        return "[COLOUR: " + str(self.color) + ", SHAPE: " + str(self.shape) + ", SIZE: " + str(self.size) \
               + ", TEXT: " + str(self.text) + "]"