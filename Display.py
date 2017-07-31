class Display(object):

    def __init__(self, objects, target, x, y):
        self.objects = objects
        self.target = target
        self.x = x
        self.y = y

    def __repr__(self):
        return "--OBJECTS:-- \n" + str(self.objects) + "\n --TARGET:-- \n" + str(self.target) \
               + "\n --X-COORD:-- \n" + str(self.x) + "\n --Y-COORD-- \n" + str(self.y)