import numpy as np
import os

class DataIO(object):

    def __init__(self):
        self.directory = os.getcwd()

    def readNumpy(self, filename):
        # Check if filename has a backslash at sart of string and include to directory if not
        if filename[0] != '/':
            self.directory += '/'

        fileDirectory = self.directory + filename
        return np.loadtxt(fileDirectory, dtype=np.float32)
