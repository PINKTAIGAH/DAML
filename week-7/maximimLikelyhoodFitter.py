import numpy as np
import scipy.linalg as linalg

class MaximumLikelyhoodFitter(object):

    def __init__(self, minimiser=None, bounds=(0, np.inf)):
      
        self.minimiser = str(minimiser)
        self.bounds = bounds

    def negLogLikelyhood(self, )                
    

    def run(self,):
        match self.minimiser:
            
            case "minuit":
                self._fitWithMinuit()
            case "scipy":
                self._fitWithScipy()
            case _:
                raise Exception("Flag provided for minimiser not recognised. Use 'minuit' or 'scipy'.")


    def _fitWithMinuit(self,):
        pass
    
    def _fitWithScipy(self,):
        pass