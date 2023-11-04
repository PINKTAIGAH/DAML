import numpy as np


class Exponential(object):

    def __init__(self, bounds):
        
        if not isinstance(bounds, tuple):
            raise Exception("Variable bounds must be tuple of the form (boundsLow, boundsHigh)")
        
        self.boundLow, self.boundHigh = bounds
    
    def setParameters(self, parameters):
        
        if not hasattr(parameters, "__len__"):
            raise Exception("Variable parameter must be Arraylike with form [tau,]")
        if len(parameters) != 1:
            raise Exception("Variable parmater must have lenth 3 with form [tau,]")
        
        self.tau, = parameters

    def _computeNormalisation(self,):
        
        integral = self.tau * ( np.exp(-self.boundLow/self.tau) - np.exp(-self.boundHigh/self.tau) )
        return 1/integral

    def evaluate(self, time):
        
        normalisationFactor = self._computeNormalisation()
        return normalisationFactor * np.exp(-time/self.tau)


class ExponentialResonance(object):

    def __init__(self, bounds, sigma=0.2, completeNormalisation=False):

        if not isinstance(bounds, tuple):
            raise Exception("Variable bounds must be tuple of the form (boundsLow, boundsHigh)")
        
        self.boundLow, self.boundHigh = bounds
        self.sigma = sigma

    def setParameters(self, parameters):
        
        if not hasattr(parameters, "__len__"):
            raise Exception("Variable parameter must be Arraylike with form [fraction, exponentialParameter, resonanceMean,]")
        if len(parameters) != 3:
            raise Exception("Variable parmater must have lenth 3 with form [fraction, exponentialParameter, resonanceMean,]")
        
        # Fraction, Exponential, ResonanceMean
        self.p0, self.p1, self.p2 = parameters

    def _computeNormalisation(self,):
        normalisationExponential = self.p1 * ( np.exp(-self.boundLow/self.p1) - np.exp(-self.boundHigh/self.p1) )
        return normalisationExponential

    def evaluate(self, data,):
        normalisationFactor = self._computeNormalisation()
        pdf1 = (self.p0 * np.exp(-data/self.p1))
        pdf2 = (1-self.p0) * ( 1/(self.sigma*np.sqrt(2*np.pi)) ) * np.exp( -0.5 * ( (data-self.p2)/self.sigma )**2 ) 

        return pdf1/normalisationFactor + pdf2
