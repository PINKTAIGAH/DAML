import numpy as np
from scipy.optimize import minimize

class MaximumLikelyhoodFitter(object):

    def __init__(self, minimiser=None,):
        
        self.minimiser = str(minimiser)
       
    def _normaliseExponential(self, tau):

        boundLow, boundHigh = self.bounds
        integral = tau * ( np.exp(-boundLow/tau) - np.exp(-boundHigh/tau) )
        return 1/integral

    def _negLogLikelyhood(self, p, time):
        tau, = p
        normalisationFactor = self._normaliseExponential(tau)

        logPDF = np.log(normalisationFactor * np.exp(-time/tau))
        logLikelyhood = logPDF.sum()
        return -logLikelyhood

    def run(self, data, p0, bounds, verbose=True):
        assert isinstance(bounds, tuple), "Variable bounds is not a tuple containing minimum & maximum bound"

        self.bounds = bounds
        self.data = data
        self.p0 = p0
        self.verbose = verbose
        
        match self.minimiser:
            
            case "minuit":
                self._fitWithMinuit()
            case "scipy":
                self._fitWithScipy()
            case _:
                raise Exception("Flag provided for minimiser not recognised. Use 'minuit' or 'scipy'.")
        
        return self.results.x


    def _fitWithMinuit(self,):
        pass
    
    def _fitWithScipy(self,):
        self.results = minimize(self._negLogLikelyhood, self.p0, args=(self.data), method="nelder-mead") 
        if self.verbose:
            print(self.results)
        
