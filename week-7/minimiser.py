import numpy as np
from scipy.optimize import minimize
from iminuit import Minuit
from negativeLogLikelihood import NegativeLogLikelihood

class ScipyFitter(object):

    def __init__(self, data, pdf,):
        
        self.negLogLikelihood = NegativeLogLikelihood(pdf, data, usingScipy=True)
    
    def fit(self, p0, verbose=True, method="nelder-mead"):

        results = minimize(self.negLogLikelihood.evaluate, p0, method=method)
        if verbose:
            print(results)
        return results

class MinuitFitter(object):

    def __init__(self, data, pdf,):
        
        self.negLogLikelihood = NegativeLogLikelihood(pdf, data)
    
    def fit(self, p0Dictionary, errorDef, limitDictionary, verbose=True):

        forcedParameters = list(p0Dictionary.keys())
        limitVariableNames = ["limit_" + paramName for paramName in forcedParameters]

        m = Minuit(self.negLogLikelihood.evaluate, **p0Dictionary, name=forcedParameters)   
        
        result = m.migrad()
        if verbose:
            print(result)

        return result
