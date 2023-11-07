import numpy as np
import matplotlib.pyplot as plt

"""
########## Patent Classes ##########
"""

class MinimisationStatistic(object):
    """
    Class containing minimisation statistic to be for pdf fitting
    """

    def __init__(self, pdf, data):

        self.pdf = pdf
        self.data = data

    def setData(self, data):
        """
        Assign data class member to new dataset for the reuse of this class
        """
        
        self.data = data

"""
########## Child Classes ###########
"""

class NegativeLogLikelihood(MinimisationStatistic):
    """
    Class constaining Negative log likelihood statistic for optimisation
    """

    def __init__(self, pdf, data,):

        # Initialise parent class
        super().__init__(pdf, data)

    def evaluate(self, *fittingParameters):
        """
        Evaluate negative log likelihood statisctic for passed parameters
        """

        # Assign fitting parametes
        match len(fittingParameters):
            case 2: 
                slope, intercept = fittingParameters
                signalFraction = None
            case 3:
                signalFraction, slope, intercept = fittingParameters
            case _:
                raise ValueError("Variable fitting parameter has too many or too few elements. Should have 2 or 3")

        # Compute likelyhood using passed parameters
        likelihood = self.pdf._evaluate(self.data, signalFraction=signalFraction, slope=slope, intercept=intercept)
        # Set any negative likelihoods to neglegable positive values
        if (likelihood <= 0).any():
            likelihood[likelihood<=0] = 1e-6
        logLikelihood = np.log(likelihood)
        return -logLikelihood.sum()
    
class ChiSquared(MinimisationStatistic):
    """
    Class constaining chi squared statistic for optimisation
    """

    def __init__(self, pdf, data, dataUncertanty):

        # Initialise parent class
        super().__init__(pdf, data)
        # Define class members
        self.dataErrors = dataUncertanty

    def evaluate(self, *fittingParameters):
        """
        Evaluate chi squared statisctic for passed parameters
        """

        # Assign fitting parametes
        match len(fittingParameters):
            case 2: 
                slope, intercept = fittingParameters
                signalFraction = None
            case 3:
                signalFraction, slope, intercept = fittingParameters
            case _:
                raise ValueError("Variable fitting parameter has too many or too few elements. Should have 2 or 3")

        # Compute predicted value by model
        predicted_data = self.pdf._evaluate(self.data, signalFraction=signalFraction, slope=slope, intercept=intercept)

        return (predicted_data-self.data)**2/self.dataUncertanty

class ChiSquaredModified(MinimisationStatistic):
    """
    Class containing chi suqared equivalent of the log likelyhood statistic for optimisation
    """

    def __init__(self, pdf, data, nBins=100):

        # Initialise parent class
        super().__init__(pdf, data)

        # Define class members
        self.nObservedMeasurments, bins = plt.hist(self.pdf.mass, bins=nBins)
        self.massBins = bins + (bins[1]-bins[0])/2                         # Center-point value of bins

    def evaluate(self, *fittingParameters):
        """
        Evaluate modified chi squared
        """

        # Assign fitting parametes
        match len(fittingParameters):
            case 2: 
                slope, intercept = fittingParameters
                signalFraction = None
            case 3:
                signalFraction, slope, intercept = fittingParameters
            case _:
                raise ValueError("Variable fitting parameter has too many or too few elements. Should have 2 or 3")

        nExpectedMeasurments = self.pdf._evaluate(self.massBins, signalFraction=signalFraction, slope=slope, intercept=intercept)

        summand = nExpectedMeasurments - self.nObservedMeasurments + self.nObservedMeasurments*np.log(self.nObservedMeasurments/nExpectedMeasurments)

        return 2*summand.sum()

if __name__ == "__main__":

    pass