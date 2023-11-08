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

    def findNormalisationFactor(self,):
        """
        Find integral of pdf 
        """
        
        # Define integration limits
        normalisationLimits = (self.pdf.boundMin, self.pdf.boundMax)

        return self.pdf.integrate(normalisationLimits)

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

    def evaluateNull(self, slope, intercept):
        """
        evaluate negative log likelihood statisctic for passed parameters
        """

        # set new parameters
        self.pdf.setParameters(slope=slope, intercept=intercept)

        # compute likelyhood using passed parameters
        normalisation = self.pdf.integrate((self.pdf.boundMin, self.pdf.boundMax))
        likelihood = self.pdf._evaluate(self.data,) / normalisation 
        # set any negative likelihoods to neglegable positive values
        if (likelihood <= 0).any():
            likelihood[likelihood<=0] = 1e-6
        loglikelihood = np.log(likelihood)
        return -loglikelihood.sum()

    def evaluateAlternative(self, signalFraction, slope, intercept):
        """
        evaluate negative log likelihood statisctic for passed parameters
        """

        # set new parameters
        self.pdf.setParameters(signalFraction, slope=slope, intercept=intercept)

        # compute likelyhood using passed parameters
        normalisation = self.pdf.integrate((self.pdf.boundMin, self.pdf.boundMax))
        likelihood = self.pdf._evaluate(self.data,) / normalisation
        # set any negative likelihoods to neglegable positive values
        if (likelihood <= 0).any():
            likelihood[likelihood<=0] = 1e-6
        loglikelihood = np.log(likelihood)
        return -loglikelihood.sum()
    
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

        # Set new parameters
        self.pdf.setParameters(signalFraction=signalFraction, slope=slope, intercept=intercept)

        # Compute predicted value by model
        predicted_data = self.pdf._evaluate(self.data,)

        return (predicted_data-self.data)**2/self.dataUncertanty

class ChiSquaredModified(MinimisationStatistic):
    """
    Class containing chi suqared equivalent of the log likelyhood statistic for optimisation
    """

    def __init__(self, pdf, data, nBins=100):

        # Initialise parent class
        super().__init__(pdf, data)

        # Define class members
        self.nObservedMeasurments, bins , _ = plt.hist(data, bins=nBins, density=True)
        self.massBins = ( bins + (bins[1]-bins[0])/2 )[:-1]          # Center-point value of bins (Eliminate last element)


    def evaluateNull(self, slope, intercept):
        """
        Evaluate modified chi squared
        """
        # Set new parameters
        self.pdf.setParameters(slope=slope, intercept=intercept)

        # Compute normalised expected measurments
        nExpectedMeasurments = self.pdf._evaluate(self.massBins,) / self.findNormalisationFactor()

        # limit outputs of Expected obs to positive
        if (nExpectedMeasurments <= 0).any():
            nExpectedMeasurments[nExpectedMeasurments<=0] = 1e-3

        summand = nExpectedMeasurments - self.nObservedMeasurments + self.nObservedMeasurments*np.log(self.nObservedMeasurments/nExpectedMeasurments)

        if (summand == np.nan).any():
            isNan=True

        return 2*summand.sum()

    def evaluateAlternative(self, signalFraction, slope, intercept):
        """
        Evaluate modified chi squared
        """

        # Set new parameters
        self.pdf.setParameters(signalFraction=signalFraction, slope=slope, intercept=intercept)

        # Compute normalised expected measurments
        nExpectedMeasurments = self.pdf._evaluate(self.massBins,) / self.findNormalisationFactor()

        # limit outputs of Expected obs to positive
        if (nExpectedMeasurments <= 0).any():
            nExpectedMeasurments[nExpectedMeasurments<=0] = 1e-3

        summand = nExpectedMeasurments - self.nObservedMeasurments + self.nObservedMeasurments*np.log(self.nObservedMeasurments/nExpectedMeasurments)

        if (summand == np.nan).any():
            isNan=True

        return 2*summand.sum()


if __name__ == "__main__":

    pass