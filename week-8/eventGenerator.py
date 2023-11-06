import numpy as np
import math as m
import random
from utils import find_max
from scipy.integrate import quad

class ProbabilityDensityFunction(object):
    """
    Parent class containing common methods and members to be used by all pdf classes
    """

    def __init__(self, bounds):

        if (not isinstance(bounds, tuple)):
            raise TypeError("Variable bound must be a tuple with the form (boundMin, boundMax)")
        if (not len(bounds) == 2):
            raise ValueError("Variable bound must have form (boundMin, boundMax)")
        if (not bounds[0] < bounds[1]):
            raise ValueError("First element in tuple must be smaller than second")
        
        # Initialise class variables
        self.boundMin, self.boundMax = bounds
        # Initialise list to hold randomly generated mass values
        self.mass = []

    def integrate(self, limits):
        """
        Evaluate the integral of the pdf within the specified bounds
        ##### NOTE: Integral is not normalised within the specified bounds of the class #####
        """

        if (not isinstance(limits, tuple)):
            raise TypeError("Variable bound must be a tuple with the form (limitMin, limitMax)")
        if (not len(limits) == 2):
            raise ValueError("Variable bound must have form (limitMin, limitMax)")
        if (not limits[0] < limits[1]):
            raise ValueError("First element in tuple must be smaller than second")
        if (not limits[0] >= self.boundMin):
            raise ValueError("Lower integral limit must be larger than lower bound of pdf")
        if (not limits[1] <= self.boundMax):
            raise ValueError("Higher integral limit must be smaller than upper bound of pdf")
    
        limitLow, limitHigh = limits
        integralResult, IntegralError = quad(self._evaluate, limitLow, limitHigh) 
        return integralResult
    
    def getMass(self,):
        """
        Return numpy array containing all generated values
        """

        return np.array(self.mass)

class Linear(ProbabilityDensityFunction):
    """
    Class that will generate a random value according to a linear distribution using a box method
    """

    def __init__(self, slope, intercept, bounds):

        # Initialise parent class
        super().__init__(bounds)

        # Initialise class variables
        self.intercept = intercept
        self.slope = slope
        # Find maximum value of the distribution within the bounds
        self.maxValue = find_max(self._evaluate, self.boundMin, self.boundMax)

    def _evaluate(self, x, slope=None, intercept=None):
        """
        Evaluate the linear function of the distribution
        NOTE: Returns un-normalised values
        """

        # Use default values for parameters of none are passed through kwargs
        if slope == None:               slope = self.slope
        if intercept == None:           intercept = self.intercept       

        return self.intercept + self.slope * x

    def next(self,):
        """
        Generate a single random variable according to the class' distribution using the box method.
        Will return and append generated variable to mass list.
        """

        # Iterate untill a random value within the distribution is generated
        while True:
            # Generate random x variable within the distribution's bounds
            x = random.uniform(self.boundMin, self.boundMax)
            # Evaluate y value for x and the maximum y value allowed by the system
            y1 = self._evaluate(x)
            y2 = random.uniform(0, self.maxValue)
            # Accept x if y2 is less than y1
            if (y2<y1):
                filteredX = x
                self.mass.append(filteredX)
                return filteredX

class Gaussian(ProbabilityDensityFunction):
    """
    Class that will generate a random value according to a gaussian distribution using numpy.random.normal
    """

    def __init__(self, mean, sigma, bounds):

        # Initialise parent class
        super().__init__(bounds)
        
        # Initialise class variables
        self.mean = mean
        self.sigma = sigma
        # Find maximum value of the distribution within the bounds
        self.maxValue = find_max(self._evaluate, self.boundMin, self.boundMax)

    def _evaluate(self, x, mean=None, sigma=None):
        """
        Evaluate the gaussian function of the distribution
        ##### NOTE: Returns un-normalised values between the bounds #####
        """

        # Use default values for parameters of none are passed through kwargs
        if mean == None:                mean = self.mean
        if sigma == None:               sigma = self.sigma

        return 1/(sigma * np.sqrt(2.0*np.pi)) * np.exp( -(x-mean)**2 / (2.0 * sigma**2) )

    def next(self,):
        """
        Generate a single random variable according to the class' distribution using numpy.random.normal method
        Will return and append generated variable to mass list.
        """

        # Use .item() to append vsariable inside array and not array itself
        x = np.random.normal(self.mean, self.sigma, size=1).item()
        self.mass.append(x)
        return x

class SignalWithBackground(ProbabilityDensityFunction):
    """
    Class that will generate a random distribution consisting of gaussian signal with a linear background signal
    """

    def __init__(self, mean, sigma, slope, intercept, bounds, signalFraction,):
        
        # Initialise parent class
        super().__init__(bounds)

        self.signalFraction = signalFraction
        # Initialise lists to hold randomly generated values
        self.massBackground = []
        self.massSignal = []
    
        # Initialise pdf objects
        self.signal = Gaussian(mean, sigma, bounds)
        self.background = Linear(slope, intercept, bounds)

    def _evaluate(
        self, x, signalFraction=None, mean=None, sigma=None, slope=None, intercept=None
    ):
        """
        Evaluate the function of the distribution
        NOTE: Returns un-normalised values between the bounds
        """

        # Use default values for parameters of none are passed through kwargs
        if signalFraction == None:      signalFraction = self.signalFraction
        if mean == None:                mean = self.singal.mean
        if sigma == None:               sigma = self.signal.sigma
        if slope == None:               slope = self.background.slope
        if intercept == None:           intercept = self.background.intercept       

        return self.signalFraction*self.signal._evaluate(x, mean=mean, sigma=sigma) + \
                (1-self.signalFraction)*self.background._evaluate(x, slope=slope, intercept=intercept)

    def next(self,):
        """
        Generate a single random variable according to the class' distribution using numpy.random.normal method
        Will return and append generated variable to mass list.
        """

        randomProbability = random.uniform(0.0, 1.0)

        if (randomProbability <= self.signalFraction):
            # Draw x from Signal distribution
            filteredX = self.signal.next()
            self.massSignal.append(filteredX)
        else:
            # Draw x from background distribution
            filteredX = self.background.next()
            self.massBackground.append(filteredX)
        
        # Add filtered x to mass list irrespective of what distribution it was drawn from
        self.mass.append(filteredX)
        return filteredX

    def getMassBackground(self,):
        """
        Return numpy array containing all generated values for background distribution
        """

        return np.array(self.massBackground)

    def getMassSignal(self,):
        """
        Return numpy array containing all generated values for signal distribution
        """

        return np.array(self.massSignal)