import numpy as np
import math as m
import random
from utils import find_max
from scipy.integrate import quad

class Linear(object):
    """
    Class that will generate a random value according to a linear distribution using a box method
    """

    def __init__(self, slope, intercept, bounds):

        if (not isinstance(bounds, tuple)):
            raise TypeError("Variable bound must be a tuple with the form (boundMin, boundMax)")
        if (not len(bounds) == 2):
            raise ValueError("Variable bound must have form (boundMin, boundMax)")
        if (not bounds[0] < bounds[1]):
            raise ValueError("First element in tuple must be smaller than second")
        
        # Initialise class variables
        self.intercept = intercept
        self.slope = slope
        self.boundMin, self.boundMax = bounds
        # Initialise list to hold randomly generated mass values
        self.mass = []
        # Find maximum value of the distribution within the bounds
        self.maxValue = find_max(self._evaluate, self.boundMin, self.boundMax)

    def _evaluate(self, x):
        """
        Evaluate the linear function of the distribution
        NOTE: Returns un-normalised values
        """
        return self.intercept + self.slope * x

    def integrate(self, limits):
        """
        Evaluate the integral of the linear function within the specified bounds
        ##### NOTE: Integral is not normalised within the specified bounds of the class #####
        """

        if (not isinstance(limits, tuple)):
            raise TypeError("Variable bound must be a tuple with the form (boundMin, boundMax)")
        if (not len(limits) == 2):
            raise ValueError("Variable bound must have form (boundMin, boundMax)")
        if (not limits[0] < limits[1]):
            raise ValueError("First element in tuple must be smaller than second")
        if (not limits[0] >= self.boundMin):
            raise ValueError("Lower integral limit must be larger than lower bound of pdf")
        if (not limits[1] <= self.boundMax):
            raise ValueError("Higher integral limit must be smaller than upper bound of pdf")
    
        limitLow, limitHigh = limits
        integralResult, IntegralError = quad(self._evaluate, limitLow, limitHigh) 
        return integralResult
    
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

    def getMass(self,):
        """
        Return numpy array containing all generated values
        """

        return np.array(self.mass)

class Gaussian(object):
    """
    Class that will generate a random value according to a gaussian distribution using numpy.random.normal
    """

    def __init__(self, mean, sigma, bounds):

        if (not isinstance(bounds, tuple)):
            raise TypeError("Variable bound must be a tuple with the form (boundMin, boundMax)")
        if (not len(bounds) == 2):
            raise ValueError("Variable bound must have form (boundMin, boundMax)")
        if (not bounds[0] < bounds[1]):
            raise ValueError("First element in tuple must be smaller than second")
        
        # Initialise class variables
        self.mean = mean
        self.sigma = sigma
        self.boundMin, self.boundMax = bounds
        # Initialise list to hold randomly generated mass values
        self.mass = []
        # Find maximum value of the distribution within the bounds
        self.maxValue = find_max(self._evaluate, self.boundMin, self.boundMax)

    def _evaluate(self, x):
        """
        Evaluate the gaussian function of the distribution
        ##### NOTE: Returns un-normalised values between the bounds #####
        """
        return 1/(self.sigma * np.sqrt(2.0*np.pi)) * np.exp( -(x-self.mean)**2 / (2.0 * self.sigma**2) )

    def integrate(self, limits):
        """
        Evaluate the integral of the gaussian function within the specified bounds
        ##### NOTE: Integral is not normalised within the specified bounds of the class #####
        """

        if (not isinstance(limits, tuple)):
            raise TypeError("Variable bound must be a tuple with the form (boundMin, boundMax)")
        if (not len(limits) == 2):
            raise ValueError("Variable bound must have form (boundMin, boundMax)")
        if (not limits[0] < limits[1]):
            raise ValueError("First element in tuple must be smaller than second")
        if (not limits[0] >= self.boundMin):
            raise ValueError("Lower integral limit must be larger than lower bound of pdf")
        if (not limits[1] <= self.boundMax):
            raise ValueError("Higher integral limit must be smaller than upper bound of pdf")
    
        limitLow, limitHigh = limits
        integralResult, IntegralError = quad(self._evaluate, limitLow, limitHigh) 
        return integralResult
    
    def next(self,):
        """
        Generate a single random variable according to the class' distribution using numpy.random.normal method
        Will return and append generated variable to mass list.
        """

        # Use .item() to append vsariable inside array and not array itself
        x = np.random.normal(self.mean, self.sigma, size=1).item()
        self.mass.append(x)
        return x

    def getMass(self,):
        """
        Return numpy array containing all generated values
        """

        return np.array(self.mass)

class SignalWithBackground(object):
    """
    Class that will generate a random distribution consisting of gaussian signal with a linear background signal
    """

    def __init__(self, mean, sigma, slope, intercept, bounds, signalFraction,):
        
        if (not signalFraction > 0.0) and (not signalFraction < 1.0):
            raise ValueError("Variable signal fraction must be a float within the bounds 0.0 <= signalFraction <= 1.0")
        if not isinstance(bounds, tuple):
            raise TypeError("Variable bound must be a tuple with the form (boundMin, boundMax)")
        if not len(bounds) == 2:
            raise ValueError("Variable bound must have form (boundMin, boundMax)")
        if (not bounds[0] < bounds[1]):
            raise ValueError("First element in tuple must be smaller than second")

        self.signalFraction = signalFraction
        # Initialise lists to hold randomly generated values
        self.massBackground = []
        self.massSignal = []
        self.mass = []
    
        # Initialise pdf objects
        self.signal = Gaussian(mean, sigma, bounds)
        self.background = Linear(slope, intercept, bounds)

    def _evaluate(self, x):
        """
        Evaluate the function of the distribution
        NOTE: Returns un-normalised values between the bounds
        """
        return self.signalFraction*self.signal._evaluate(x) + (1-self.signalFraction)*self.background._evaluate(x)

    def integrate(self, limits):
        """
        Evaluate the integral of the signal with background function within the specified bounds
        ##### NOTE: Integral is not normalised within the specified bounds of the class #####
        """

        if (not isinstance(limits, tuple)):
            raise TypeError("Variable bound must be a tuple with the form (boundMin, boundMax)")
        if (not len(limits) == 2):
            raise ValueError("Variable bound must have form (boundMin, boundMax)")
        if (not limits[0] < limits[1]):
            raise ValueError("First element in tuple must be smaller than second")
        if (not limits[0] >= self.boundMin):
            raise ValueError("Lower integral limit must be larger than lower bound of pdf")
        if (not limits[1] <= self.boundMax):
            raise ValueError("Higher integral limit must be smaller than upper bound of pdf")
    
        limitLow, limitHigh = limits
        integralResult, IntegralError = quad(self._evaluate, limitLow, limitHigh) 
        return integralResult

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

    def getMass(self,):
        """
        Return numpy array containing all generated values for overall distribution
        """

        return np.array(self.mass)

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