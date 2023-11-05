import numpy as np
import math as m
import random
from utils import *
import matplotlib.pyplot as plt

class Linear(object):
    """
    Class that will generate a random value according to a linear distribution using a box method
    """

    def __init__(self, slope, intercept, bounds):

        if (not isinstance(bounds, tuple)):
            raise TypeError("Variable bound must be a tuple with the form (boundMin, boundMax)")
        if (not len(bounds) == 2):
            raise ValueError("Variable bound must have form (boundMin, boundMax)")
        
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

class Gaussian(object):
    """
    Class that will generate a random value according to a gaussian distribution using numpy.random.normal
    """

    def __init__(self, mean, sigma, bounds):

        if (not isinstance(bounds, tuple)):
            raise TypeError("Variable bound must be a tuple with the form (boundMin, boundMax)")
        if (not len(bounds) == 2):
            raise ValueError("Variable bound must have form (boundMin, boundMax)")
        
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
        NOTE: Returns un-normalised values between the bounds
        """
        return 1/(self.sigma * np.sqrt(2.0*np.pi)) * np.exp( -(x-self.mean)**2 / (2.0 * self.sigma**2) )
    
    def next(self,):
        """
        Generate a single random variable according to the class' distribution using numpy.random.normal method
        Will return and append generated variable to mass list.
        """

        x = np.random.normal(self.mean, self.sigma, size=1)
        self.mass.append(x)
        return x

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


def single_toy(mean, sigma, slope, intercept, bounds, n_events_signal=300, n_events_background=10000, n_bins=100):
    """
    Generate a toy dataset of a gaussian signal with linear background
    """
        
    if not isinstance(bounds, tuple):
        raise TypeError("Variable bound must be a tuple with the form (boundMin, boundMax)")
    if not len(bounds) == 2:
        raise ValueError("Variable bound must have form (boundMin, boundMax)")

    n_events_total = n_events_signal + n_events_background
    signal_fraction = n_events_signal/n_events_total
    # Generate pdf object
    pdf = SignalWithBackground(mean, sigma, slope, intercept, bounds, signalFraction=signal_fraction)

    # Generate random events
    for _ in range(n_events_total): pdf.next()

    # Get events for each distribution
    data = pdf.mass
    signal_data = pdf.massSignal
    background_data = pdf.massBackground
    
    axes_data = [signal_data, background_data, data]
    axes_titles = [
        f"Signal distribution ({n_events_signal}  events)",
        f"Background distribution ({n_events_background}  events)",
        f"Overall distribution ({n_events_total}  events)",
    ]

    # Plot data
    fig, axes = plt.subplots(3, 1, sharex="col")

    for idx in range(len(axes)):
        axes[idx].hist(axes_data[idx], bins = n_bins)
        axes[idx].set_title(axes_titles[idx])
    
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    single_toy(10, 0.5, -1, 20, (0, 20),)







