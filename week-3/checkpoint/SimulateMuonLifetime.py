"""
Class which is capable of computing and retuning simulated lifetimes of muon.
Method of lifetime simulation uses the box method with a triangular and square 
shape.
"""

import numpy as np
import matplotlib.pyplot as plt

class SimulateMuonLifetime(object):

    def __init__(self, muLifetimeTruth, numSimulations, distributionLimits=(0, 10)):
        #=======================================================================
        # Class constructor. Distribution limits are in units of microseconds 
    
        # Define the inverse lifetime for easier computations
        self.muLifetimeTruth = muLifetimeTruth            
        self.lowLimit, self.highLimit = distributionLimits
        self.numSimulations = numSimulations
        self.randomLifetimesList = []
        self.randomLifetimesArray = np.array([])

    def exponantialTruth(self, x, lifetime):
        #=======================================================================
        # Return output of exponential finction using specified decay rate

        return (1/lifetime)*np.exp(-x/lifetime)
    
    def simulateSingleRandomLifetime(self,):
        #=======================================================================
        # Simulate the lifetime of a single muon 
        
        return np.random.exponential(self.muLifetimeTruth)
    
    def wipeRandomLifetimeList(self,):
        #=======================================================================
        # Remove entries from random lifetime list

        self.randomLifetimesList = []

    def simulateLifetimes(self,):
        #=======================================================================
        # Return an array containing all the desired randomised muon lifetimes

        if len(self.randomLifetimesList) != 0:
            raise Exception("self.randomLifetimeList is not empty. Wipe entries before inititing a new simulation")

        # iterate over all desired simulation points
        while True:
            randomLifetimeCanditade = self.simulateSingleRandomLifetime()
            # Check if generated muon lifetime is within the desired bounds
            if (self.lowLimit < randomLifetimeCanditade) and (randomLifetimeCanditade<self.highLimit):
                self.randomLifetimesList.append(randomLifetimeCanditade)
            
            if len(self.randomLifetimesList) >= self.numSimulations:
                # Create object containing simulated lifetime of muons
                self.randomLifetimesArray = np.array(self.randomLifetimesList)
                # Wipe data from estimated lifetime list for a subsequent simulation
                self.wipeRandomLifetimeList()
                break

        return self.randomLifetimesArray
    
    def computeEstimatedLifetime(self,):
        #=======================================================================
        # Compute the overall simulated lifetime of a muon using simulated 
        # As muon lifetime is drawn from an exponential PDF, decay rate is computed by computing
        # average value        

        if (self.randomLifetimesArray.size == 0):
            raise Exception("The random lifetims array has not been initialised.")
        return self.randomLifetimesArray.mean()




def test():
    N = 1000
    simulation = SimulateMuonLifetime(2.2, N)
    simulatedLifetimes = simulation.simulateLifetimes()
    print(f"Simulated average muon lifetime for single simulation run is {simulation.computeEstimatedLifetime():.3f} mus")

    plt.hist(simulatedLifetimes, bins=100, density=True)
    plt.plot(np.linspace(0, 10, N), simulation.exponantialTruth(np.linspace(0, 10, N), 2.2))
    plt.show()



if __name__ == "__main__":
    test()