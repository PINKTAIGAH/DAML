"""
Class which is capable of computing and retuning simulated lifetimes of muon.
Method of lifetime simulation uses the box method with a triangular and square 
shape.
"""

import numpy as np
import matplotlib.pyplot as plt

class SimulateMuonLifetime(object):

    def __init__(self, muLifetimeTruth, distributionLimits=(0, 10)):
        #=======================================================================
        # Class constructor. Distribution limits are in units of microseconds 
    
    # Define the inverse lifetime for easier computations
        self.muLifetimeTruth = muLifetimeTruth            
        self.lowLimit, self.highLimit = distributionLimits
        self.randomLifetimesList = []
        self.randomLifetimesArray = np.array([])

    def exponantialTruth(self, x, lifetime):
        #=======================================================================
        # Return output of exponential finction using specified decay rate

        return (1/lifetime)*np.exp(-x/lifetime)
    
    def sampleSingleRandomLifetime(self,):
        #=======================================================================
        # Simulate the lifetime of a single muon 
        
        return np.random.exponential(self.muLifetimeTruth)
    
    def wipeRandomLifetimeList(self,):
        #=======================================================================
        # Remove entries from random lifetime list

        self.randomLifetimesList = []

    def sampleLifetimes(self, numSamples):
        #=======================================================================
        # Return an array containing all the desired randomised muon lifetimes

        if len(self.randomLifetimesList) != 0:
            raise Exception("self.randomLifetimeList is not empty. Wipe entries before inititing a new simulation")

        # iterate over all desired simulation points
        while True:
            randomLifetimeCanditade = self.sampleSingleRandomLifetime()
            # Check if generated muon lifetime is within the desired bounds
            if (self.lowLimit < randomLifetimeCanditade) and (randomLifetimeCanditade<self.highLimit):
                self.randomLifetimesList.append(randomLifetimeCanditade)
            
            if len(self.randomLifetimesList) >= numSamples:
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

    def simulateMultipleMuonLifetimes(self, numSamples, numSimulations):
        #=======================================================================
        # Run full simulation chain fro specified number of simulations and return
        # array containing results
        
        simulatedLifetimeList = []
        
        # Iterate over simulation chain for  desired events
        for _ in range(numSimulations):
            _ = self.sampleLifetimes(numSamples)
            simulatedLifetimeList.append(self.computeEstimatedLifetime())
        
        return np.array(simulatedLifetimeList)



def test():
    N = 1000
    muLifetimeTruth = 2.2
    simulationLimit5 = SimulateMuonLifetime(muLifetimeTruth, distributionLimits=(0,5))
    simulationLimit10 = SimulateMuonLifetime(muLifetimeTruth, distributionLimits=(0,10))
    simulationLimit100 = SimulateMuonLifetime(muLifetimeTruth, distributionLimits=(0,100))
    simulatedLifetimes = simulationLimit10.sampleLifetimes(N)
    print(f"Simulated average muon lifetime for single simulation run is {simulationLimit10.computeEstimatedLifetime():.3f} mus")

    plt.hist(simulatedLifetimes, bins=100, density=True)
    plt.plot(np.linspace(0, 10, N), 
             simulationLimit10.exponantialTruth(np.linspace(0, 10, N), muLifetimeTruth)
    )
    plt.show()

    multipleSimulatedLifetimes10 = []
    multipleSimulatedLifetimes100 = []
    multipleSimulatedLifetimes5 = []
    for _ in range(500):
        # limit 10
        _ = simulationLimit10.sampleLifetimes(N)
        multipleSimulatedLifetimes10.append(simulationLimit10.computeEstimatedLifetime())
    
        # limit 100
        _ = simulationLimit100.sampleLifetimes(N)
        multipleSimulatedLifetimes100.append(simulationLimit100.computeEstimatedLifetime())

        # limit 10
        _ = simulationLimit5.sampleLifetimes(N)
        multipleSimulatedLifetimes5.append(simulationLimit5.computeEstimatedLifetime())

    plt.hist(multipleSimulatedLifetimes10, bins = 50)
    plt.hist(multipleSimulatedLifetimes100, bins = 50)
    plt.hist(multipleSimulatedLifetimes5, bins = 50)
    plt.show()


if __name__ == "__main__":
    test()
