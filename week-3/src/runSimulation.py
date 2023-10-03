from SimulateMuonLifetime import SimulateMuonLifetime
import matplotlib.pyplot as plt
import numpy as np
from utils import normalize



def main():
    """
    Parameters
    """
    MU_LIFETIME_TRUTH = 2.2         # in microseconds
    NUM_SAMPLES = 1000
    NUM_SIMULATIONS = 500
    LIMITS_1 = (0, 10)
    LIMITS_2 = (0, 5)
    LIMITS_3 = (0, 100)
    TIME = np.linspace(*LIMITS_1, 1000)

    # Initualise simulatuon class
    muonSimulation = SimulateMuonLifetime(MU_LIFETIME_TRUTH, LIMITS_1)

    # Get 1000 random ramples of muon lifetime
    sampledMuonLifetimes = muonSimulation.sampleLifetimes(NUM_SAMPLES)
    print(f"The simulated average muon lifetime with range 0 < t < 10 is {muonSimulation.computeEstimatedLifetime():.3f} mu s")

    # Plot distribution of sampled lifetimes with predicted distribution
    plt.hist(sampledMuonLifetimes, bins=100, label="Sampled distribution", density=True)
    plt.plot(TIME, muonSimulation.exponantialTruth(TIME, MU_LIFETIME_TRUTH),label="Predicted distribution") 
    plt.xlabel("Muon Lifetime, t ($\mu s$)")
    plt.ylabel("Probability")
    plt.xlim(LIMITS_1)
    plt.title("Distribution of randomly sampled muon lifetimes")
    plt.legend()
    plt.savefig("../figures/sampled_distribution.png")
    plt.show()
    
    """
    Simulate multiple average muon lifetime
    """

    # Initialise other muon simulation classes with different limits
    muonSimulationLowLimits = SimulateMuonLifetime(MU_LIFETIME_TRUTH, LIMITS_2)
    muonSimulationHighLimits = SimulateMuonLifetime(MU_LIFETIME_TRUTH, LIMITS_3)
    
    simulatedLifetimesNormalLimits = muonSimulation.simulateMultipleMuonLifetimes(
        NUM_SAMPLES, 
        NUM_SIMULATIONS
    )
    simulatedLifetimesLowLimits = muonSimulationLowLimits.simulateMultipleMuonLifetimes(
        NUM_SAMPLES, 
        NUM_SIMULATIONS
    )
    simulatedLifetimesHighLimits = muonSimulationHighLimits.simulateMultipleMuonLifetimes(
        NUM_SAMPLES, 
        NUM_SIMULATIONS
    )

    # Place all simulated datasets in a list to allow for them to be looped over

    overallSimulatedDataset = [
        simulatedLifetimesNormalLimits,
        simulatedLifetimesLowLimits,
        simulatedLifetimesHighLimits
    ]
    overallLabels = [
        "limits: 0 < t < 10",
        "limits: 0 < t < 5",
        "limits: 0 < t < 100"
    ]

    yMax = 0        # Initialising variable for use in loop    
    
    # Plot histogram of simulated muon lifetimes with different limits
    for i in range(len(overallSimulatedDataset)):
        binHights, _, _ = plt.hist(overallSimulatedDataset[i], bins=50, label=overallLabels[i])
        currentYMax = binHights.max()
        yMax = currentYMax if (yMax < currentYMax) else yMax

    
    plt.xlabel("Simulated muon lifetimes, t ($\mu s)")
    plt.ylabel("Probability")
    plt.title("Probability distribution function of simulated muon lifetimes")
    plt.vlines(MU_LIFETIME_TRUTH, 0, yMax, colors="k", linestyles="dashed", label="True muon lifetime" )
    plt.legend()
    plt.savefig("../figures/simulated_distribution.png")
    plt.show()
    
if __name__ == "__main__":
    main()


