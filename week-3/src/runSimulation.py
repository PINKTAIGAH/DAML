from SimulateMuonLifetime import SimulateMuonLifetime
import matplotlib.pyplot as plt
import numpy as np

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
    TIME = np.linspace(LIMITS_1, 1000)

    # Initualise simulatuon class
    muonSimulation = SimulateMuonLifetime(MU_LIFETIME_TRUTH, LIMITS_1)

    # Get 1000 random ramples of muon lifetime
    sampledMuonLifetimes = muonSimulation.sampleLifetimes(NUM_SAMPLES)
    print(f"The simulated average muon lifetime is {muonSimulation.computeEstimatedLifetime():.3f} mu s")

    # Plot distribution of sampled lifetimes with predicted distribution
    plt.hist(sampledMuonLifetimes, bins=100, density=True, label="Sampled distribution")
    plt.plot(Time, muonSimulation.exponantialTruth(TIME), label="Predicted distribution") 
    plt.xlabel("Muon Lifetime, t ($\mu s$)")
    plt.ylabel("Counts")
    plt.title("Distribution of randomly sampled muon lifetimes")
    plt.legend()
    plt.show()

    # Simulate multiple average muon lifetime
    simulatedMuonLifetimes = 
    


