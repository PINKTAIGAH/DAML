from SimulateMuonLifetime import SimulateMuonLifetime
import matplotlib.pyplot as plt

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

    muonSimulation = SimulateMuonLifetime(MU_LIFETIME_TRUTH, LIMITS_1)
    


