import numpy as np
import matplotlib.pyplot as plt

def compute_energy_difference(filename):
    # Import ntuple files and take mean energy of all particles
    energy_data = np.loadtxt(filename, delimiter=",", comments="#")

    averaged_energy_data = energy_data.T.mean(axis=1)
    
    # Find true energy
    true_energy = averaged_energy_data[0]

    # Compute calibration constant for each layer
    calibration_constants = 1 / (averaged_energy_data[1:] / true_energy)

    # Compute calibrated energies
    calibrated_energies = (energy_data[:, 1:] * calibration_constants).flatten()

    # Remove all zero energies
    calibrated_energies = calibrated_energies[calibrated_energies!=0.0]

    # Compute the energy difference
    energy_difference = (calibrated_energies - true_energy) / true_energy

    return energy_difference

# Define global parameters
DATA_DIRECTORY  = "./data/2d_hist_data/"
ENERGY_RANGE    = [10000 + increment for increment in range(0, 95000, 5000)]

# Make list containing the filename of each data file
filenames = [DATA_DIRECTORY + "electron_" + str(energy) + "mev_1000.csv" for energy in ENERGY_RANGE]

for idx, filename in enumerate(filenames):
    if idx == 0:
        energy_differences = compute_energy_difference(filename)
        continue
    energy_differences = np.dstack((energy_differences, compute_energy_difference(filename)))

print(energy_differences)
    