import numpy as np
import matplotlib.pyplot as plt

def compute_energy_difference(filename):
    # Import ntuple files and take mean energy of all particles
    electron_energy_data = np.loadtxt(filename, delimiter=",", comments="#")

    electron_averaged_energy_data = electron_energy_data.T.mean(axis=1)
    
    # Find true energy
    true_energy = electron_averaged_energy_data[0]

    # Compute calibration constant for each layer
    electron_calibration_constants = 1 / (electron_averaged_energy_data[1:] / true_energy)

    # Compute calibrated energies
    electron_calibrated_energies = (electron_energy_data[:, 1:] * electron_calibration_constants).flatten()

    # Remove all zero energies
    electron_calibrated_energies = electron_calibrated_energies[electron_calibrated_energies!=0.0]

    # Compute the energy difference
    electron_energy_difference = (electron_calibrated_energies - true_energy) / true_energy

    return electron_energy_difference

# Make list containing the filename of each data file
