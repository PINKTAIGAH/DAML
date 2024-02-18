import numpy as np
import matplotlib.pyplot as plt

# Define dirtectory for data files
ELECTRON_DATA = {
    "histogram"     : "./data/electron_100/output_h1_LayerTotal.csv",
    "nTuple"        : "./data/electron_100/output_nt_Energy.csv",
}

NEUTRON_DATA = {
    "histogram"     : "./data/neutron_100/output_h1_LayerTotal.csv",
    "nTuple"        : "./data/neutron_100/output_nt_Energy.csv",
}

# Define simulation parameters
TRUE_ENERGY = 300.0   # MeV
DETECTOR_LAYERS = 6

# Import ntuple files and take mean energy of all particles
electron_energy_data = np.loadtxt(ELECTRON_DATA["nTuple"], delimiter=",", comments="#")
# neutron_energy_data = np.loadtxt(NEUTRON_DATA["nTuple"], delimiter=",", comments="#")

electron_averaged_energy_data = electron_energy_data.T.mean(axis=1)
# neutron_averaged_energy_data = neutron_energy_data.T.mean(axis=1)

# Assert that the true energy is correct
assert TRUE_ENERGY == electron_averaged_energy_data[0], "Average true energy of electron simulation is not consistent with global constant."
# assert TRUE_ENERGY == neutron_averaged_energy_data[0], "Average true energy of neutron simulation is not consistent with global constant."

# Compute calibration constant for each layer
electron_calibration_constants = 1 / (electron_averaged_energy_data[1:] / TRUE_ENERGY)
# neutron_calibration_constants = 1 / (neutron_averaged_energy_data[1:] / TRUE_ENERGY)

# Compute calibrated energies
electron_calibrated_energies = (electron_energy_data[:, 1:] * electron_calibration_constants).flatten()
# neutron_calibrated_energies = (neutron_energy_data[:, 1:] * neutron_calibration_constants).flatten()

# Remove all zero energies
electron_calibrated_energies = electron_calibrated_energies[electron_calibrated_energies!=0.0]
# neutron_calibrated_energies = neutron_calibrated_energies[neutron_calibrated_energies!=0.0]

# Compute the energy difference
electron_energy_difference = (electron_calibrated_energies - TRUE_ENERGY) / TRUE_ENERGY
# neutron_energy_difference = (neutron_calibrated_energies - TRUE_ENERGY) / TRUE_ENERGY

#### COMPUTE DETECTOR RESOLUTION #### 

electron_resolution = electron_energy_difference.std()
print(f"Detector resolution for electrons is {electron_resolution:.2f} MeV")

#### PLOT THE CALIBRATION ####
plt.hist(electron_energy_difference, bins=50, histtype="step", color="maroon", label=r"e$^-$")
# plt.hist(neutron_energy_difference, bins=50, histtype="step", color="darkblue", label="neutron")
plt.xlabel(r"$\frac{E_{cal} - E_{MC}}{E_{MC}}$")
plt.ylabel("Event counts")
plt.legend()
plt.title("Energy percentage difference for Geant4 simulation")
plt.show()