import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_energy_difference(filename, detector_components):
    electron_energy = pd.read_csv(filename,  comment="#", names=detector_components)

    # Group together important keys
    detector_layer_keys = [key for key in electron_energy.columns if "layer" in key]

    # Compute energy detected
    electron_energy["detected"] = electron_energy[detector_layer_keys].sum(axis=1)

    # Compute calibration factor
    electron_calibration_factor = (electron_energy["truth"]/electron_energy["detected"]).mean()

    # Obtain calibrated energies
    electron_calibrated_energies = electron_energy["detected"] * electron_calibration_factor

    # obtain the energy resolution
    energy_difference = (electron_calibrated_energies - electron_energy["truth"]) / electron_energy["truth"]

    return energy_difference

# Define global parameters
DATA_DIRECTORY  = "./data/2d_hist_data/"
ENERGY_RANGE    = [10000 + increment for increment in range(0, 95000, 5000)]
DETECTOR_LAYERS = 6

# Build a list containing the detector's components
detector_components = []
for idx in range(DETECTOR_LAYERS):
    if idx == 0:
        detector_components.append("truth")
    else:
        detector_components.append(f"layer {idx}")

# Make list containing the filename of each data file
filenames = [DATA_DIRECTORY + "electron_" + str(energy) + "mev_1000.csv" for energy in ENERGY_RANGE]


energy_differences = []
true_energies = []
for idx, filename in enumerate(filenames):
    energy_difference = compute_energy_difference(filename, detector_components)
    true_energy = np.full_like(energy_difference, ENERGY_RANGE[idx])
    
    energy_differences.extend(list(energy_difference))
    true_energies.extend(list(true_energy))

plt.hist2d(true_energies, energy_differences, bins=(20, 80))
plt.ylabel(r"$\frac{E_{cal} - E_{MC}}{E_{MC}}$")
plt.xlabel("Energy (MeV)")
plt.title("Detector resolution distribution at different particle energies")
plt.savefig("../images/2D_resolution_hist.png")
plt.show()