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

    return energy_difference, electron_energy["truth"]

# Define global parameters
DATA_DIRECTORY  = "./data/2d_hist_data/"
DETECTOR_LAYERS = 6

# Build a list containing the detector's components
detector_components = []
for idx in range(DETECTOR_LAYERS):
    if idx == 0:
        detector_components.append("truth")
    else:
        detector_components.append(f"layer {idx}")


energy_differences, true_energies = compute_energy_difference(DATA_DIRECTORY+"output_nt_Energy.csv", detector_components)

plt.hist2d(true_energies, energy_differences, bins=(80, 100))
plt.ylabel(r"$\frac{E_{cal} - E_{MC}}{E_{MC}}$")
plt.xlabel("Energy (MeV)")
plt.title("Detector resolution distribution at different particle energies")
plt.savefig("../images/2D_resolution_hist.png")
plt.show()