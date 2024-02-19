import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define dirtectory for data files
ELECTRON_DATA = {
    "histogram"     : "./data/electron_20gev_200/output_h1_LayerTotal.csv",
    "nTuple"        : "./data/electron_20gev_200/output_nt_Energy.csv",
}

# Define simulation parameters
DETECTOR_LAYERS = 6

# Build a list containing the detector's components
detector_components = []
for idx in range(DETECTOR_LAYERS):
    if idx == 0:
        detector_components.append("truth")
    else:
        detector_components.append(f"layer {idx}")

# Load in energy data
electron_energy = pd.read_csv(ELECTRON_DATA["nTuple"],  comment="#", names=detector_components)

# Group together important keys
mc_truth_keys = [key for key in electron_energy.columns if "truth" in key]
detector_layer_keys = [key for key in electron_energy.columns if "layer" in key]

# Compute energy detected
electron_energy["detected"] = electron_energy[detector_layer_keys].sum(axis=1)

# Compute calibration factor
electron_calibration_factor = (electron_energy["truth"]/electron_energy["detected"]).mean()

# Obtain calibrated energies
electron_calibrated_energies = electron_energy["detected"] * electron_calibration_factor

# obtain the energy resolution
electron_energy["resolution"] = (electron_calibrated_energies - electron_energy["truth"]) / electron_energy["truth"]

#### PLOT THE CALIBRATION ####
plt.hist(electron_energy["resolution"], bins=50, histtype="step", color="maroon", label=r"e$^-$")
plt.xlabel(r"$\frac{E_{cal} - E_{MC}}{E_{MC}}$")
plt.ylabel("Event counts")
plt.legend()
plt.title("Energy resolution for Geant4 simulation")
plt.show()