import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Define simulation parameters
DATA_FILE = "./data/output_nt_Energy.csv"
DETECTOR_LAYERS = 6

# Build a list containing the detector's components
detector_components = []
for idx in range(DETECTOR_LAYERS + DETECTOR_LAYERS-1):
    if idx == 0:
        detector_components.append("truth")
    elif idx < 6:
        detector_components.append(f"layer {idx}")
    else:
        detector_components.append(f"electron layer {idx-5}")

# Define important keys
truth_key = "truth"
detected_energy_keys = [key for key in detector_components if not ("electron" in key or "truth" in key)]
electron_energy_keys = [key for key in detector_components if "electron" in key]

# Inport energy data
energy_data = pd.read_csv(DATA_FILE, comment="#", names=detector_components)

# Compute mean deposited energy of electorns
electron_deposited_energy = energy_data[electron_energy_keys].sum(axis=1).to_numpy().mean()

# Compute the mean deposited energy of the event 
event_deposited_energy = energy_data[detected_energy_keys].sum(axis=1).to_numpy().mean()

# Compute energy fraction
energy_fraction = electron_deposited_energy/event_deposited_energy

# Print out result
print(f"The fraction of the total detected energy in an event that was deposited by electrons is {energy_fraction:.3f}")
