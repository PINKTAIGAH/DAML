import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define dirtectory for data files
ELECTRON_DATA = {
    "histogram"     : "./data/electron_300mev_20/output_h1_LayerTotal.csv",
    "nTuple"        : "./data/electron_300mev_20/output_nt_Energy.csv",
}

NEUTRON_DATA = {
    "histogram"     : "./data/neutron_300mev_20/output_h1_LayerTotal.csv",
    "nTuple"        : "./data/neutron_300mev_20/output_nt_Energy.csv",
}

# Define simulation parameters
INIT_ENERGY = 300.0   # MeV
DETECTOR_LAYERS = 6

# Build a list containing the detector's components
detector_components = []
for idx in range(DETECTOR_LAYERS):
    if idx == 0:
        detector_components.append("true")
    else:
        detector_components.append(f"layer {idx}")

# Import ntuple files and take mean energy of all particles
energy_data = {
    "electron"  : pd.read_csv(ELECTRON_DATA["nTuple"],  comment="#", names=detector_components),
    "neutron"   : pd.read_csv(NEUTRON_DATA["nTuple"], comment="#", names=detector_components),
}

#### PLOT THE ENERGY DISTRIBUTIONS #### 
fig, ax = plt.subplots(1, 2, squeeze=True, sharey=True)

for idx, key in enumerate(energy_data.keys()):
    ax[idx].hist(energy_data[key].T, 10, label=detector_components, stacked=True)
    ax[idx].set(
        xlabel  = "Energy (MeV)",
        ylabel  = "Entries",
        title   = key,
    )
    ax[idx].legend()
fig.suptitle("MC Energy in each detector layer")
fig.savefig("../images/detector_energy_hist.png")
plt.show()