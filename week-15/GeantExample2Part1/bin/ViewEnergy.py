import numpy as np
import matplotlib.pyplot as plt

# Define dirtectory for data files
ELECTRON_DATA = {
    "histogram"     : "./data/electron_20/output_h1_LayerTotal.csv",
    "nTuple"        : "./data/electron_20/output_nt_Energy.csv",
}

NEUTRON_DATA = {
    "histogram"     : "./data/neutron_20/output_h1_LayerTotal.csv",
    "nTuple"        : "./data/neutron_20/output_nt_Energy.csv",
}

# Define simulation parameters
INIT_ENERGY = 300.0   # MeV
DETECTOR_LAYERS = 6

# Import ntuple files and take mean energy of all particles
energy_data = {
    "electron"  : np.loadtxt(ELECTRON_DATA["nTuple"], delimiter=",", comments="#").T.mean(axis=1),
    "neutron"   : np.loadtxt(NEUTRON_DATA["nTuple"], delimiter=",", comments="#").T.mean(axis=1),
}

# Build a list containing the detector's components
detector_components = []
for idx in range(DETECTOR_LAYERS):
    if idx == 0:
        detector_components.append("init")
    else:
        detector_components.append(f"layer {idx}")

#### PLOT THE ENERGY DISTRIBUTIONS #### 
colors = ("maroon", "darkblue")

x = np.arange(len(detector_components))  # the label locations
width = 0.35  # the width of the bars
multiplier = 0

fig, ax = plt.subplots()

# Plot each dataset on the axes
for idx, (key, energy) in enumerate(energy_data.items()):
    offset = width * multiplier
    rects = ax.bar(x + offset, energy, width, label=key, alpha=0.5, color=colors[idx])
    # ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Energy (MeV)')
ax.set_title('Average energy distribution in detector layers', y=1.03, fontsize="14")
ax.set_xticks(x)
ax.set_xticklabels(detector_components)
ax.legend(loc='upper right')
plt.show()
