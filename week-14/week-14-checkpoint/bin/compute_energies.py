import numpy as np

##### COMPUTE RATIO FOR PLUTONIUM DIOXIDE TARGET #####

# Define file names
shield_nt_dir = "datasets/output_nt_Shield_plutonium.csv"
target_nt_dir = "datasets/output_nt_Target_plutonium.csv"

# load in ntuple files
target_energies = np.loadtxt(target_nt_dir, dtype=np.float32, comments="#", delimiter=",")
shield_energies = np.loadtxt(shield_nt_dir, dtype=np.float32, comments="#", delimiter=",")

# Compute average ration of energies and set any nans in numpy array to 0
average_energy_ratio = np.nan_to_num( (shield_energies/target_energies).mean(), nan=0.0)

print(f"The average energy deposit ratio for a plutonium oxide target is {average_energy_ratio:.2f} MeV")


##### COMPUTE RATIO FOR LAr TARGET #####

# Define file names
shield_nt_dir = "datasets/output_nt_Shield_LAr.csv"
target_nt_dir = "datasets/output_nt_Target_LAr.csv"

# load in ntuple files
target_energies = np.loadtxt(target_nt_dir, dtype=np.float32, comments="#", delimiter=",")
shield_energies = np.loadtxt(shield_nt_dir, dtype=np.float32, comments="#", delimiter=",")

# Compute average ration of energies and set any nans in numpy array to 0
average_energy_ratio = np.nan_to_num( (shield_energies/target_energies).mean(), nan=0.0)

print(f"The average energy deposit ratio for a LAr target is {average_energy_ratio:.2f} MeV\n")

print(f"From these results, we see that the plutonium absorbed a greater amount of energy relative to the shield compated to the liquid argon target.")
print(f"This is likely as a result of the plutonium oxide having heavier nucli which results in more interaction for the simulated phontons, and hence a greater energy deposition.")