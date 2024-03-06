
import pandas as pd
import numpy as np

# Define global constants
TRACKER1_FILENAME   = "./data/output_nt_Tracker1.csv"
TRACKER2_FILENAME   = "./data/output_nt_Tracker1.csv"
TRUTH2_FILENAME     = "./data/output_nt_Truth.csv"
NUM_EVENTS          = 1000

def compute_hits(truth_df, tracker1_df, tracker2_df):
    """
    Compute the number of detector hits for all events simulated.
    Ironically enough, we can use this same function to compute the number of MC particles in each event
    """

    # Compute the multiplicity of event number for each detector part
    num_hits_truth = truth_df["EventID"].value_counts()
    num_hits_tracker1 = tracker1_df["EventID"].value_counts()
    num_hits_tracker2 = tracker2_df["EventID"].value_counts()

    # Create dataframe containing all hit information & replace nan with 0
    hit_df = pd.DataFrame(
        {"Truth":num_hits_truth, "Tracker1":num_hits_tracker1, "Tracker2":num_hits_tracker2}
    ).fillna(0.0)

    # Replace float to int and return dataframe

    return hit_df.astype(int)


# Read csv files
truth_df = pd.read_csv(TRUTH2_FILENAME, comment="#", names=["EventID", "Phi", "Theta", "Momentum"])
tracker1_df = pd.read_csv(TRACKER1_FILENAME, comment="#", names=["EventID", "Phi", "Theta"])
tracker2_df = pd.read_csv(TRACKER2_FILENAME, comment="#", names=["EventID", "Phi", "Theta"])

# Compute hits
hit_df = compute_hits(truth_df, tracker1_df, tracker2_df)

# Print out hit info
for idx in range(NUM_EVENTS):
    # Exit printing loop if NUM_EVENTS was larger than dataframe size
    if idx+1 > hit_df.shape[0]:
        break
    print(hit_df.iloc[idx], "\n")