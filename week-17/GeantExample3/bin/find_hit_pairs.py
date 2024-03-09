import math
import pandas as pd
import numpy as np

# Define global constants
TRACKER1_FILENAME   = "./data/output_nt_Tracker1.csv"
TRACKER2_FILENAME   = "./data/output_nt_Tracker2.csv"
TRUTH2_FILENAME     = "./data/output_nt_Truth.csv"
NUM_EVENTS          = 1000

def compute_hit_pairs(tracker1_df, tracker2_df, print_pairs=False):
    """
    Compute tracker hit pairs in events
    """

    # Empty list to contain tuples with idx of hit pairs
    hit_pairs = [ [] for _ in range(NUM_EVENTS) ]


    # Loop through each event and find idx of hit pairs in tracker 1 and tracker 2
    for event_idx in range(NUM_EVENTS):
        # Get all hits on trackers for event
        tracker1_event = tracker1_df.loc[tracker1_df["EventID"] == event_idx]
        tracker2_event = tracker2_df.loc[tracker2_df["EventID"] == event_idx]

        # Loop over all hits in event to find pairs
        for tracker1_hit_idx, tracker1_hit in tracker1_event.iterrows():
            for tracker2_hit_idx, tracker2_hit in tracker2_event.iterrows():
                # Check if hits constitute a pair
                if math.fabs(tracker1_hit["Theta"] - tracker2_hit["Theta"]) < 1e-4:
                    # Append tuple of hit idx to hit pair list
                    # Structure of tuple is: (tracker 1 df idx, tracker 2 df idx)
                    hit_pairs[event_idx].append( (tracker1_hit_idx, tracker2_hit_idx) ) 
                    
        # Print that a pair was found 
        if print_pairs: print(f"{len(hit_pairs[event_idx])} pairs found in event {event_idx+1}")
    
    return hit_pairs 

# Read csv files
truth_df = pd.read_csv(TRUTH2_FILENAME, comment="#", names=["EventID", "Phi", "Theta", "Momentum"])
tracker1_df = pd.read_csv(TRACKER1_FILENAME, comment="#", names=["EventID", "Phi", "Theta"])
tracker2_df = pd.read_csv(TRACKER2_FILENAME, comment="#", names=["EventID", "Phi", "Theta"])

# Compute hits
hit_pairs = compute_hit_pairs(tracker1_df, tracker2_df, True)

