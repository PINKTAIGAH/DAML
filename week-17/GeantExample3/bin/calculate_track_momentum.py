import math
import pandas as pd
import numpy as np

# Define global constants
TRACKER1_FILENAME       = "./data/output_nt_Tracker1.csv"
TRACKER2_FILENAME       = "./data/output_nt_Tracker2.csv"
TRUTH2_FILENAME         = "./data/output_nt_Truth.csv"
NUM_EVENTS              = 1000
MAGNETIC_FIELD_STRENGH  = 0.5       # Tesla
TOTAL_PATH_LENGTH       = 8.0       # Meters
INNER_RADIUS            = 4.0       # Meters

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

def find_track_momentum(tracker1_df, tracker2_df, hit_pair_idx, print_momentum=False):
    """
    Function will compute the track momentum of any hit pairs found.
    """

    # Empty list to contain the track momenta
    track_momenta = [ [] for _ in hit_pair_idx ]

    # loop over events in simulation
    for event_idx in range(NUM_EVENTS):
        # Print Event number
        if print_momentum: print(f"EVENT {event_idx+1}")
        # Loop over found hit pairs (tracks) in each event
        for tracker1_idx, tracker2_idx in hit_pair_idx[event_idx]:
            # Retrive the data from the hits from each tracker
            tracker1_hit = tracker1_df.iloc[tracker1_idx]
            tracker2_hit = tracker2_df.iloc[tracker2_idx]

            # Compute the momentum of the track
            track_momentum = calculate_momentum(
                tracker1_hit["Theta"], 
                tracker1_hit["Phi"],
                tracker2_hit["Phi"]
            )

            # Append momentum of track to the list
            track_momenta[event_idx].append(track_momentum)
            # Print total momentum
            if print_momentum: print(f"p = {track_momentum[-1]:.4f} GeV")

    return track_momenta
    

def calculate_momentum(theta, phi1, phi2,):
    """
    Calculate the momentum of a single track using change in azimuthal angle of track under a magnetic field
    """

    # Find the sagitta
    delta_phi   = math.fabs(phi1 - phi2)
    sagitta     = math.sin(delta_phi) * INNER_RADIUS

    # Find the track's bending radius
    bending_radius = TOTAL_PATH_LENGTH**2 / (8*sagitta)

    # Compute the transverse momentum and total moementum of the track
    transverse_momentum = 0.3 * MAGNETIC_FIELD_STRENGH * bending_radius     # GeV
    total_momentum      = transverse_momentum / math.sin(theta)             # GeV

    return transverse_momentum, total_momentum

# Read csv files
truth_df = pd.read_csv(TRUTH2_FILENAME, comment="#", names=["EventID", "Phi", "Theta", "Momentum"])
tracker1_df = pd.read_csv(TRACKER1_FILENAME, comment="#", names=["EventID", "Phi", "Theta"])
tracker2_df = pd.read_csv(TRACKER2_FILENAME, comment="#", names=["EventID", "Phi", "Theta"])

# Find tracks
hit_pair_idx = compute_hit_pairs(tracker1_df, tracker2_df)

# Find track momenta 
find_track_momentum(tracker1_df, tracker2_df, hit_pair_idx, True)