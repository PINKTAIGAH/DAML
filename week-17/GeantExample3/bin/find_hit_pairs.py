??? from here until ???END lines may have been inserted/deleted
import pandas as pd
import numpy as np

# Define global constants
TRACKER1_FILENAME   = "./data/output_nt_Tracker1.csv"
TRACKER2_FILENAME   = "./data/output_nt_Tracker1.csv"
TRUTH2_FILENAME     = "./data/output_nt_Truth.csv"
NUM_EVENTS          = 1000

def compute_hit_pairs(tracker1_df, tracker2_df):
    """
    Compute tracker hit pairs in events
    """

    # Empty list to contain tuples with idx of hit pairs
    hit_pairs = [ [] for _ in range(tracker1_df.shape[0] ]


    # Loop through each event and find idx of hit pairs in tracker 1 and tracker 2
    for event_idx in range(tracker1_df.shape[0]):
        
        print(tracker1_df.loc[['EventID' == event_idx]])




# Read csv files
truth_df = pd.read_csv(TRUTH2_FILENAME, comment="#", names=["EventID", "Phi", "Theta", "Momentum"])
tracker1_df = pd.read_csv(TRACKER1_FILENAME, comment="#", names=["EventID", "Phi", "Theta"])
tracker2_df = pd.read_csv(TRACKER2_FILENAME, comment="#", names=["EventID", "Phi", "Theta"])

# Compute hits
hit_df = compute_hit_pairs(import pandas as pd
import numpy as np

# Define global constants
TRACKER1_FILENAME   = "./data/output_nt_Tracker1.csv"
TRACKER2_FILENAME   = "./data/output_nt_Tracker1.csv"
TRUTH2_FILENAME     = "./data/output_nt_Truth.csv"
NUM_EVENTS          = 1000

def compute_hit_pairs(tracker1_df, tracker2_df):
    """ 
    Compute tracker hit pairs in events
    """

    # Empty list to contain tuples with idx of hit pairs
    hit_pairs = [ [] for _ in range(tracker1_df.shape[0] ]
    

    # Loop through each event and find idx of hit pairs in tracker 1 and tracker 2
    for event_idx in range(tracker1_df.shape[0]):                          

        print(tracker1_df.loc[['EventID' == event_idx]])




# Read csv files
truth_df = pd.read_csv(TRUTH2_FILENAME, comment="#", names=["EventID", "Phi", "Theta", "Momentum"])
tracker1_df = pd.read_csv(TRACKER1_FILENAME, comment="#", names=["EventID", "Phi", "Theta"])
tracker2_df = pd.read_csv(TRACKER2_FILENAME, comment="#", names=["EventID", "Phi",tracker1_df, tracker2_df)

???END
