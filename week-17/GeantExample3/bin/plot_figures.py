import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define global constants
TRACKER1_FILENAME       = "./data/output_nt_Tracker1.csv"
TRACKER2_FILENAME       = "./data/output_nt_Tracker2.csv"
TRUTH2_FILENAME         = "./data/output_nt_Truth.csv"
IMAGE_OUTDIR            = "./images/"
NUM_EVENTS              = 1000
MAGNETIC_FIELD_STRENGH  = 0.5       # Tesla
TOTAL_PATH_LENGTH       = 8.0       # Meters
INNER_RADIUS            = 4.0       # Meters

def flatten_list(list):
    return [x for xs in list for x in xs]


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


def find_mc_truth(truth_df, tracker_df, tracker_hit_idx, skiphits=[]):
    """
    Find the corresponding mc truth data for a hit from a tracker
    """

    # Find the data of the tracker hit
    tracker_hit = tracker_df.iloc[tracker_hit_idx]

    # Find all mc truth hits from event of tracker hit
    mc_truth_hits = truth_df.loc[truth_df["EventID"] == tracker_hit["EventID"]] 
    
    # Iterate over mc truth hits in event to match tracker hit with mc truth hits
    for mc_hit_idx, mc_truth_hit in mc_truth_hits.iterrows():
        # Skip if hit has already been matched
        if mc_hit_idx in skiphits:
            continue
        # Check if theta hits are from same particle
        if math.fabs(mc_truth_hit["Theta"] - tracker_hit["Theta"]) < 1e-4:
            # Return idx of truth data
            return mc_hit_idx

    # Return None if no matching truth particle was found
    return None


def find_truth_momentum(truth_df, track_df, hit_pair_idx, track_momenta,):
    """
    Create a list containing the truth and detected momenta from the Geant4 simulation
    """

    # Create empty lists for hit momenta and mc truth momenta
    hit_momenta = [ [] for _ in range(NUM_EVENTS) ]
    truth_momenta = [ [] for _ in range(NUM_EVENTS) ]

    for event_idx in range(NUM_EVENTS):
        # List to contain the truth hit idx which have been matched
        matched_truth_hit_idx = []
        
        # Loop over all  
        for hit_idx, track_momentum in zip(hit_pair_idx[event_idx], track_momenta[event_idx]):
            # Find the idx for the matching truth idx
            truth_idx = find_mc_truth(truth_df, track_df, hit_idx[0], matched_truth_hit_idx)
            # Append momenta to lists only if matching truth hit was found
            if truth_idx is not None:
                # Append truth idx to matched hit idx
                matched_truth_hit_idx.append(truth_idx)
                # Append tracker hit momentum
                hit_momenta[event_idx].append(track_momentum[-1])
                # Append truth momentum
                truth_momenta[event_idx].append(truth_df.iloc[truth_idx]["Momentum"])
    
    # Remove empty tuples in list
    hit_momenta = [momenta_list for momenta_list in hit_momenta if len(momenta_list)>0]
    truth_momenta = [momenta_list for momenta_list in truth_momenta if len(momenta_list)>0]

    # # Flatten truth and hit momenta
    # hit_momenta, truth_momenta = flatten_list(hit_momenta), flatten_list(truth_momenta)
    
    return hit_momenta, truth_momenta


def plot_momentum_resolution(hit_momenta, truth_momenta):
    """
    Create a plot containing the momentum resolution distribution of the tracker in the simulation
    """
    
    # Flatten momenta lists
    hit_momenta = np.array(flatten_list(hit_momenta))
    truth_momenta = np.array(flatten_list(truth_momenta))

    # Compute the momenta resolution
    momenta_resolution = (hit_momenta-truth_momenta) / truth_momenta

    # Mask any momenta resolution anomalies due to are occasions when a track curves across the boundary between
    # -pi and pi
    momenta_resolution = momenta_resolution[momenta_resolution>-1.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4)) 

    # Plot histogram w/o log scale
    ax1.hist(momenta_resolution, 30, color="maroon", log=False)
    ax1.set(
        xlabel="Momentum Resolution",
        ylabel="Num of reconstructed particles",
    )
    # Plot histogram w/ log scale
    ax2.hist(momenta_resolution, 30, color="maroon", log=True)
    ax2.set(
        xlabel="Momentum Resolution",
        ylabel="Num of reconstructed particles (log)",
    )

    plt.tight_layout()
    
    # Save the plot
    fig.savefig(IMAGE_OUTDIR + "momentum_resolution_hist.png")


def plot_mass_distribution(hit_momenta):
    """
    Create a plot containing the mass distribution of the tracker in the simulation
    """

    # Compute initial energy of event
    event_init_energy = [sum(np.abs(np.array(momenta_list))) for momenta_list in hit_momenta]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4)) 

    # Plot histogram w/o log scale
    ax1.hist(event_init_energy, 30, color="maroon", log=False)
    ax1.set(
        xlabel = "Total reconstructed energy (GeV)",
        ylabel = "Num of reconstructed particles",
        xticks = np.arange(0, 100, 10)
    )
    # Plot histogram w/ log scale
    ax2.hist(event_init_energy, 30, color="maroon", log=True)
    ax2.set(
        xlabel = "Total reconstructed energy (GeV)",
        ylabel = "Num of reconstructed particles (log)",
        xticks = np.arange(0, 100, 10)
    )

    plt.tight_layout()
    
    # Save the plot
    fig.savefig(IMAGE_OUTDIR + "mass_distribution.png")


# Read csv files
truth_df = pd.read_csv(TRUTH2_FILENAME, comment="#", names=["EventID", "Phi", "Theta", "Momentum"])
tracker1_df = pd.read_csv(TRACKER1_FILENAME, comment="#", names=["EventID", "Phi", "Theta"])
tracker2_df = pd.read_csv(TRACKER2_FILENAME, comment="#", names=["EventID", "Phi", "Theta"])

# Find tracks
hit_pair_idx = compute_hit_pairs(tracker1_df, tracker2_df)

# Find track momenta 
track_momenta = find_track_momentum(tracker1_df, tracker2_df, hit_pair_idx,)

# Find truth momenta
hit_momenta, truth_momenta = find_truth_momentum(truth_df, tracker1_df, hit_pair_idx, track_momenta)

# Plot the momentum resolution
plot_momentum_resolution(hit_momenta, truth_momenta)

# Plot the mass distributiuon
plot_mass_distribution(hit_momenta)