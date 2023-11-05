from utils import plot_signal_with_linear
from eventGenerator import SignalWithBackground 

"""
#################### FUNCTIONS ######################
"""

def single_toy(
    mean, sigma, slope, intercept, bounds, n_events_signal=300, 
    n_events_background=10000, n_bins=100, save_plot=False,
):
    """
    Generate a toy dataset of a gaussian signal with linear background
    """
        
    if not isinstance(bounds, tuple):
        raise TypeError("Variable bound must be a tuple with the form (boundMin, boundMax)")
    if not len(bounds) == 2:
        raise ValueError("Variable bound must have form (boundMin, boundMax)")

    n_events_total = n_events_signal + n_events_background
    signal_fraction = n_events_signal/n_events_total
    # Generate pdf object
    pdf = SignalWithBackground(mean, sigma, slope, intercept, bounds, signalFraction=signal_fraction)

    # Generate random events
    for _ in range(n_events_total): pdf.next()

    # Get events for each distribution
    data = pdf.getMass()
    signal_data = pdf.getMassSignal()
    background_data = pdf.getMassBackground()
    
    plot_signal_with_linear(data, signal_data, background_data, n_bins=n_bins, save_plot=save_plot)


"""
#################### EXCERCISES ######################
"""


def excercise_1():
    """
    Run question 1 of checkpoint 4
    """

    # Define distribution parameters
    BOUNDS = (0, 20)
    MEAN = 10.0
    SIGMA = 0.5
    SLOPE = -1.0
    INTERCEPT = 20.0
    N_SIGNAL_EVENT = 500
    N_BACKGROUND_EVENTS = 10000
    N_BINS = 100
    SAVE_PLOT = True
    
    # Generate single toy dataset
    single_toy(
        MEAN, SIGMA, SLOPE, INTERCEPT, BOUNDS, n_events_signal=N_SIGNAL_EVENT,
        n_events_background=N_BACKGROUND_EVENTS, save_plot=SAVE_PLOT
    )

if __name__ == "__main__":
    excercise_1()