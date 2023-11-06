from utils import plot_signal_with_background, find_significance, print_question_header
from eventGenerator import *
import numpy as np

"""
#################### FUNCTIONS ######################
"""

def single_toy(
    mean, sigma, slope, intercept, bounds, n_events_signal=300, 
    n_events_background=10000, n_bins=100, plot_distribution=False,
):
    """
    Generate a toy dataset of a gaussian signal with linear background
    """
        
    if (not isinstance(bounds, tuple)):
        raise TypeError("Variable bound must be a tuple with the form (boundMin, boundMax)")
    if (not len(bounds)) == 2:
        raise ValueError("Variable bound must have form (boundMin, boundMax)")
    if (not bounds[0] < bounds[1]):
        raise ValueError("First element in tuple must be smaller than second")

    n_events_total = n_events_signal + n_events_background
    signal_fraction = n_events_signal/n_events_total
    # Generate pdf object
    pdf = SignalWithBackground(mean, sigma, slope, intercept, bounds, signalFraction=signal_fraction)

    # Generate random events
    for _ in range(n_events_total): 
        _ = pdf.next()

    # Get events for each distribution
    data = pdf.getMass()
    signal_data = pdf.getMassSignal()
    background_data = pdf.getMassBackground()

    # Plot the toy dataset if requested
    plot_signal_with_background(
        data, signal_data, background_data, bounds, n_bins=n_bins, save_plot=plot_distribution,
    )
    
    # Return SignalWithBAckground object for further analysis
    return pdf


"""
#################### EXCERCISES ######################
"""


def excersice_1():
    """
    Run question 1 of checkpoint 4
    """

    print_question_header(question=1, mode="start")

    # Define distribution parameters
    BOUNDS = (0.0, 20.0)
    MEAN = 10.0
    SIGMA = 0.5
    SLOPE = -1.0
    INTERCEPT = 20.0
    N_SIGNAL_EVENT = 300
    N_BACKGROUND_EVENTS = 10000
    N_BINS = 100
    PLOT_DISTRIBUTION = True
    
    # Generate single toy dataset
    _ = single_toy(
        MEAN, SIGMA, SLOPE, INTERCEPT, BOUNDS, n_events_signal=N_SIGNAL_EVENT,
        n_events_background=N_BACKGROUND_EVENTS, plot_distribution=PLOT_DISTRIBUTION, n_bins=N_BINS
    )

    print_question_header(question=1, mode="end")

def excersice_2():
    """
    Run question 2 of checkpoint 4
    """

    print_question_header(question=2, mode="start")

    # Define parameters for gaussian given by events from background process
    MEAN = 10000.0
    MEASURMENT = 10000.0 + 300.0
    SIGMA = np.sqrt(MEAN)                       # Using Poison -> Gaussian approximation
    BOUNDS = (MEAN - 1000.0, MEAN + 1000.0)
    INTERVAL_LIMITS = (MEASURMENT, BOUNDS[1])      # Integral limits for k >= 10300

    # Define gaussian pdf
    pdf = Gaussian(MEAN, SIGMA, BOUNDS,)

    _, n_sigma = find_significance(pdf, INTERVAL_LIMITS)

    print(f"The significance that there is a fluctuation of {int(MEASURMENT)} events from a mean of {int(MEAN)} is {n_sigma:.3f} sigma")

    print_question_header(question=2, mode="end")

def excersice_3():
    """
    Run question 3 of checkpoint 4
    """

    print_question_header(question=3, mode="start")

    # Define parameters for Background pdf
    SLOPE = -1.0
    INTERCEPT = 20.0
    BOUNDS_LINEAR = (0.0, 20.0)
    BOUNDS_POSSIBLE_SIGNAL = (5.0, 15.0)

    # Define parameters for gaussian given by events from background process
    MEAN = 0.5 * 10000.0
    MEASURMENT = 0.5 * 10000.0 + 300.0
    SIGMA = np.sqrt(MEAN)                       # Using Poison -> Gaussian approximation
    BOUNDS_GAUSSIAN = (MEAN - 500.0, MEAN + 500.0)
    INTERVAL_LIMITS = (MEASURMENT, BOUNDS_GAUSSIAN[1])      # Integral limits for k >= 10300

    # Define a gaussian and linear pdf
    pdf_background = Linear(SLOPE, INTERCEPT, BOUNDS_LINEAR)
    pdf_gaussain = Gaussian(MEAN, SIGMA, BOUNDS_GAUSSIAN)

    # Compute integrals of background 
    total_background_integral = pdf_background.integrate(BOUNDS_LINEAR)
    possible_signal_integral = pdf_background.integrate(BOUNDS_POSSIBLE_SIGNAL)

    background_event_ratio = possible_signal_integral/total_background_integral

    print(f"Ratio between expected background events between ranges {BOUNDS_LINEAR} and {BOUNDS_POSSIBLE_SIGNAL} is {background_event_ratio*100:.2f}%")

    # Compute significance of there being 300 aditional events for a background with 50% expected background events
    _, n_sigma = find_significance(pdf_gaussain, INTERVAL_LIMITS)

    print(f"The significance that there is a fluctuation of {int(MEASURMENT)} events from a mean of {int(MEAN)} is {n_sigma:.3f} sigma")

    print_question_header(question=3, mode="end")

def excersice_4():
    """
    Run question 4 of checkpoint 4
    """

    print_question_header(question=4, mode="start")


    # Define toy distribution parameters
    BOUNDS = (0.0, 20.0)
    MEAN = 10.0
    SIGMA = 0.5
    SLOPE = -1.0
    INTERCEPT = 20.0
    N_SIGNAL_EVENT = 150 
    N_BACKGROUND_EVENTS = 10000
    N_BINS = 100
    PLOT_DISTRIBUTION = False 

    # Generate SignalWithBackground object
    pdf = single_toy(
        MEAN, SIGMA, SLOPE, INTERCEPT, BOUNDS, n_events_signal=N_SIGNAL_EVENT,
        n_events_background=N_BACKGROUND_EVENTS, n_bins=N_BINS, plot_distribution=PLOT_DISTRIBUTION 
    )

    print_question_header(question=4, mode="end")

if __name__ == "__main__":
    excersice_1()
    excersice_2()
    excersice_3()
    excersice_4()