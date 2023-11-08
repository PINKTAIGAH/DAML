from utils import *
from eventGenerator import *
from minimisationStatistics import *
import numpy as np
from iminuit import Minuit
from scipy.stats import chi2
from time import time


"""
#################### FUNCTIONS ######################
"""

def single_toy(
    mean, sigma, slope, intercept, bounds, n_events_signal=300, 
    n_events_background=10000, n_bins=100, plot_distribution=False, skip_plot=False,
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
    if not skip_plot:
        plot_signal_with_background(
            data, signal_data, background_data, bounds, n_bins=n_bins, save_plot=plot_distribution,
        )
    
    # Return SignalWithBAckground object for further analysis
    return pdf

def compute_delta_chi_squared(
    mean, sigma, slope, intercept, bounds, n_events_signal, n_events_background, n_bins,
):
    """
    Function that will generate a toy Linear function (using SignalWithLinear class), test a null hypothesis (Background)
    and alternative hypothesis (Signal+Background) and compute a delta chi squared using Wilk's theorem 
    """

    # Compute signalFraction of pdf
    signalFraction = n_events_signal / (n_events_signal + n_events_background)

    # Generate Background distribution with SignalWithBackground object
    pdf = single_toy(
        mean, sigma, slope, intercept, bounds, n_events_signal=n_events_signal,
        n_events_background=n_events_background, n_bins=n_bins, skip_plot=True
    )

    observed_data = pdf.getMass()
    
    # Construct hypothesis objects
    null_hypothesis = SignalWithBackground(
        mean=1e-6, sigma=1e-6, signalFraction=0.0, slope=slope, intercept=intercept, bounds=bounds,
    )
    alternative_hypothesis = SignalWithBackground(
        mean=mean, sigma=sigma, signalFraction=signalFraction, slope=slope, intercept=intercept, bounds=bounds, 
    )

    # Initialise minimisation statistic objects
    null_statistic = ChiSquaredModified(null_hypothesis, observed_data)
    alternative_statistic = ChiSquaredModified(alternative_hypothesis, observed_data)
    
    #  Initialise minuit objects
    m_null = Minuit( null_statistic.evaluateNull, slope=-0.3, intercept=18.0,)
    m_alternative = Minuit( alternative_statistic.evaluateAlternative, signalFraction=0.5, slope=-0.3, intercept=18.0)

    # Minimise hypothesis
    result_null = m_null.migrad()
    result_alternative = m_alternative.migrad()

    # Compute difference in chi squared using Wilks theorem
    delta_chi_squared = result_null.fval - result_alternative.fval      # Note this statistic has a 1 dof (3-2)

    return abs(delta_chi_squared)


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

    print(f"The significance that there is a fluctuation of {int(MEASURMENT)} events from a mean of {int(MEAN)} is {n_sigma:.2f} sigma")

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

    print(f"The significance that there is a fluctuation of {int(MEASURMENT)} events from a mean of {int(MEAN)} is {n_sigma:.2f} sigma")

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
    TOTAL_EXPECTED_EVENTS = N_SIGNAL_EVENT + N_BACKGROUND_EVENTS
    SIGNAL_FRACTION = N_SIGNAL_EVENT/TOTAL_EXPECTED_EVENTS

    # Generate SignalWithBackground object
    pdf = single_toy(
        MEAN, SIGMA, SLOPE, INTERCEPT, BOUNDS, n_events_signal=N_SIGNAL_EVENT,
        n_events_background=N_BACKGROUND_EVENTS, n_bins=N_BINS, plot_distribution=PLOT_DISTRIBUTION 
    )

    observed_data = pdf.getMass()
    
    # Construct hypothesis objects
    null_hypothesis = SignalWithBackground(
        mean=1e-6, sigma=1e-6, signalFraction=0.0, slope=SLOPE, intercept=INTERCEPT, bounds=BOUNDS,
    )
    alternative_hypothesis = SignalWithBackground(
        mean=MEAN, sigma=SIGMA, signalFraction=SIGNAL_FRACTION, slope=SLOPE, intercept=INTERCEPT, bounds=BOUNDS, 
    )

    # Initialise minimisation statistic objects
    null_statistic = ChiSquaredModified(null_hypothesis, observed_data)
    alternative_statistic = ChiSquaredModified(alternative_hypothesis, observed_data)

    #  Initialise minuit objects
    m_null = Minuit( null_statistic.evaluateNull, slope=-0.3, intercept=18.0,)
    m_alternative = Minuit( alternative_statistic.evaluateAlternative, signalFraction=0.5, slope=-0.3, intercept=18.0)

    # Minimise hypothesis
    result_null = m_null.migrad()
    result_alternative = m_alternative.migrad()

    # Print out and save minimised chi squared
    chi_squared_null = output_minuit_results("Null hypotheses", result_null)
    chi_squared_alternative = output_minuit_results("Alternative hypotheses", result_alternative)

    # Compute difference in chi squared using Wilks theorem
    delta_chi_squared = chi_squared_null - chi_squared_alternative      # Note this statistic has a 1 dof (3-2)
    print(f"Wilk's Theorem: \t delta chi_squared = {delta_chi_squared:.4f}")

    # Find p-value and significance of deviation between two hypotheses
    p_value = 1 - chi2.cdf(x=delta_chi_squared, df=1)
    n_sigma = compute_z_score(p_value)
    
    print(f"The significance that there is a deviation vetween the two hypothese is {n_sigma:.2f} sigma and a p value of {p_value:.3f}")

    # Plot hypothesis
    mass = np.linspace(*BOUNDS, 1000,)

    plt.hist(observed_data, bins=100, density=True, color="hotpink", label="MC data")
    plt.plot(mass, null_hypothesis._evaluate(mass)/null_hypothesis.integrate(BOUNDS), c="mediumblue", ls="--", label="Null hypothesis")
    plt.plot(mass, alternative_hypothesis._evaluate(mass)/alternative_hypothesis.integrate(BOUNDS), c="black", ls="-", label="Alternate hypothesis")
    plt.legend()
    plt.title("Observed data with fitted hypothese")
    plt.xlabel("mass, m (GeV)")
    plt.ylabel("Probability")
    plt.savefig("plots/hypothese_plots.png", dpi=600)
    plt.show()

    print_question_header(question=4, mode="end")

def excersice_5():

    """
    Run question 4 of checkpoint 4
    """

    print_question_header(question=5, mode="start")

    # Define toy distribution parameters
    BOUNDS = (0.0, 20.0)
    MEAN = 10.0
    SIGMA = 0.5
    SLOPE = -1.0
    INTERCEPT = 20.0
    N_SIGNAL_EVENT = 0
    N_BACKGROUND_EVENTS = 10000
    N_BINS = 100
    N_DELTA_CHI_SQUARED_EVENTS = 1000

    # Array to contain delta chi squared
    delta_chi_squared_array = []
    # Iterate to compute multiple delta chi squared for linear distribution
    for epoch in range(N_DELTA_CHI_SQUARED_EVENTS):
        delta_chi_squared = compute_delta_chi_squared(
            MEAN, SIGMA, SLOPE, INTERCEPT, BOUNDS, N_SIGNAL_EVENT, N_BACKGROUND_EVENTS, N_BINS,
        )
        delta_chi_squared_array.append(delta_chi_squared) 
    # Convert list to numpy array
    delta_chi_squared_array = np.array(delta_chi_squared_array)

    plt.close()
    # Plot delta chi squared distribution
    fig, axes = plt.subplots(2, 1,)
    axes[0].hist(delta_chi_squared_array, bins=N_BINS, density=True, color="hotpink", range=(0.0, 0.006))
    axes[0].set_ylabel("Probability")
    axes[0].set_xlabel(r"$\Delta\chi^2$ statistic (Wilk's Theorem)")
    axes[0].set_title(r"$\Delta\chi^2$ Probability density function")
    axes[1].hist(delta_chi_squared_array, bins=N_BINS, density=True, color="hotpink", cumulative=True, range=(0.0, 0.006))
    axes[1].set_ylabel("Probability")
    axes[1].set_title(r'$\Delta\chi^2$ Cumulative density function')
    axes[1].set_xlabel(r"Cumulative $\Delta\chi^2$ statistic (Wilk's Theorem)")
    fig.tight_layout()
    plt.show()


    print_question_header(question=5, mode="end")

if __name__ == "__main__":
    excersice_1()
    excersice_2()
    excersice_3()
    excersice_4()
    excersice_5()