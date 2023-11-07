import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcinv

def print_question_header(question, mode=None):
    """
    Print header of question to command line
    """
    match mode:
        case "start":
            print("\n" + "#"*10 + f" Start of question {question} " + "#"*10 + "\n")
        case "end":
            print("\n" + "#"*10 + f" End of question {question} " + "#"*10 + "\n")

def output_minuit_results(minimised_name, results):
    """
    Print the minimised function and number of dof for a minimised minuit object
    Also returns the minimised function value
    """

    print(f"{minimised_name}: \t chi_squared = {results.fval:.2f} ##### n_dof = {results.ndof:.0f}")
    return results.fval

def find_max(function, bound_low, bound_high, grid=100000,):
    """
    Return the maximum value of a function
    """
    if (not grid >= 0):
        raise ValueError("Grid must be a a positive intiger")
    # Generate grid of x values
    x = np.linspace(bound_low, bound_high, num=grid, endpoint=True,)
    y = function(x)
    return y.max()

def find_significance(pdf, interval_limits):
    """
    Compute the p_value and significance value of a interval
    """

    # Compute integrals
    full_integral = pdf.integrate( (pdf.boundMin, pdf.boundMax) )
    interval_integral = pdf.integrate(interval_limits)

    # Compute p value of probability
    p_value = interval_integral/full_integral

    # Compute Z score
    n_sigma = np.sqrt(2) * erfcinv(p_value)

    return (p_value, n_sigma) 

def plot_signal_with_background(data, signal_data, background_data, bounds, n_bins=100, save_plot=False):
    """
    Plot a figure of the distribution of the signal with background along with 
    individual components of the background and signal distribution
    """

    axes_data = [signal_data, background_data, data]
    axes_titles = [
        f"Signal distribution ({signal_data.size}  events)",
        f"Background distribution ({background_data.size}  events)",
        f"Overall distribution ({data.size}  events)",
    ]

    # Plot data
    fig, axes = plt.subplots(3, 1, sharex="col")

    for idx in range(len(axes)):
        axes[idx].hist(axes_data[idx], bins = n_bins, range=bounds)
        axes[idx].set_title(axes_titles[idx])
    
    axes[-1].set_xlabel("x")
    fig.supylabel("Frequency of Events")

    fig.tight_layout()

    # Save plot to designated folder
    fig.savefig("plots/linear_with_background.png", dpi=600)

    plt.show()