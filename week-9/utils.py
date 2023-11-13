import numpy as np
from scipy.special import erfinv

def print_question_header(question, mode=None):
    """
    Print header of question to command line
    """
    match mode:
        case "start":
            print("\n" + "#"*10 + f" Start of question {question} " + "#"*10 + "\n")
        case "end":
            print("\n" + "#"*10 + f" End of question {question} " + "#"*10 + "\n")

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

    n_sigma = compute_z_score(p_value)

    return (p_value, n_sigma) 

def compute_z_score(p_value):
    """
    Compute significance value from p_value
    """

    # Compute Z score
    n_sigma = np.sqrt(2) * erfinv(1 - p_value)

    return n_sigma

