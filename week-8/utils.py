import numpy as np
import matplotlib.pyplot as plt

def find_max(function, bound_low, bound_high, grid=100000,):
    """
    Return the maximum value of a function
    """
    if not grid >= 0:
        raise ValueError("Grid must be a a positive intiger")
    # Generate grid of x values
    x = np.linspace(bound_low, bound_high, num=grid, endpoint=True,)
    y = function(x)
    return y.max()

def plot_signal_with_linear(data, signal_data, background_data, n_bins=100, save_plot=False):
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
        axes[idx].hist(axes_data[idx], bins = n_bins)
        axes[idx].set_title(axes_titles[idx])
    
    axes[-1].set_xlabel("x (Events)")
    fig.tight_layout()

    if save_plot:
        # Save the figure to plot directory
        fig.savefig("plots/linear_with_background.png", dpi=600)

    plt.show()