import numpy as np
from utils import *
from scipy.stats import binom
import matplotlib.pyplot as plt

########## Checkpoints ##########

def excersice_1():

    print_question_header(question=1, mode="start",)

    CLAIMED_FAULTY_RATE = 0.05
    ACTUAL_FAULTY_RATE = 0.15
    SAMPLE_SIZE = 100
    X_CUT = 9
    
    fig, ax = plt.subplots(1, 1)

    x = np.arange(0, 100, 1,)
    claimed_faulty_distribution = binom(SAMPLE_SIZE, CLAIMED_FAULTY_RATE)
    actual_faulty_distribution = binom(SAMPLE_SIZE, ACTUAL_FAULTY_RATE)

    # Compute Type-I error 
    x_mask_type_1 = x <= X_CUT
    type_1_error = claimed_faulty_distribution.pmf(x)[x_mask_type_1].sum()

    # Compute Type 2 mask
    # Not that the mask used is the inverse of the mask calculated above
    x_mask_type_2 = x >= X_CUT
    type_2_error = actual_faulty_distribution.pmf(x)[x_mask_type_2].sum()

    print(f"The type I error is {type_1_error:.3f}")
    print(f"The type II error is {type_2_error:.3f}")

    # Plot claimed distribution
    ax.plot(x, claimed_faulty_distribution.pmf(x), 'bo', ms=8, label='Claimed Distribution')
    ax.vlines(x, 0, claimed_faulty_distribution.pmf(x,), colors='b', lw=3, alpha=0.5)

    # Plot actual distribution
    ax.plot(x, actual_faulty_distribution.pmf(x), 'rx', ms=8, label='Actual Distribution')
    ax.vlines(x, 0, actual_faulty_distribution.pmf(x,), colors='r', lw=3, alpha=0.5)
    
    plt.legend()
    plt.xlim(-1, 30)
    plt.show()
    print_question_header(question=1, mode="end",)



if __name__ == "__main__":
    excersice_1()
