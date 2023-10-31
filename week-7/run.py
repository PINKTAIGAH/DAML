from dataIO import DataIO
from maximimLikelyhoodFitter import MaximumLikelyhoodFitter
import matplotlib.pyplot as plt
import numpy

def main():
    io = DataIO()
    dataset = io.readNumpy("/datafile-exp.txt")

    ## Plot the dataset
    plt.hist(dataset, bins=50)
    plt.show()

if __name__ == "__main__":
    main()