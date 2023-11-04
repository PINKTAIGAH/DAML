from dataIO import DataIO
from maximimLikelyhoodFitter import MaximumLikelyhoodFitter
import matplotlib.pyplot as plt
import numpy as np

def main():
    io = DataIO()
    fitter = MaximumLikelyhoodFitter(minimiser="scipy")
    dataset = io.readNumpy("/datafile-exp.txt")

    p0 = [2.0]
    bounds = (0, 10)

    ## Plot the dataset
    
    tau, = fitter.run(dataset, p0, bounds)
    plt.hist(dataset, bins=50)
    plt.show()

    timeArray = np.linspace(1, 10, 100)
    like = []
    for time in timeArray:
        like.append(fitter._negLogLikelyhood([time], dataset))
    plt.plot(timeArray, like)
    plt.show()


if __name__ == "__main__":
    main()