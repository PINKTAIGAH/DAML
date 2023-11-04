from dataIO import DataIO
from minimiser import *
import matplotlib.pyplot as plt
import numpy as np
from probabilityDesityFunctions import *

def excercise1Scipy():
    io = DataIO()
    dataset = io.readNumpy("/datafile-exp.txt")

    pTruth = [2.2]
    p0 = [1.0]
    bounds = (0, 10)

    pdf = Exponential(bounds)
    fitter = ScipyFitter(dataset, pdf)
    
    _ = fitter.fit(p0,)

    plt.hist(dataset, bins=50)
    plt.show()


def excercise1Minuit():
    io = DataIO()
    dataset = io.readNumpy("/datafile-exp.txt")

    pTruth = {"lifetime":2.2,}
    p0 = {"lifetime":1.0}
    bounds = (0, 10)
    errorDef = 0.5


    pdf = Exponential(bounds)
    fitter = MinuitFitter(dataset, pdf)
    
    _ = fitter.fit(p0, errorDef, None)

    plt.hist(dataset, bins=50)
    plt.show()


def excercise2Scipy():
    io = DataIO()
    dataset = io.readNumpy("/datafile-expresonance.txt")

    p0 = [0.5, 3.0, 2.0]
    bounds = (0, 10)

    pdf = ExponentialResonance(bounds)
    fitter = ScipyFitter(dataset, pdf)
    
    _ = fitter.fit(p0, method="l-bfgs-b")

    plt.hist(dataset, bins=50)
    plt.show()

def excercise2Minuit():
    io = DataIO()
    dataset = io.readNumpy("/datafile-expresonance.txt")

    p0 = {
        "fraction": 1.0,
        "lifetime": 2.0, 
        "mean":     3.0,
    }
    pTrue = {
        "fraction": 0.9,
        "lifetime": 5.0, 
        "mean":     2.5,
    }

    bounds = (0, 10)
    errorDef = 0.5

    pdf = ExponentialResonance(bounds)
    fitter = MinuitFitter(dataset, pdf)
    
    _ = fitter.fit(p0, errorDef, None)

    plt.hist(dataset, bins=50)
    plt.show()

if __name__ == "__main__":

    excercise1Scipy()
    excercise1Minuit()

    excercise2Scipy()
    excercise2Minuit()
