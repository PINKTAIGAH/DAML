#
#   Worled example of Checkpoint: Toy Monte Carlo Studies
#   This shows complete solution for MSc course
#   The solution for NumRep is obtained byy using the base class as pdf.
#  

import math
import numpy
import scipy.integrate as integrate
import scipy.optimize as minimiser
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm

#===============================================
# A set of simple helper functions

# function to make x,y map of a shape function
def mapShape( shape, lolimit, hilimit, steps ):
    y = []
    x = []
    increment = (hilimit-lolimit)/float(steps)
    for i in range( steps ):
        t = lolimit+i*increment
        x.append(t)
        y.append(shape.evaluate(t))
    return x,y

# function to plot a pdf and matching histogram data
def plotShape( data, pdf, lolimit, hilimit, nbins ):
    x,y = mapShape( pdf, lolimit, hilimit, nbins )
    scaleFactor = len(data)/numpy.sum(y)
    y = numpy.multiply( y, scaleFactor )
    plt.plot(x,y)
    plt.hist(data, bins=nbins, range=[lolimit,hilimit])
    plt.show()

# To draw a random sample of N events from a pdf using box method
def drawSample( pdf, lolimit, hilimit, nevents):
    times = []
    for i in range(nevents):
        ythrow = 1.
        yval=0.
        while ythrow > yval:
            tthrow = lolimit + (hilimit-lolimit)* numpy.random.uniform()
            ythrow = pdf.maxVal() * numpy.random.uniform()
            yval =  pdf.evaluate(tthrow)
        times.append(tthrow)
    return times

# To fnd the maximum value of a function
def findMax( func, lolimit, hilimit ):
    max = 0
    grid= 100000
    dt = (hilimit-lolimit)/float(grid)
    for i in range(grid):
        if( func(lolimit+i*dt) > max ) : max = func(lolimit+i*dt)
    return max


def printOptimizeResult( result ):
    values =  result.x
    print('\nScipy Optimise fit results are:')
    fstr = '{0:8.4f}'
    for i in range(len(values)):
        print( '   {:15s}'.format('p'+str(i)), ':  \t', fstr.format(values[i]) )
        #print( " p"+str(i)+" = "+str(values[i]) )


def showIminuitResult( m ):
    #averageList = {}
    valDict = m.values
    errDict = m.errors
    paramNames = list(valDict.keys())
    
    print('\nMinuit fit results are')
    fstr = '{0:8.4f}'
    for pn in paramNames :
        print('   {:15s}'.format(pn), ':  \t', fstr.format(valDict[pn]), ' +/- ', fstr.format(errDict[pn]))
    #averageList.update( { pn : [ valDict[pn], errDict[pn] ] } )
    print('\n')





