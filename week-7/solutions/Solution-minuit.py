#
#   Worled example of Checkpoint: Toy Monte Carlo Studies
#   This shows complete solution for MSc course
#   The solution for NumRep is obtained byy using the base class as pdf.
#  

import math
import numpy
#import scipy.integrate as integrate
#import scipy.optimize as minimiser
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
from MyHelperModule import *

from iminuit import Minuit



#===============================================
# To calculate NLL for optimize
class NegativeLLcalculator:
    
    def __init__( self, pdf, data ):
        self.pdf = pdf
        self.data = data
 
    # To calcualte an NLL frm a dataset and a pdf
    # This format is suitable for use by optimize
    def evaluate( self, params ):
        nll = 0.
        self.pdf.setParameters(params)
        #print params
        for i in range(len(self.data)):
            prob = self.pdf.evaluate(self.data[i])
            if prob <=0 : prob = 0.00000001
            logprob = math.log(prob)
            nll+= logprob
        return -nll

    def updateData( self, data):
        self.data = data

#===============================================
# Exponential pdf
# Base class implementing many methods which all depend only upon self.shape
# self.shape is set in the constructor and can be over-ridden to make more complex pdfs
class Exponential:
    
    # Constructor
    def __init__(self, lolim, hilim, params  ):
        self.lolimit = lolim
        self.hilimit = hilim
        self.setParameters( params )
        self.max = findMax( self.evaluate, self.lolimit, self.hilimit )
    
    # Set parameters
    def setParameters(self, params ):
        self.lifetime = params[0]
    
     # Returns max value of function
    def maxVal( self ) :
        return self.max

    # Evaluate method (un-normalised)
    def evaluate( self, t ):
        part1 = 1./abs(self.lifetime) * math.exp(-t/self.lifetime)
        norm1 = math.exp(-self.lolimit/self.lifetime) - math.exp(-self.hilimit/self.lifetime)
        return part1/norm1
 
 
    # Draw N random number from distribution
    def next(self, nevents):
        data  = drawSample( self, self.lolimit, self.hilimit, nevents)
        return data




#===============================================
# Exponential pdf
class ExponentialWithResonance:
    
    # Constructor
    def __init__(self, lolim, hilim, params  ):
        self.lolimit = lolim
        self.hilimit = hilim
        self.setParameters( params )
        self.max = findMax( self.evaluate, self.lolimit, self.hilimit )

    # Set parameters
    def setParameters(self, params ):
        self.lifetime = params[0]
        self.fraction  = params[1]
        self.t0 = params[2]
        if len(params) == 4:
           self.sigma = params[3]
        else:
           self.sigma = 0.2
  
    # Evaluate method (un-normalised)
    def evaluate( self, t ):
        part1 = 1./abs(self.lifetime) * math.exp(-t/self.lifetime)
        norm1 = math.exp(-self.lolimit/self.lifetime) - math.exp(-self.hilimit/self.lifetime)
        part2 = math.exp( -(t-self.t0)**2 / (2.0 * self.sigma**2 ) ) / (self.sigma*math.sqrt(2.0*math.pi ))
        return self.fraction* part1/norm1 + (1-self.fraction)*part2


    # Returns max value of function
    def maxVal( self ) :
        return self.max
    
    
    # Draw N random number from distribution
    def next(self, nevents):
        data  = drawSample( self, self.lolimit, self.hilimit, nevents)
        return data



#===============================================
# To generate and fit to exponential with flat bakground

def Gen0( nevents):
    
    lolimit = 0.
    hilimit = 10.
    lifetime = 2.2  #main exponential lifetime
    
    #Create the pdf
    paramsin = [lifetime]
    pdf = Exponential( lolimit, hilimit, paramsin )
    
    #Generate a single experiment
    data = pdf.next( nevents)
    #Plot function and data
    #plotShape( data, pdf, lolimit, hilimit, 100 )
    #Write it out
    numpy.savetxt( "datafile-exp.txt", data)

#===============================================
# To fit to simple exponential

def Fit0():
    
    lolimit = 0.
    hilimit = 10.
    lifetime = 2.2  #main exponential lifetime
    
    #Create the pdf
    paramsin = [lifetime ]
    pdf = Exponential( lolimit, hilimit, paramsin )
    
    #Get data
    data = numpy.loadtxt( "datafile-exp.txt")
    plotShape( data, pdf, lolimit, hilimit, 100 )

    #Create an NLL calculator
    nllcalc = NegativeLLcalculator( pdf, data )
  
    #Set start values
    svals = {'lifetime':lifetime}

    #Create Minuit minimiser
    m = Minuit(nllcalc.evaluate, use_array_call=True, forced_parameters=["lifetime"], errordef=0.5, **svals)
    
    #Fit for best parameters
    mresult = m.migrad()
    showIminuitResult(m)

#===============================================
# To generate exponential plus resonance
def Gen1(nevents):
    
    lolimit = 0.
    hilimit = 10.
    lifetime = 5.0  #main exponential lifetime
    fraction = 0.9
    mean = 2.5
    #sigma = 0.2
    
    #Create the pdf
    paramsin = [lifetime, fraction, mean]
    pdf = ExponentialWithResonance( lolimit, hilimit, paramsin )
    
    #Generate a single experiment
    data = pdf.next( nevents)
    #Plot function and data
    #plotShape( data, pdf, lolimit, hilimit, 100 )
    #Write it out
    numpy.savetxt( "datafile-expresonance.txt", data)


#===============================================
# To fit to exponential plus resonance

def Fit1( ):

    lolimit = 0.
    hilimit = 10.
    lifetime = 5.0  #main exponential lifetime
    fraction = 0.9
    mean = 2.5
    #sigma = 0.2
  
    #Create the pdf
    paramsin = [lifetime, fraction, mean]
    pdf = ExponentialWithResonance( lolimit, hilimit, paramsin )

    #get the data
    data = numpy.loadtxt( "datafile-expresonance.txt")
    plotShape( data, pdf, lolimit, hilimit, 100 )

    #Create an NLL calculator
    nllcalc = NegativeLLcalculator( pdf, data )
  
    #Set start values
    svals = {}
    svals.update({'lifetime':lifetime})
    svals.update({'fraction':fraction})
    svals.update({'mean':mean})

    #Create minimiser
    m = Minuit(nllcalc.evaluate,  \
               use_array_call=True, forced_parameters=['lifetime', 'fraction','mean'], errordef=0.5 , \
               limit_lifetime=[0.2,20.], limit_fraction=[0.,1.], limit_mean=[0.1,10.], **svals )

    #Fit for best parameters
    mresult = m.migrad()
    showIminuitResult(m)

#===============================================
#Main

def main():
 
    # Generate and fit the simple exponential
    Gen0(10000)
    Fit0()

    # Generate and fit the exponential + gaussian background
    #Gen1(10000)
    #Fit1()



main()




  
