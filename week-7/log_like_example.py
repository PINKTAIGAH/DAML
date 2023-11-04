import numpy as np
import scipy
import matplotlib.pyplot as plt

tau_true = 10

def gendata(N):
    """ Return the times from the exponential distribution """
    return scipy.stats.expon(scale=tau_true).rvs(N)

TOBS = gendata(1000)

plt.hist(TOBS)

def like(p, TOBS):
    """
    return the -loglikelihood summed over all the data
    """
    
    tau = p[0]
    
    # This is the probability density of each data point under the exponential model
    log_pdf = np.log(1 / tau * np.exp(-TOBS / tau))
    
    loglike =  log_pdf.sum()
       
    return -loglike

print ('here are the optimization results')
res = scipy.optimize.minimize(like,[1],args=(TOBS))
print (res)

print ('our best fit tau is' ,res.x, 'the true one is ', tau_true)

plt.show()

