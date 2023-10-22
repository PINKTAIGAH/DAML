import numpy as np
import matplotlib.pylab as pl

#let's setup a tolerance:
chi2_tolerance = 0.001
#define the change in chi2 for 1 sigma
delta_chi2_1sigma = 1.0

#we could read the data in, but this is fine for now.
#testdata1
data_x = [1,2,3,4,5,6,7,8,9,10]
data_y = [1.05,0.93,0.99,1.06,0.94,1.08,1.01,0.89,0.95,1.03]
data_yerr = [0.03,0.04,0.02,0.02,0.01,0.03,0.02,0.04,0.025,0.03]
#testdata2
#data_y = [1.05,1.01,1.03,1.02,1.05,1.07,1.06,1.09,2.01,2.03]
#relist because you can't loop over a zipped list twice
zipped_data = list(zip(data_x,data_y,data_yerr))

#let's define the value of the straight line function at a point x
def evaluate_model(m,c,x):
    return m*x+c

#let's define the chi2 value
def evaluate_chi2(m,c):
    chi2_value = 0
    for x,y,err in zipped_data:
        delta = y-evaluate_model(m,c,x)
        chi2_contribution = (delta*delta)/(err*err)
        chi2_value += chi2_contribution
    return chi2_value

#let's set some sensible(?) first values for m and c:
m0 = (data_y[-1]-data_y[0])/(data_x[-1]-data_x[0])
c0 = data_y[-1] - m0*data_x[-1]  

print('Setting initial values:')
print('m:', m0)
print('c:', c0)

#now evaluate chi2 
chi2_0 = evaluate_chi2(m0,c0)
print('chi2 for initial values:',chi2_0)

#let's set a sensible first step for m and c.
ymax = max(data_y)
ymin = min(data_y)
x_ymax = data_x[data_y.index(ymax)]
x_ymin = data_x[data_y.index(ymin)]
mext = (ymax-ymin)/(x_ymax-x_ymin)
cext = ymax - mext*x_ymax  
deltam = (mext-m0)/2
deltac = (cext-c0)/2

print('First step sizes', deltam, deltac)

#get ready to loop!
chi2_current = chi2_0
m_current = m0 + deltam
c_current = c0
chi2_new = evaluate_chi2(m_current,c_current)
chi2_diff = chi2_new - chi2_current
chi2_current = chi2_new

#loop!
keepGoing_m = True
keepGoing_c = True
loopRunning = True
chi2_calls = 2

while(loopRunning):
    chi2_start = chi2_current
        
    while (keepGoing_m):
        #vary m
        if chi2_diff > 0:
            deltam = -0.5*deltam
        m_current = m_current+deltam
        chi2_calls += 1
        chi2_new = evaluate_chi2(m_current,c_current)
        chi2_diff = chi2_new - chi2_current
        chi2_current = chi2_new
        keepGoing_m = abs(chi2_diff) > chi2_tolerance
    if chi2_diff < 0:
        m_current = m_current+deltam

    while (keepGoing_c):
        #vary c
        if chi2_diff > 0:
            deltac = -0.5*deltac
        c_current = c_current+deltac
        chi2_calls += 1
        chi2_new = evaluate_chi2(m_current,c_current)
        chi2_diff = chi2_new - chi2_current
        chi2_current = chi2_new
        keepGoing_c = abs(chi2_diff) > chi2_tolerance
    if chi2_diff < 0:
        c_current = c_current+deltac
    
    chi2_diff_loop = chi2_current - chi2_start
    keepGoing_m = True
    keepGoing_c = True
    loopRunning = abs(chi2_diff_loop) > chi2_tolerance
    
m_best = m_current
c_best = c_current
chi2_min = chi2_current

#get the uncertainty on m:
m_unc = 1.0*deltam
m_unc_deltachi2 = 0
while m_unc_deltachi2 < delta_chi2_1sigma:
    m_unc_chi2 = evaluate_chi2(m_best+m_unc,c_best)
    m_unc_deltachi2 = m_unc_chi2-chi2_min
    m_unc+=deltam

#get the uncertainty on c:
c_unc = 1.0*deltac
c_unc_deltachi2 = 0
while c_unc_deltachi2 < delta_chi2_1sigma:
    c_unc_chi2 = evaluate_chi2(m_best,c_best+c_unc)
    c_unc_deltachi2 = c_unc_chi2-chi2_min
    c_unc+=deltam

print('Number Evaluations for central values', chi2_calls)
print('Final Step Sizes:',deltam,deltac)
print('Best fit m', m_best,'+/-',m_unc)
print('Best fit c', c_best,'+/-',c_unc)
print('Minimum chi2:', chi2_min)

#plot to check the result is reasonable
model_y = []
for x in data_x:
    model_y.append(evaluate_model(m_current,c_current,x))
    
import matplotlib.pylab as pl
plot_x = np.array(data_x)
plot_y = np.array(data_y)
plot_modely = np.array(model_y)
plot_yerr = np.array(data_yerr)
pl.figure(0)
pl.errorbar(x=plot_x,y=plot_y,yerr = plot_yerr,c='blue')
pl.errorbar(x=plot_x,y=plot_modely,c='red')
pl.savefig('scatter.pdf')

