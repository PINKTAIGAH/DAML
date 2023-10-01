from MyGaussianPdf import *
import time
import matplotlib.pyplot as plt


#....................................................................
# Function to make x,y map of a shape
# The input "shape" must be an object which implements an "evaluate" method
def mapShape( shape, lolimit, hilimit, steps ):
    y = []
    x = []
    increment = (hilimit-lolimit)/steps
    for i in range( steps ):
        t = lolimit+i*increment
        x.append(t)
        y.append(shape.evaluate(t))
    return x,y


# .....................................................................
# Generate  random numbers using my own generator

mean = 0.0
sigma = 1.0
npoints = 1000
nbins = 50

print("\nPART A")
print("   This is a home made random number generator + analytical overlay")

mygauss = MyGaussianPdf( mean, sigma)

data =[]

for i in range(npoints):
    data.append( next(mygauss) )

#Make a histogram od the data set created
plt.hist(data, bins=nbins)
#plt.savefig('outputHistogram.pdf')
#plt.show()

#Plot the analytic shape
x,y = mapShape( mygauss, mean-3*sigma, mean+3*sigma, nbins )
#Scale it to overlay data
scale  = npoints/nbins * 6*sigma
y = [ scale*i for i in y ]
plt.plot(x,y)
plt.show()

# .....................................................................
# Generate  random numbers using numpy generator.

print("   This is the numpy gaussian number generator")

data = np.random.normal( mean, sigma, npoints )

plt.hist(data, bins=50)
#plt.savefig('outputHistogram.pdf')
plt.show()



# .....................................................................
# Test numerical integration.

integralAnalytic = mygauss.integralAnalytic(-5.0, 5.0 )

print("\nPART B")
print("   Timing different methods of doing integration")

start= time.time()
integralNumeric1 = mygauss.integralNumericBox(-5.0, 5.0 )
end= time.time()
print(("The Box method integral took "+str(end-start)+"  secs"))

start= time.time()
integralNumeric2 = mygauss.integralNumericAvg(-5.0, 5.0 )
end= time.time()
print(("The Average method integral took "+str(end-start)+"  secs"))

start= time.time()
integralNumeric3 = mygauss.integralNumericFaster(-5.0, 5.0 )
end= time.time()
print(("The Vectorised Average method Integral took "+str(end-start)+"  secs"))


print("\nPART C")
print("   Here are all the results  as (value, error) ")

print(" Integrals: ")
print(("  analytic  "+str(integralAnalytic)))
print(("  numericSlow  "+str(integralNumeric1)))
print(("  numericBox   "+str(integralNumeric2)))
print(("  numericCum   "+str(integralNumeric3)))

