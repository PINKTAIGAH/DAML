import numpy as np

class Minimiser(object):

    def __init__(
            self, function, x, y, yError, bounds,
            p0=None, tolerance=None, learningRate=None    
        ):

        self.function = function
        self.inputs =  np.vstack([x, y, yError])
        self.bounds = bounds
        self.parameters = self._defineP0() if p0 is None else p0
        self.tolerance = 0.1 if tolerance is None else tolerance
        
        self.learningRate = self._estimateInitialLearningRate() if learningRate is None else learningRate
        self.initialValueOfFunction = function(*self.inputs, *self.parameters) 
    
    def _findInitialP0(self, steps=10):
        paramGrid = np.array([ np.linspace(self.bounds[idx][0], self.bounds[idx][1], steps) for idx in range(len(self.bounds)) ])
        bestParameters = None
        bestValue = None
        for i in range(paramGrid[0].size):
            for j in range(paramGrid[1].size):
                trialParameter = (i, j)
                currentValue = self.funtion(*self.inputs, *trialParameter)

                if currentValue < bestValue:
                    bestValue = currentValue
                    bestParameters = trialParameter
        
        return bestParameters
    
    def _estimateInitialLearningRate(self,):
        return np.max( (self.bounds - self.bounds)/10 )
        
    def _isFinished(self, previousValueOfFunction):

        currentMinimum = function(*self.inputs, *self.parameters)
        difference = abs(currentMinimum-previousValueOfFunction)
        
        if difference <= self.tolerance:
            return True
        
        return False

    def _minimiseIndividualParameter(self, idx):
        # Idx is index of parameter currently being minimised

        direction = +1       # +1 will be to the right, -1 will be to the left 
        previousValueOfFunction = self.initialValueOfFunction
        while (not self._isFinished(previousValueOfFunction)):
            
            valueOfFunction = function(*self.inputs, *self.parameters)

            if valueOfFunction > previousValueOfFunction:
                direction *= -1
                self.learningRate /= 2

            previousValueOfFunction = valueOfFunction

            self.parameter[idx] = self.parameter[idx] + direction*self.learningRate
        
    def minimise(self,):

        previousValueOfFunction = self.initialValueOfFunction
        while (not self._isFinished(previousValueOfFunction)):
            for idx in range(len(self.parameters())):
                self._minimiseIndividualParameter(idx)

                previousValueOfFunction = function(*self.inputs, *self.parameters)

        return self.parameters


def function(x, y, yErr, gradient, intercept):
    yPredicted = gradient*x + intercept
    return np.sum((y - yPredicted)**2/yErr**2)

def test():
    pass   

               


    


    