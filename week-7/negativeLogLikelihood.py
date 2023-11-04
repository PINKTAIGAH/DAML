import numpy as np


class NegativeLogLikelihood(object):

    def __init__(self, pdf, data, usingScipy=False):

        self.pdf = pdf
        self.data = data
        self.usingScipy = usingScipy

    def evaluate(self, *fittingParameters):

        if self.usingScipy:
            fittingParameters = fittingParameters[0]

        self.pdf.setParameters(fittingParameters) 
        likelihood = self.pdf.evaluate(self.data)
        if (likelihood <= 0).any():
            likelihood[likelihood<=0] = 0.00000001
        logLikelihood = np.log(likelihood)
        return -logLikelihood.sum()
