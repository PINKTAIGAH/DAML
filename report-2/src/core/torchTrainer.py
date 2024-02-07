
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from ..models.torch import DenseVAE, DenseAE

class Trainer(object):
    """
    Class used to train and validate a given model. Currently designed to train VAEs
    """

    def __init__(self, config, trainData, valData, saveNetwork=True):

        # Define class configuration dictionary 
        self.config = config
        self.epochs = self.config["trainer"]["epochs"]
        self.beta   = self.config["trainer"]["beta"]

        # Initialise iterable datasets
        self.trainData  = trainData
        self.valData    = valData

        # Initialise Networks
        self.initialiseNetwork()

        # Initialise optimiser
        self.initialiseOptimiser()
        
        # Initialise loss 
        self.lossFunction = self.denseVAELoss if self.config["model_name"] == "denseVAE" else None

        # Set device 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialise metric dictionary
        self.metrics = {
            "val" : {
                "loss" : []
            },
            "train": {
                "loss" : []
            }
        }

    def initialiseNetwork(self,):
        """
        Initialise network objects
        """
        self.network = None
        if self.config["model_name"] == "denseVAE":
            self.network    = DenseVAE(self.config["network"]["input_dims"], self.config["network"]["latent_dims"]) 
        if self.config["model_name"] == "denseAE":
            self.network    = DenseAE(self.config["network"]["input_dims"], self.config["network"]["latent_dims"]) 
        if self.network is None:
            raise Exception("YOU FUCKED UP")

    def initialiseOptimiser(self,):
        """
        Initialise optimiser network
        """
        if self.config["optimiser"]["name"] == "adam":
            self.optimiser  = optim.Adam(
               params=self.network.parameters(), lr=self.config["learning_rate"] 
            )
        else: 
            optim.SGD(
                params=self.network.parameters(), lr=self.config["learning_rate"]
            )

    def denseVAELoss(self, xTruth, xGenerated, mean=None, logvar=None):
        """
        Loss function consists of (1-beta)*MSE + beta*KL 
        """
        l1Loss = nn.L1Loss(xTruth, xGenerated)
        # klLoss  = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        # return (1-self.beta)*mseLoss + self.beta*klLoss
        return l1Loss*100 

    def trainStep(self, input):
        """
        Execute a single training step
        """
        # Flush gradients
        self.optimiser.zero_grad()

        # Perform forward step
        output = self.network(input)
        # Compute loss function
        loss = self.lossFunction(input, output)
        
        # Preform backwards step
        loss.backward()
        self.optimiser.step()

        # print(loss)
        # Return loss value for step
        return loss

    def validationStep(self, input):
        """
        Execute a single training step
        """
        with torch.no_grad():
            # Perform validation step
            # Compute network output
            output = self.network(input)

            # Compute loss function
            loss = self.lossFunction(input, output)
            # print(loss)
        return loss
    
    def printResults(self, epoch, loss, mode):
        """
        Print out results to console through tqdm
        """

        print(f"EPOCH: {epoch} {'#'*5} MODE: {mode} {'#'*5} LOSS: {loss}")    

    def logMetrics(self, trainLoss, valLoss):
        """
        Log metrics to metric dictionary
        """
        # Append Metrics  
        self.metrics["train"]["loss"].append(trainLoss)      
        self.metrics["val"]["loss"].append(valLoss)      

    def getMetrics(self):
        """
        Return metrics object
        """
        return self.metrics
    
    def runBatchProcess(self,):
        """
        Execute the main training loop
        """

        # Send network to device
        self.network = self.network.to(self.device)

        # Iterate per epoch
        for epoch in range(self.epochs):
            
            print(f"##### EPOCH {epoch+1} #####")

            # Define variable to hold running loss
            runningTrainLoss    = 0.0
            runningValLoss      = 0.0
            
            # Set network to training mode
            self.network.train()
            # Run train step
            for input in iter(self.trainData):
                # Send input to device
                input = input.to(self.device)
                # Compute iteration metrics
                runningTrainLoss += self.trainStep(input).mean().item()

            # Compute overall training loss
            trainLoss = runningTrainLoss/len(self.trainData)
            # Print Training step results
            self.printResults(epoch+1, trainLoss, "Train")

            
            # Set model to evaluate mode
            self.network.eval()
            # Run Validation step
            for input in iter(self.valData):
                # Send input to device
                input = input.to(self.device)
                # Compute iteration metics
                runningValLoss += self.validationStep(input).mean().item()

            # Compute overall validation loss
            valLoss = runningValLoss/len(self.valData)
            # Print validation step results
            self.printResults(epoch+1, valLoss, "Validation")
            # Log metrics
            self.logMetrics(trainLoss, valLoss)