
import os, time
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from ..models.torch import DenseVAE, DenseAE

class TorchTrainer(object):
    """
    Class used to train and validate a given model. Currently designed to train VAEs
    """

    def __init__(self, config, dataset,):

        # Define class configuration dictionary 
        self.config = config
        self.epochs = self.config["trainer"]["epochs"]
        self.beta   = self.config["trainer"]["beta"]

        # Set device 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialise datasets
        self.dataset = dataset
        self.initialiseDatasets()

        # Initialise Networks
        self.initialiseNetwork()

        # Initialise optimiser
        self.initialiseOptimiser()
        
        # Initialise loss 
        self.lossFunction = self.denseVAELoss 

        # Initialise metric dictionary
        self.metrics = {
            "val" : {
                "loss" : []
            },
            "train": {
                "loss" : []
            }
        }

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def initialiseNetwork(self,):
        """
        Initialise network objects
        """
        self.network = None
        if self.config["trainer"]["model_name"] == "denseVAE":
            self.network    = DenseVAE(self.config["network"]["input_dims"], self.config["network"]["latent_dims"], self.device,) 
        if self.config["trainer"]["model_name"] == "denseAE":
            self.network    = DenseAE(self.config["network"]["input_dims"], self.config["network"]["latent_dims"]) 
        if self.network is None:
            raise Exception("YOU FUCKED UP")

        # Randomise weights
        self.network.apply(self._init_weights)

    def initialiseOptimiser(self,):
        """
        Initialise optimiser network
        """
        if self.config["trainer"]["optimiser"]["name"] == "adam":
            self.optimiser  = optim.Adam(
               params=self.network.parameters(), lr=self.config["trainer"]["learning_rate"] 
            )
        else: 
            optim.SGD(
                params=self.network.parameters(), lr=self.config["trainer"]["learning_rate"]
            )

    def initialiseDatasets(self,):
        # Define data split fractions
        self.trainData, self.valData = random_split(self.dataset, self.config["trainer"]["train_test_split"])
        # Define train and val loaders
        self.trainLoader = DataLoader(
            self.trainData,
            batch_size = self.config["dataloader"]["batch_size"],
            shuffle = True,
            num_workers = self.config["trainer"]["num_workers"],
        )
        self.valLoader = DataLoader(
            self.valData,
            batch_size = self.config["dataloader"]["batch_size"],
            shuffle = True,
            num_workers = self.config["trainer"]["num_workers"],
        )

    def denseVAELoss(self, xTruth, xGenerated, mean=None, logvar=None):
        """
        Loss function consists of (1-beta)*MSE + beta*KL 
        """
        l2Loss = nn.functional.mse_loss(xTruth, xGenerated)
        # klLoss  = - 0.5 * torch.sum(1+ logvar - mean.pow(2) - logvar.exp())

        # return (1-self.beta)*mseLoss + self.beta*klLoss
        #return (1-self.config["trainer"]["beta"])*l2Loss + self.config["trainer"]["beta"]*klLoss
        return l2Loss 

    def trainStep(self, input):
        """
        Execute a single training step
        """
        # Flush gradients
        self.optimiser.zero_grad()

        if self.config["trainer"]["model_name"] == "denseVAE":
            output, mean, logvar = self.network(input)
            # Compute loss function
            loss = self.lossFunction(input, output, mean, logvar)
        else:
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
            if self.config["trainer"]["model_name"] == "denseVAE":
                output, mean, logvar = self.network(input)
                # Compute loss function
                loss = self.lossFunction(input, output, mean, logvar)
            else:
                output = self.network(input)
                # Compute loss function
                loss = self.lossFunction(input, output)

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

    def getNetwork(self):
        """
        Return network object
        """
        return self.network

    def saveCheckpoint(self):
        """
        Save a copy of the network parameters
        """
        savedir = str(self.config["trainer"]["output_dir"]) + "model_" + str(self.config["trainer"]["run_id"]) + ".pt"
        torch.save(self.network.state_dict(), savedir )

    def runBatchProcess(self,):
        """
        Execute the main training loop
        """

        # Send network to device
        self.network = self.network.to(self.device)
        # Save training start time
        self.trainingStartTime = time.time() 

        # Iterate per epoch
        for epoch in range(self.epochs):
            
            print(f"##### EPOCH {epoch+1} #####")

            # Define variable to hold running loss
            runningTrainLoss    = 0.0
            runningValLoss      = 0.0
            
            # Set network to training mode
            self.network.train()
            # Run train step
            for input in iter(self.trainLoader):
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
            for input in iter(self.valLoader):
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

        # Save model 
        if self.config["trainer"]["save_model"]:
            self.saveCheckpoint()

        # Print out global training time
        print(f"Time taken to train: \t {(time.time()-self.trainingStartTime):.3f} s")